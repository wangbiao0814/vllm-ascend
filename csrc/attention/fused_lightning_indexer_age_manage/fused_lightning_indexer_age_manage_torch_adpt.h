#ifndef FUSED_LIGHTNING_INDEXER_AGE_MANAGE_TORCH_ADPT_H_
#define FUSED_LIGHTNING_INDEXER_AGE_MANAGE_TORCH_ADPT_H_

#include "../fused_lightning_indexer_manage/op_kernel/fused_lightning_indexer_manage_constants.h"

#include "op_kernel/age_policy.h"

namespace vllm_ascend {

inline void npu_fused_lightning_indexer_age_manage(
    const at::Tensor& index_weights, const at::Tensor& query_dequant_scale,
    const at::Tensor& query, const at::Tensor& index_key_dequant_scale,
    const at::Tensor& index_key_cache, const at::Tensor& index_block_table,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_key,
    const at::Tensor& offload_seq_lengths_key,
    const at::Tensor& num_cache_tokens, const at::Tensor& request_state,
    const at::Tensor& req_pool_entries, at::Tensor cache_slots_pool,
    at::Tensor topk_src_ids, at::Tensor topk_dst_slots,
    at::Tensor topk_miss_counts, at::Tensor miss_src_ids,
    at::Tensor miss_dst_slots, at::Tensor miss_counts,
    at::Tensor source_ages_pool, int64_t probation_age, int64_t diagnostic_flags) {
  TORCH_CHECK(query.size(0) == index_block_table.size(0), "Age LIM supports MTP=0 only (T=B).");
  TORCH_CHECK(source_ages_pool.sizes() == cache_slots_pool.sizes() &&
              source_ages_pool.scalar_type() == at::kInt &&
              source_ages_pool.device() == query.device() && source_ages_pool.is_contiguous(),
              "source_ages_pool must match cache_slots_pool and be contiguous int32 on NPU.");
  TORCH_CHECK(probation_age >= 0 && probation_age <= LIMAge::MAX_AGE,
              "probation_age exceeds the supported age range.");
  TORCH_CHECK(diagnostic_flags >= 0 && diagnostic_flags <= LIMAge::DIAGNOSTIC_MASK,
              "diagnostic_flags must be in [0,127]; nonzero is for ablation only.");
  constexpr int64_t kTopK = LIMConfig::TOPK;
  constexpr int64_t kMissCapacity = LIMConfig::MISS_CAPACITY;
  constexpr int64_t kBlockSize = LIMConfig::CACHE_BLOCK_SIZE;
  TORCH_CHECK(query.dim() == 3 &&
                  (query.size(1) == LIMConfig::QUERY_HEADS_SMALL ||
                   query.size(1) == LIMConfig::QUERY_HEADS_LARGE) &&
                  query.size(2) == LIMConfig::HEAD_DIM && query.size(0) > 0,
              "LIM-MTP query must be [T, H, 128], H=32 or 64 and T>0.");
  TORCH_CHECK(query.device().is_privateuseone(), "LIM-MTP tensors must be on NPU.");
  const int64_t total_queries = query.size(0);
  TORCH_CHECK(index_weights.dim() == 2 && index_weights.size(0) == total_queries &&
                  index_weights.size(1) == query.size(1),
              "index_weights must be [T, H].");
  TORCH_CHECK(query_dequant_scale.sizes() == index_weights.sizes(),
              "query_dequant_scale must be [T, H].");
  TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 &&
                  index_key_cache.size(1) == kBlockSize &&
                  index_key_cache.size(2) == LIMConfig::KEY_HEADS &&
                  index_key_cache.size(3) == LIMConfig::HEAD_DIM,
              "index_key_cache must be [blocks, 128, 1, 128].");
  TORCH_CHECK(index_key_dequant_scale.dim() == 3 &&
                  index_key_dequant_scale.size(0) == index_key_cache.size(0) &&
                  index_key_dequant_scale.size(1) == kBlockSize &&
                  index_key_dequant_scale.size(2) == LIMConfig::KEY_HEADS,
              "index_key_dequant_scale must be [blocks, 128, 1].");
  TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) > 0 &&
                  index_block_table.size(1) > 0 &&
                  index_block_table.size(1) <= LIMConfig::MAX_BLOCKS_PER_REQUEST,
              "index_block_table must be non-empty [B, max_blocks], max_blocks<=16384.");
  const int64_t batch_size = index_block_table.size(0);
  TORCH_CHECK(total_queries >= batch_size &&
                  total_queries <= batch_size * LIMConfig::MAX_ROUTES,
              "total_queries must satisfy B <= T <= 14B.");
  auto check_batch_vector = [batch_size](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch_size, name, " must be [B].");
  };
  check_batch_vector(actual_seq_lengths_query, "actual_seq_lengths_query");
  check_batch_vector(actual_seq_lengths_key, "actual_seq_lengths_key");
  check_batch_vector(offload_seq_lengths_key, "offload_seq_lengths_key");
  check_batch_vector(num_cache_tokens, "num_cache_tokens");
  check_batch_vector(request_state, "request_state");
  check_batch_vector(req_pool_entries, "req_pool_entries");
  TORCH_CHECK(cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 &&
                  cache_slots_pool.size(1) == index_block_table.size(1) * kBlockSize,
              "cache_slots_pool width must equal max_blocks*128.");
  TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(0) == total_queries &&
                  topk_src_ids.size(1) == 1 && topk_src_ids.size(2) == kTopK &&
                  topk_dst_slots.sizes() == topk_src_ids.sizes(),
              "topk outputs must be [T, 1, 2048].");
  TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries,
              "topk_miss_counts must be [T].");
  TORCH_CHECK(miss_src_ids.dim() == 2 && miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kMissCapacity &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes(),
              "miss outputs must be [B, 32768].");
  check_batch_vector(miss_counts, "miss_counts");

  TORCH_CHECK(query.scalar_type() == index_key_cache.scalar_type() &&
                  query.scalar_type() == index_weights.scalar_type() &&
                  (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16),
              "query/index_key_cache/index_weights must share fp16 or bf16 dtype.");
  TORCH_CHECK(query_dequant_scale.scalar_type() == at::kFloat &&
                  index_key_dequant_scale.scalar_type() == at::kFloat,
              "dequant scales must be fp32.");
  const at::Tensor int_tensors[] = {
      index_block_table, actual_seq_lengths_query, actual_seq_lengths_key,
      offload_seq_lengths_key, num_cache_tokens, request_state, req_pool_entries,
      cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts,
      miss_src_ids, miss_dst_slots, miss_counts};
  for (const auto& tensor : int_tensors) {
    TORCH_CHECK(tensor.scalar_type() == at::kInt, "metadata and outputs must be int32.");
  }
  const at::Tensor all_tensors[] = {
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots, miss_counts};
  const auto device = query.device();
  for (const auto& tensor : all_tensors) {
    TORCH_CHECK(tensor.device() == device, "all LIM-MTP tensors must be on the same NPU.");
    TORCH_CHECK(tensor.is_contiguous(), "all LIM-MTP tensors must be contiguous.");
  }

  EXEC_NPU_CMD(
      aclnnFusedLightningIndexerAgeManageV2,
      index_weights, query_dequant_scale, query, index_key_dequant_scale,
      index_key_cache, index_block_table, actual_seq_lengths_query,
      actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens,
      request_state, req_pool_entries, cache_slots_pool, source_ages_pool, probation_age, diagnostic_flags, topk_src_ids,
      topk_dst_slots, topk_miss_counts, miss_src_ids, miss_dst_slots,
      miss_counts, cache_slots_pool, source_ages_pool);
}

}  // namespace vllm_ascend
#endif
