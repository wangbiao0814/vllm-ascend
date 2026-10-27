#include <torch/extension.h>
#include <limits>
#include <optional>
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ATen/Parallel.h>
#include <torch/script.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <omp.h>
#include <assert.h>
#include <torch/torch.h>
#include <pthread.h>

#include "lru_resident.h"

int32_t compute_lru_resident_addrs(const at::Tensor& miss_count, const at::Tensor& miss_tokens,
                                   const at::Tensor& miss_slots, const at::Tensor& block_table,
                                   const int32_t block_size, const int32_t token_size_bytes_k,
                                   const int32_t token_size_bytes_v, const int64_t gvas_k_base,
                                   const int64_t gvas_v_base, const int64_t addr_k_base, const int64_t addr_v_base,
                                   const int32_t resident_capacity, const int32_t max_num_threads,
                                   at::Tensor& gvas_buffer, at::Tensor& addr_buffer, at::Tensor& size_buffer,
                                   at::Tensor& num_tokens_buffer) {
  const int32_t num_reqs = miss_tokens.size(0);
  const int32_t topk = miss_tokens.size(1);
  const int32_t max_num_tokens_to_load = num_reqs * topk;
  const int32_t max_num_blocks = block_table.size(1);
  const int32_t block_size_bytes_k = block_size * token_size_bytes_k;
  const int32_t block_size_bytes_v = block_size * token_size_bytes_v;
  TORCH_CHECK(miss_count.size(0) >= num_reqs, "miss_count size not enough for loading.");
  TORCH_CHECK(miss_slots.size(0) == num_reqs && miss_slots.size(1) == topk,
              "miss_slots shape should match miss_tokens.");
  TORCH_CHECK(gvas_buffer.size(0) >= max_num_tokens_to_load * 2, "gvas_buffer size not enough for loading.");
  TORCH_CHECK(addr_buffer.size(0) >= max_num_tokens_to_load * 2, "addr_buffer size not enough for loading.");
  TORCH_CHECK(size_buffer.size(0) >= max_num_tokens_to_load * 2, "size_buffer size not enough for loading.");
  TORCH_CHECK(miss_count.scalar_type() == at::kInt, "miss_count wrong dtype, should be int32.");
  TORCH_CHECK(miss_tokens.scalar_type() == at::kInt, "miss_tokens wrong dtype, should be int32.");
  TORCH_CHECK(miss_slots.scalar_type() == at::kInt, "miss_slots wrong dtype, should be int32.");
  TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table wrong dtype, should be int32.");
  TORCH_CHECK(gvas_buffer.scalar_type() == at::kLong, "gvas_buffer wrong dtype, should be int64.");
  TORCH_CHECK(addr_buffer.scalar_type() == at::kLong, "addr_buffer wrong dtype, should be int64.");
  TORCH_CHECK(size_buffer.scalar_type() == at::kInt, "size_buffer wrong dtype, should be int32.");

  const int32_t* miss_count_ptr = static_cast<int32_t*>(miss_count.data_ptr());
  const int32_t* miss_tokens_ptr = static_cast<int32_t*>(miss_tokens.data_ptr());
  const int32_t* miss_slots_ptr = static_cast<int32_t*>(miss_slots.data_ptr());
  const int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());
  int64_t* gvas_buffer_ptr = static_cast<int64_t*>(gvas_buffer.data_ptr());
  int64_t* addr_buffer_ptr = static_cast<int64_t*>(addr_buffer.data_ptr());
  int32_t* size_buffer_ptr = static_cast<int32_t*>(size_buffer.data_ptr());
  int32_t* num_tokens_buffer_ptr = static_cast<int32_t*>(num_tokens_buffer.data_ptr());

  int n_threads = std::min(num_reqs, max_num_threads);
  n_threads = std::max(n_threads, 1);

  std::vector<int32_t> num_tokens_to_load_req(num_reqs);
#pragma omp parallel for num_threads(n_threads)
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t count = miss_count_ptr[req_idx];
    count = std::max(count, 0);
    count = std::min(count, topk);
    num_tokens_to_load_req[req_idx] = count;
  }
  int32_t num_tokens_to_load_sum = std::accumulate(num_tokens_to_load_req.begin(), num_tokens_to_load_req.end(), 0);
  std::vector<int32_t> req_start_locs(num_reqs);
  int32_t cumsum = 0;
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    req_start_locs[req_idx] = cumsum;
    cumsum += num_tokens_to_load_req[req_idx];
  }

#pragma omp parallel for num_threads(n_threads)
  for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    int32_t req_start_loc_k = req_start_locs[req_idx];
    int32_t req_start_loc_v = num_tokens_to_load_sum + req_start_loc_k;
    for (int32_t idx = 0; idx < num_tokens_to_load_req[req_idx]; ++idx) {
      const int32_t token = miss_tokens_ptr[req_idx * topk + idx];
      const int32_t slot = miss_slots_ptr[req_idx * topk + idx];
      if (token < 0 || slot < 0 || slot >= resident_capacity) {
        continue;
      }
      const int32_t block_id = token / block_size;
      if (block_id < 0 || block_id >= max_num_blocks) {
        continue;
      }
      const int32_t offset_in_block = token % block_size;
      const int32_t block_indice = block_table_ptr[req_idx * max_num_blocks + block_id];
      if (block_indice < 0) {
        continue;
      }
      const int64_t gvas_k = gvas_k_base + block_indice * block_size_bytes_k + offset_in_block * token_size_bytes_k;
      const int64_t gvas_v = gvas_v_base + block_indice * block_size_bytes_v + offset_in_block * token_size_bytes_v;
      const int64_t addr_k = addr_k_base + (req_idx * resident_capacity + slot) * token_size_bytes_k;
      const int64_t addr_v = addr_v_base + (req_idx * resident_capacity + slot) * token_size_bytes_v;
      gvas_buffer_ptr[req_start_loc_k + idx] = gvas_k;
      gvas_buffer_ptr[req_start_loc_v + idx] = gvas_v;
      addr_buffer_ptr[req_start_loc_k + idx] = addr_k;
      addr_buffer_ptr[req_start_loc_v + idx] = addr_v;
    }
  }

  std::fill_n(size_buffer_ptr, num_tokens_to_load_sum, token_size_bytes_k);
  std::fill_n(&(size_buffer_ptr[num_tokens_to_load_sum]), num_tokens_to_load_sum, token_size_bytes_v);

  num_tokens_buffer_ptr[0] = num_tokens_to_load_sum * 2;
  return num_tokens_to_load_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  m.def("lru_resident_compact", &lru_resident_compact,
        "CPU LRU resident compact miss prepare with OpenMP row-level parallelism");
  m.def("reuse_prioritized_lru_resident_compact", &reuse_prioritized_lru_resident_compact,
        "CPU reuse-prioritized LRU resident miss prepare with probation ages");
  m.def("compute_lru_resident_addrs", &compute_lru_resident_addrs,
        "Compute sparse H2D metadata for compact LRU resident miss loads");
}
