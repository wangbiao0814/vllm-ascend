// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Keep the CPU planner independent of torch/ACL so its production implementation
// can also be compiled and tested without an NPU runtime.
inline int lru_resident_thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
  #define HOT_FUNCTION __attribute__((hot))
  #define FORCE_INLINE inline __attribute__((always_inline))
  #define LIKELY(x) __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define RESTRICT __restrict__
#else
  #define HOT_FUNCTION
  #define FORCE_INLINE inline
  #define LIKELY(x) (x)
  #define UNLIKELY(x) (x)
  #define RESTRICT
#endif

constexpr int32_t EPOCH_RESET_THRESHOLD = 1 << 30;

FORCE_INLINE int choose_lru_resident_threads(const int num_reqs, const int workspace_threads,
                                             const int requested_threads) {
#ifndef _OPENMP
  return 1;
#endif
  if (num_reqs <= 1) {
    return 1;
  }
  int active_threads = num_reqs;
  if (workspace_threads > 0) {
    active_threads = std::min(active_threads, workspace_threads);
  }
  if (requested_threads > 0) {
    active_threads = std::min(active_threads, requested_threads);
  }
  return std::max(active_threads, 1);
}

FORCE_INLINE bool is_valid_lru_resident_token(const int32_t token, const int32_t max_token) {
  return token >= 0 && token < max_token;
}

FORCE_INLINE void reset_lru_resident_row(int32_t* RESTRICT slot_to_token_row, int32_t* RESTRICT lru_slots_row,
                                         const int32_t capacity) {
  std::fill(slot_to_token_row, slot_to_token_row + capacity, -1);
  for (int32_t slot = 0; slot < capacity; ++slot) {
    lru_slots_row[slot] = slot;
  }
}

FORCE_INLINE int32_t next_lru_resident_epoch(int32_t* RESTRICT token_mark, int32_t* RESTRICT token_pos,
                                             int32_t* RESTRICT epoch, const int32_t max_token) {
  int32_t base = epoch[0] + 1;
  if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
    std::fill(token_mark, token_mark + max_token, 0);
    std::fill(token_pos, token_pos + max_token, -1);
    base = 1;
  }
  epoch[0] = base;
  return base;
}

FORCE_INLINE void process_one_lru_resident_row(
    const int row, const int32_t topk, const int32_t capacity, const int32_t max_token, const int64_t* RESTRICT req_ids,
    int64_t* RESTRICT last_req_ids, const int32_t* RESTRICT topk_indices, const int32_t* RESTRICT stable_prefix_lens,
    int32_t* RESTRICT slot_to_token, int32_t* RESTRICT lru_slots, int32_t* RESTRICT current_slots,
    int32_t* RESTRICT miss_count, int32_t* RESTRICT miss_tokens_out, int32_t* RESTRICT miss_slots_out,
    int32_t* RESTRICT token_mark, int32_t* RESTRICT token_pos, int32_t* RESTRICT slot_workspace,
    int32_t* RESTRICT miss_positions, int32_t* RESTRICT epoch) {
  int32_t* RESTRICT slot_to_token_row = slot_to_token + static_cast<int64_t>(row) * capacity;
  int32_t* RESTRICT lru_slots_row = lru_slots + static_cast<int64_t>(row) * capacity;
  int32_t* RESTRICT current_slots_row = current_slots + static_cast<int64_t>(row) * topk;
  int32_t* RESTRICT miss_tokens_row = miss_tokens_out + static_cast<int64_t>(row) * topk;
  int32_t* RESTRICT miss_slots_row = miss_slots_out + static_cast<int64_t>(row) * topk;
  const int32_t* RESTRICT topk_row = topk_indices + static_cast<int64_t>(row) * topk;
  int32_t* RESTRICT hit_slots = slot_workspace;
  int32_t* RESTRICT evictable_slots = slot_workspace + capacity;
  int32_t* RESTRICT assigned_miss_slots = slot_workspace + static_cast<int64_t>(capacity) * 2;

  std::fill(current_slots_row, current_slots_row + topk, -1);
  std::fill(miss_tokens_row, miss_tokens_row + topk, -1);
  std::fill(miss_slots_row, miss_slots_row + topk, -1);
  miss_count[row] = 0;

  const int64_t req_id = req_ids[row];
  if (UNLIKELY(last_req_ids[row] != req_id)) {
    reset_lru_resident_row(slot_to_token_row, lru_slots_row, capacity);
    last_req_ids[row] = req_id;
  }
  const int32_t stable_prefix_len = std::clamp(stable_prefix_lens[row], 0, max_token);

  const int32_t base = next_lru_resident_epoch(token_mark, token_pos, epoch, max_token);
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = topk_row[pos];
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] != base) {
      token_mark[token] = base;
      token_pos[token] = pos;
    }
  }

  int32_t hit_count = 0;
  int32_t evictable_count = 0;
  for (int32_t order = 0; order < capacity; ++order) {
    const int32_t slot = lru_slots_row[order];
    if (UNLIKELY(slot < 0 || slot >= capacity)) {
      continue;
    }
    int32_t token = slot_to_token_row[slot];
    // The speculative suffix may have been overwritten in the CPU KV pool.
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && UNLIKELY(token >= stable_prefix_len)) {
      slot_to_token_row[slot] = -1;
      token = -1;
    }
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] == base) {
      const int32_t pos = token_pos[token];
      current_slots_row[pos] = slot;
      hit_slots[hit_count] = slot;
      ++hit_count;
    } else {
      evictable_slots[evictable_count] = slot;
      ++evictable_count;
    }
  }

  int32_t local_miss_count = 0;
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = topk_row[pos];
    if (LIKELY(is_valid_lru_resident_token(token, max_token)) && current_slots_row[pos] < 0) {
      miss_tokens_row[local_miss_count] = token;
      miss_positions[local_miss_count] = pos;
      ++local_miss_count;
    }
  }

  const int32_t assign_count = std::min(local_miss_count, evictable_count);
  for (int32_t miss_idx = 0; miss_idx < assign_count; ++miss_idx) {
    const int32_t slot = evictable_slots[miss_idx];
    const int32_t token = miss_tokens_row[miss_idx];
    const int32_t pos = miss_positions[miss_idx];
    slot_to_token_row[slot] = token;
    current_slots_row[pos] = slot;
    miss_slots_row[miss_idx] = slot;
    assigned_miss_slots[miss_idx] = slot;
    miss_count[row] = miss_idx + 1;
  }

  int32_t write_pos = 0;
  for (int32_t idx = assign_count; idx < evictable_count; ++idx) {
    lru_slots_row[write_pos] = evictable_slots[idx];
    ++write_pos;
  }
  for (int32_t idx = 0; idx < assign_count; ++idx) {
    lru_slots_row[write_pos] = assigned_miss_slots[idx];
    ++write_pos;
  }
  for (int32_t idx = 0; idx < hit_count; ++idx) {
    lru_slots_row[write_pos] = hit_slots[idx];
    ++write_pos;
  }
}

HOT_FUNCTION inline void lru_resident_compact(uintptr_t req_ids_ptr, uintptr_t last_req_ids_ptr, uintptr_t topk_indices_ptr,
                                       uintptr_t stable_prefix_lens_ptr, uintptr_t slot_to_token_ptr,
                                       uintptr_t lru_slots_ptr, uintptr_t current_slots_ptr, uintptr_t miss_count_ptr,
                                       uintptr_t miss_tokens_ptr, uintptr_t miss_slots_ptr,
                                       uintptr_t token_mark_workspace_ptr, uintptr_t token_pos_workspace_ptr,
                                       uintptr_t slot_workspace_ptr, uintptr_t miss_position_workspace_ptr,
                                       uintptr_t epochs_ptr, int64_t num_reqs, int64_t topk, int64_t capacity,
                                       int64_t max_token, int64_t workspace_threads, int64_t requested_threads) {
  if (num_reqs <= 0 || topk <= 0 || capacity <= 0 || max_token <= 0) {
    return;
  }

  auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
  auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
  auto* RESTRICT topk_indices = reinterpret_cast<int32_t*>(topk_indices_ptr);
  auto* RESTRICT stable_prefix_lens = reinterpret_cast<int32_t*>(stable_prefix_lens_ptr);
  auto* RESTRICT slot_to_token = reinterpret_cast<int32_t*>(slot_to_token_ptr);
  auto* RESTRICT lru_slots = reinterpret_cast<int32_t*>(lru_slots_ptr);
  auto* RESTRICT current_slots = reinterpret_cast<int32_t*>(current_slots_ptr);
  auto* RESTRICT miss_count = reinterpret_cast<int32_t*>(miss_count_ptr);
  auto* RESTRICT miss_tokens = reinterpret_cast<int32_t*>(miss_tokens_ptr);
  auto* RESTRICT miss_slots = reinterpret_cast<int32_t*>(miss_slots_ptr);
  auto* RESTRICT token_mark_workspace = reinterpret_cast<int32_t*>(token_mark_workspace_ptr);
  auto* RESTRICT token_pos_workspace = reinterpret_cast<int32_t*>(token_pos_workspace_ptr);
  auto* RESTRICT slot_workspace = reinterpret_cast<int32_t*>(slot_workspace_ptr);
  auto* RESTRICT miss_position_workspace = reinterpret_cast<int32_t*>(miss_position_workspace_ptr);
  auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);

  const int num_reqs_int = static_cast<int>(num_reqs);
  const int topk_int = static_cast<int>(topk);
  const int capacity_int = static_cast<int>(capacity);
  const int max_token_int = static_cast<int>(max_token);
  const int active_threads = choose_lru_resident_threads(num_reqs_int, static_cast<int>(workspace_threads),
                                                         static_cast<int>(requested_threads));

  if (active_threads == 1) {
    for (int row = 0; row < num_reqs_int; ++row) {
      process_one_lru_resident_row(row, topk_int, capacity_int, max_token_int, req_ids, last_req_ids, topk_indices,
                                   stable_prefix_lens, slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
                                   miss_slots, token_mark_workspace, token_pos_workspace, slot_workspace,
                                   miss_position_workspace, epochs);
    }
    return;
  }

#pragma omp parallel num_threads(active_threads)
  {
    const int thread_id = lru_resident_thread_id();
    int32_t* RESTRICT token_mark = token_mark_workspace + static_cast<int64_t>(thread_id) * max_token_int;
    int32_t* RESTRICT token_pos = token_pos_workspace + static_cast<int64_t>(thread_id) * max_token_int;
    int32_t* RESTRICT slots = slot_workspace + static_cast<int64_t>(thread_id) * capacity_int * 3;
    int32_t* RESTRICT miss_positions = miss_position_workspace + static_cast<int64_t>(thread_id) * topk_int;
    int32_t* RESTRICT epoch = epochs + thread_id;
    for (int row = thread_id; row < num_reqs_int; row += active_threads) {
      process_one_lru_resident_row(row, topk_int, capacity_int, max_token_int, req_ids, last_req_ids, topk_indices,
                                   stable_prefix_lens, slot_to_token, lru_slots, current_slots, miss_count, miss_tokens,
                                   miss_slots, token_mark, token_pos, slots, miss_positions, epoch);
    }
  }
}


constexpr int32_t MAX_RESIDENT_AGE = std::numeric_limits<int32_t>::max() - 1;

// lru_stamps is indexed by logical queue position, not physical slot. Scratch
// holds three capacity-sized slot arrays plus one capacity-sized age array.
// Uniform saturated ageing preserves the survivors' order, allowing an O(C+K)
// merge of survivors, age=P refills and age=0 hits without sorting or allocation.
FORCE_INLINE void process_one_reuse_prioritized_lru_resident_row(
    const int row, const int32_t topk, const int32_t capacity, const int32_t max_token,
    const int64_t* RESTRICT req_ids, int64_t* RESTRICT last_req_ids,
    const int32_t* RESTRICT topk_indices, const int32_t* RESTRICT stable_prefix_lens,
    int32_t* RESTRICT slot_to_token, int32_t* RESTRICT lru_slots, int32_t* RESTRICT lru_stamps,
    int32_t* RESTRICT current_slots, int32_t* RESTRICT miss_count, int32_t* RESTRICT miss_tokens_out,
    int32_t* RESTRICT miss_slots_out, int32_t* RESTRICT token_mark, int32_t* RESTRICT token_pos,
    int32_t* RESTRICT slot_workspace, int32_t* RESTRICT survivor_ages,
    int32_t* RESTRICT miss_positions, int32_t* RESTRICT epoch, const int32_t probation_age) {
  const int64_t row_offset = static_cast<int64_t>(row) * capacity;
  auto* tokens = slot_to_token + row_offset;
  auto* slots = lru_slots + row_offset;
  auto* stamps = lru_stamps + row_offset;
  auto* current = current_slots + static_cast<int64_t>(row) * topk;
  auto* miss_tokens = miss_tokens_out + static_cast<int64_t>(row) * topk;
  auto* miss_slots = miss_slots_out + static_cast<int64_t>(row) * topk;
  const auto* selected = topk_indices + static_cast<int64_t>(row) * topk;
  auto* hits = slot_workspace;
  auto* survivors = slot_workspace + capacity;
  auto* free_slots = slot_workspace + static_cast<int64_t>(capacity) * 2;
  std::fill(current, current + topk, -1);
  std::fill(miss_tokens, miss_tokens + topk, -1);
  std::fill(miss_slots, miss_slots + topk, -1);

  if (last_req_ids[row] != req_ids[row]) {
    reset_lru_resident_row(tokens, slots, capacity);
    std::fill(stamps, stamps + capacity, 0);
    last_req_ids[row] = req_ids[row];
  }
  const int32_t stable = std::clamp(stable_prefix_lens[row], 0, max_token);
  const int32_t base = next_lru_resident_epoch(token_mark, token_pos, epoch, max_token);
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = selected[pos];
    if (is_valid_lru_resident_token(token, max_token) && token_mark[token] != base) {
      token_mark[token] = base;
      token_pos[token] = pos;
    }
  }

  int32_t hit_count = 0;
  int32_t survivor_count = 0;
  int32_t free_count = 0;
  for (int32_t order = 0; order < capacity; ++order) {
    const int32_t slot = slots[order];
    const int32_t token = tokens[slot];
    // Invalidate rewritten speculative KV before checking reuse, even for age=0.
    if (!is_valid_lru_resident_token(token, max_token) || token >= stable) {
      tokens[slot] = -1;
      free_slots[free_count++] = slot;
    } else if (token_mark[token] == base) {
      current[token_pos[token]] = slot;
      hits[hit_count++] = slot;
    } else {
      survivors[survivor_count] = slot;
      survivor_ages[survivor_count++] = std::min(stamps[order], MAX_RESIDENT_AGE - 1) + 1;
    }
  }

  int32_t count = 0;
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = selected[pos];
    if (is_valid_lru_resident_token(token, max_token) && token_pos[token] == pos && current[pos] < 0) {
      miss_tokens[count] = token;
      miss_positions[count++] = pos;
    }
  }
  // C >= K guarantees enough non-hit slots, including when selections repeat.
  const int32_t used_free = std::min(count, free_count);
  const int32_t used_survivors = count - used_free;
  for (int32_t idx = 0; idx < count; ++idx) {
    const int32_t slot = idx < used_free ? free_slots[idx] : survivors[idx - used_free];
    tokens[slot] = miss_tokens[idx];
    current[miss_positions[idx]] = slot;
    miss_slots[idx] = slot;
  }
  miss_count[row] = count;
  // Preserve all attention positions for repeated selections, but load each
  // distinct token only once. Repetition within a round is not reuse evidence.
  for (int32_t pos = 0; pos < topk; ++pos) {
    const int32_t token = selected[pos];
    if (is_valid_lru_resident_token(token, max_token)) {
      current[pos] = current[token_pos[token]];
    }
  }

  int32_t out = 0;
  for (int32_t idx = used_free; idx < free_count; ++idx) {
    slots[out] = free_slots[idx];
    stamps[out++] = 0;
  }
  int32_t survivor = used_survivors;
  // Existing entries win ties, followed by refills in top-k order, then hits
  // in their previous queue order. In particular this also defines P=0.
  for (; survivor < survivor_count && survivor_ages[survivor] >= probation_age; ++survivor) {
    slots[out] = survivors[survivor];
    stamps[out++] = survivor_ages[survivor];
  }
  for (int32_t idx = 0; idx < count; ++idx) {
    slots[out] = miss_slots[idx];
    stamps[out++] = probation_age;
  }
  for (; survivor < survivor_count; ++survivor) {
    slots[out] = survivors[survivor];
    stamps[out++] = survivor_ages[survivor];
  }
  for (int32_t idx = 0; idx < hit_count; ++idx) {
    slots[out] = hits[idx];
    stamps[out++] = 0;
  }
}

HOT_FUNCTION inline void reuse_prioritized_lru_resident_compact(
    uintptr_t req_ids_ptr, uintptr_t last_req_ids_ptr, uintptr_t topk_indices_ptr,
    uintptr_t stable_prefix_lens_ptr, uintptr_t slot_to_token_ptr, uintptr_t lru_slots_ptr,
    uintptr_t current_slots_ptr, uintptr_t miss_count_ptr, uintptr_t miss_tokens_ptr,
    uintptr_t miss_slots_ptr, uintptr_t token_mark_workspace_ptr, uintptr_t token_pos_workspace_ptr,
    uintptr_t slot_workspace_ptr, uintptr_t miss_position_workspace_ptr, uintptr_t epochs_ptr,
    int64_t num_reqs, int64_t topk, int64_t capacity, int64_t max_token,
    int64_t workspace_threads, int64_t requested_threads, uintptr_t lru_stamps_ptr,
    uintptr_t age_workspace_ptr, int64_t probation_age) {
  if (capacity < topk || topk <= 0 || max_token <= 0 || workspace_threads <= 0 ||
      probation_age < 0 || probation_age > MAX_RESIDENT_AGE) {
    throw std::invalid_argument("Invalid reuse-prioritized LRU dimensions or probation_age");
  }
  if (num_reqs <= 0) {
    return;
  }
  const int active_threads = choose_lru_resident_threads(
      static_cast<int>(num_reqs), static_cast<int>(workspace_threads), static_cast<int>(requested_threads));
  // Each worker owns its epoch marks and scratch; rows remain independent.
  const auto process_rows = [&](const int thread_id, const int team_size) {
    for (int row = thread_id; row < num_reqs; row += team_size) {
      process_one_reuse_prioritized_lru_resident_row(
          row, topk, capacity, max_token,
          reinterpret_cast<int64_t*>(req_ids_ptr), reinterpret_cast<int64_t*>(last_req_ids_ptr),
          reinterpret_cast<int32_t*>(topk_indices_ptr), reinterpret_cast<int32_t*>(stable_prefix_lens_ptr),
          reinterpret_cast<int32_t*>(slot_to_token_ptr), reinterpret_cast<int32_t*>(lru_slots_ptr),
          reinterpret_cast<int32_t*>(lru_stamps_ptr), reinterpret_cast<int32_t*>(current_slots_ptr),
          reinterpret_cast<int32_t*>(miss_count_ptr), reinterpret_cast<int32_t*>(miss_tokens_ptr),
          reinterpret_cast<int32_t*>(miss_slots_ptr),
          reinterpret_cast<int32_t*>(token_mark_workspace_ptr) + static_cast<int64_t>(thread_id) * max_token,
          reinterpret_cast<int32_t*>(token_pos_workspace_ptr) + static_cast<int64_t>(thread_id) * max_token,
          reinterpret_cast<int32_t*>(slot_workspace_ptr) + static_cast<int64_t>(thread_id) * capacity * 3,
          reinterpret_cast<int32_t*>(age_workspace_ptr) + static_cast<int64_t>(thread_id) * capacity,
          reinterpret_cast<int32_t*>(miss_position_workspace_ptr) + static_cast<int64_t>(thread_id) * topk,
          reinterpret_cast<int32_t*>(epochs_ptr) + thread_id, probation_age);
    }
  };
  // Single-row decode avoids entering even a serialized OpenMP region.
  if (active_threads == 1) {
    process_rows(0, 1);
    return;
  }
#pragma omp parallel num_threads(active_threads)
  {
#ifdef _OPENMP
    process_rows(lru_resident_thread_id(), omp_get_num_threads());
#else
    process_rows(0, 1);
#endif
  }
}
