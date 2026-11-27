// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend::hisparse {
// SIMT kernels must reserve DCache space out of the UB capacity reported by
// GetCoreMemSize before declaring local memory via SetLocalMemorySize (the
// same pattern as k2q_csr / scatter_nd_update_sk on this target).
inline constexpr uint64_t kResolverDcacheReserveBytes = 32 * 1024;
// Host-side static upper bound: A5 reports 256 KiB of UB per AIV, minus the
// DCache reservation. Used only where the platform API is unavailable.
inline constexpr uint64_t kResolverMaxSimtUbBytes = 256 * 1024 - kResolverDcacheReserveBytes;
inline constexpr uint64_t kResolverDescriptorBytes = 512;

inline constexpr uint64_t resolver_simt_ub_budget(uint64_t platform_ub_bytes) {
    return platform_ub_bytes > kResolverDcacheReserveBytes ?
        platform_ub_bytes - kResolverDcacheReserveBytes : 0;
}

constexpr uint64_t resolver_scratch_words(uint64_t hot, uint64_t hash_size) {
    const uint64_t tiles = (hot + 31) / 32;
    return (2 * hash_size + hot + 2 * tiles + 64 + 15) / 16 * 16;
}

constexpr uint64_t resolver_scratch_bytes(uint64_t hot, uint64_t hash_size) {
    return resolver_scratch_words(hot, hash_size) * sizeof(int32_t) + kResolverDescriptorBytes;
}

constexpr uint64_t resolver_hash_capacity(uint64_t hot, uint64_t union_rows, uint64_t ub_budget) {
    uint64_t hash_size = 1;
    while (hash_size <= union_rows) hash_size <<= 1;
    const uint64_t tiles = (hot + 31) / 32;
    const uint64_t fixed_words = hot + 2 * tiles + 64;
    const uint64_t available_words = (ub_budget - kResolverDescriptorBytes) / 64 * 16;
    const uint64_t max_hash_size = (available_words - fixed_words) / 2;
    return hash_size < max_hash_size ? hash_size : max_hash_size;
}

// arch22 kernels use scalar UB storage without the SIMT DCache reservation; the
// runtime reserve is 16 KiB and the complete hot-tag row stays resident across
// the union walk (see the migrated arch22 resolver header).
inline constexpr uint64_t kResolverArch22ReserveBytes = 16 * 1024;

inline constexpr uint64_t resolver_arch22_ub_budget(uint64_t platform_ub_bytes) {
    return platform_ub_bytes > kResolverArch22ReserveBytes ?
        platform_ub_bytes - kResolverArch22ReserveBytes : 0;
}

constexpr uint64_t resolver_arch22_pair_bytes(uint64_t entries) {
    // TPipe rounds each of the two int32 buffers to a 32-byte boundary.
    return (entries + 7) / 8 * 64;
}

constexpr int64_t kMaxProbationAge = 32767;
constexpr uint64_t resolver_arch22_age_bytes(uint64_t hot, bool with_age) {
    return with_age ? (hot * sizeof(int16_t) + 63) / 64 * 64 : 0;
}

constexpr uint64_t resolver_arch22_scratch_bytes(uint64_t hot, uint64_t hash_size, bool with_age = false) {
    return resolver_arch22_pair_bytes(hot) + resolver_arch22_pair_bytes(hash_size) + 2 * 1024 +
           kResolverDescriptorBytes + resolver_arch22_age_bytes(hot, with_age);
}

constexpr uint64_t resolver_arch22_hash_capacity(uint64_t hot, uint64_t union_rows, uint64_t ub_budget, bool with_age) {
    uint64_t hash_size = 1;
    while (hash_size <= union_rows) hash_size <<= 1;
    const uint64_t fixed_bytes = resolver_arch22_scratch_bytes(hot, 0, with_age);
    const uint64_t max_hash_size = ub_budget > fixed_bytes ? (ub_budget - fixed_bytes) / 64 * 8 : 0;
    return hash_size < max_hash_size ? hash_size : max_hash_size;
}
}  // namespace vllm_ascend::hisparse
