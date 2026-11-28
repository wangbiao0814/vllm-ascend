// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend::hisparse {
inline constexpr uint32_t kGroupCandidates = 1;
inline constexpr uint32_t kReuseTagScratch = 2;
inline constexpr uint32_t kVectorAge = 4;
inline constexpr uint32_t kVectorGrouping = 8;
inline constexpr uint32_t kDenseRefillAge = 16;
inline constexpr uint32_t kAsyncAgeWriteback = 32;
inline constexpr uint32_t kAgeOptions =
    kGroupCandidates | kReuseTagScratch | kVectorAge | kVectorGrouping | kDenseRefillAge | kAsyncAgeWriteback;
// Pad request rows to one DMA block so concurrent leaders cannot share it.
inline constexpr uint32_t kPendingAgeStride = 8;
}  // namespace vllm_ascend::hisparse
