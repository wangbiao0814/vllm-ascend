// SPDX-License-Identifier: Apache-2.0
#ifndef LIM_AGE_POLICY_H
#define LIM_AGE_POLICY_H
#include <cstdint>
namespace LIMAge {
constexpr int32_t MAX_AGE = (1 << 19) - 1;
constexpr uint32_t SKIP_HIT_CLEAR = 1U;
constexpr uint32_t SKIP_INCREMENT = 2U;
constexpr uint32_t SKIP_REFILL = 4U;
constexpr uint32_t HIT_BUCKETS_ONLY = 8U;
constexpr uint32_t SKIP_LOCAL_HIT_CLEAR = 16U;
constexpr uint32_t DIAGNOSTIC_MASK = 31U;
constexpr uint32_t TILE = 512U;
constexpr uint32_t AGE_INPUT = 13U;
constexpr uint32_t AGE_OUTPUT = 7U;
constexpr uint32_t OUTPUT_COUNT = 8U;
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline constexpr int32_t Increment(int32_t age) {
    return age < 0 ? 1 : (age >= MAX_AGE ? MAX_AGE : age + 1);
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline constexpr bool ValidGeometry(int32_t length, int32_t cache, int32_t actual, uint32_t capacity) {
    return length >= 2048 && cache >= 2048 && cache <= length &&
        static_cast<uint32_t>(length) <= capacity && actual > 0 &&
        length <= ((actual - 1) / 128) * 128 && (length & 127) == 0 &&
        (cache & 127) == 0 && (length <= 2048 ? cache == length : cache <= 32640);
}
}
#endif
