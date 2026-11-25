// SPDX-License-Identifier: Apache-2.0
#ifndef LIM_AGE_HIT_BLOCKS_H
#define LIM_AGE_HIT_BLOCKS_H
#include <cstdint>

namespace LIMAge {
constexpr uint32_t HIT_BLOCK_SIZE = 512U;
#if defined(__CCE_AICORE__)
#define LIM_AGE_INLINE __aicore__ inline
#else
#define LIM_AGE_INLINE inline
#endif

// Source-block buckets are temporary UB work, not a Slot-to-Source mapping.
// Each original TopK position is also its linked-list node, so no compaction
// or Source sort is required. Only Hit positions are inserted by the caller.
template <typename Heads, typename Next, typename Touched>
LIM_AGE_INLINE void AppendHitBlock(int32_t source, uint32_t position,
                                  uint32_t capacity, Heads &heads, Next &next,
                                  Touched &touched, uint32_t &touchedCount)
{
    if (source < 0 || static_cast<uint32_t>(source) >= capacity) return;
    const uint32_t block = static_cast<uint32_t>(source) / HIT_BLOCK_SIZE;
    const int32_t previous = heads.GetValue(block);
    if (previous < 0) touched.SetValue(touchedCount++, static_cast<int32_t>(block));
    next.SetValue(position, previous);
    heads.SetValue(block, static_cast<int32_t>(position));
}

template <typename Sources, typename Next, typename Ages>
LIM_AGE_INLINE void ClearHitBlock(int32_t node, Sources &sources,
                                 Next &next, Ages &ages)
{
    while (node >= 0) {
        const uint32_t position = static_cast<uint32_t>(node);
        const uint32_t offset = static_cast<uint32_t>(sources.GetValue(position)) % HIT_BLOCK_SIZE;
        ages.SetValue(offset, 0);
        node = next.GetValue(position);
    }
}
#undef LIM_AGE_INLINE
} // namespace LIMAge
#endif
