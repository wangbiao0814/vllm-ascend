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
// Input must be descending by full Source ID, with masked Misses/padding -1.
// Record boundaries only when the block changes, not once per Hit.
template <typename Sources, typename Blocks, typename Starts, typename Ends>
LIM_AGE_INLINE uint32_t BuildSortedHitRuns(Sources &sources, uint32_t count,
                                          uint32_t capacity, Blocks &blocks,
                                          Starts &starts, Ends &ends)
{
    uint32_t runs = 0U;
    uint32_t previous = UINT32_MAX;
    uint32_t end = 0U;
    for (uint32_t i = 0U; i < count; ++i) {
        const int32_t source = sources.GetValue(i);
        if (source < 0) break;
        if (static_cast<uint32_t>(source) >= capacity) continue;
        const uint32_t block = static_cast<uint32_t>(source) / HIT_BLOCK_SIZE;
        if (block != previous) {
            if (runs != 0U) ends.SetValue(runs - 1U, static_cast<int32_t>(end));
            blocks.SetValue(runs, static_cast<int32_t>(block));
            starts.SetValue(runs, static_cast<int32_t>(i));
            previous = block;
            ++runs;
        }
        end = i + 1U;
    }
    if (runs != 0U) ends.SetValue(runs - 1U, static_cast<int32_t>(end));
    return runs;
}

template <typename Sources, typename Ages>
LIM_AGE_INLINE void ClearSortedHitRun(uint32_t begin, uint32_t end,
                                     Sources &sources, Ages &ages)
{
    for (uint32_t i = begin; i < end; ++i) {
        const uint32_t offset = static_cast<uint32_t>(sources.GetValue(i)) % HIT_BLOCK_SIZE;
        ages.SetValue(offset, 0);
    }
}
#undef LIM_AGE_INLINE
} // namespace LIMAge
#endif
