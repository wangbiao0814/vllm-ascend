// SPDX-License-Identifier: Apache-2.0
#ifndef LIM_AGE_HIT_RUNS_VECTOR_H
#define LIM_AGE_HIT_RUNS_VECTOR_H
#include "kernel_operator.h"
#include "age_hit_blocks.h"

namespace LIMAge {
constexpr uint32_t HIT_BLOCK_SHIFT = 9U;
constexpr uint32_t HIT_RUN_ALIGN_WORDS = 64U; // 256-byte compare-mask alignment
constexpr uint32_t HIT_RUN_SENTINEL_WORDS = 8U; // 32-byte vector operand alignment
constexpr uint32_t HIT_RUN_OFFSET_SHIFT = 2U; // B32 element index to byte offset
constexpr uint32_t HIT_RUN_MASK_BITS_PER_HALFWORD = 16U;
constexpr uint32_t HIT_RUN_SCRATCH_EXTRA = 5U * HIT_RUN_ALIGN_WORDS;
static_assert((1U << HIT_BLOCK_SHIFT) == HIT_BLOCK_SIZE, "Hit block shift must match block size");

// count must be a multiple of 64. scratch requires 4*count+320 B32 words.
// The caller waits V_S before consuming the three retained counts/metadata.
__aicore__ inline void BuildVectorHitRunMetadata(
    AscendC::LocalTensor<int32_t> sources, AscendC::LocalTensor<float> scratch,
    AscendC::LocalTensor<int32_t> blocks, AscendC::LocalTensor<int32_t> starts,
    AscendC::LocalTensor<int32_t> ends, uint32_t count, uint32_t capacity,
    uint64_t &blockCount, uint64_t &startCount, uint64_t &endCount)
{
    using namespace AscendC;
    // Eight prefix/suffix sentinels allow aligned Gather bases for both
    // predecessor and successor vectors without per-lane scalar edits.
    LocalTensor<float> extended = scratch;
    LocalTensor<float> neighbor = scratch[count + HIT_RUN_ALIGN_WORDS];
    LocalTensor<uint32_t> offsets = scratch[2U * count + HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> blockIds = scratch[3U * count + HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint32_t>();
    LocalTensor<uint8_t> valid = scratch[4U * count + HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> upper = scratch[4U * count + 2U * HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> beginMask = scratch[4U * count + 3U * HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> endMask = scratch[4U * count + 4U * HIT_RUN_ALIGN_WORDS].ReinterpretCast<uint8_t>();
    ShiftRight(blockIds, sources.ReinterpretCast<uint32_t>(), HIT_BLOCK_SHIFT, count);
    Duplicate(extended, -1.0F, HIT_RUN_SENTINEL_WORDS);
    Duplicate(extended[count + HIT_RUN_SENTINEL_WORDS], -1.0F, HIT_RUN_SENTINEL_WORDS);
    PipeBarrier<PIPE_V>();
    Cast(extended[HIT_RUN_SENTINEL_WORDS], blockIds.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
    Cast(neighbor, sources, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    CompareScalar(valid, neighbor, 0.0F, CMPMODE::GE, count);
    // AI Core forbids direct scalar unsigned-to-float casts. The validated
    // Source capacity is <= 2^21, so the signed intermediate is exact and safe.
    const int32_t signedCapacity = static_cast<int32_t>(capacity);
    CompareScalar(upper, neighbor, static_cast<float>(signedCapacity), CMPMODE::LT, count);
    PipeBarrier<PIPE_V>();
    And(valid.ReinterpretCast<uint16_t>(), valid.ReinterpretCast<uint16_t>(),
        upper.ReinterpretCast<uint16_t>(), count / HIT_RUN_MASK_BITS_PER_HALFWORD);
    PipeBarrier<PIPE_V>();
    Duplicate(neighbor, -1.0F, count);
    PipeBarrier<PIPE_V>();
    // Invalid IDs can share a partial final block with valid IDs. Mask their
    // block values before neighbor comparison so the first valid lane is kept.
    Select(extended[HIT_RUN_SENTINEL_WORDS], valid, extended[HIT_RUN_SENTINEL_WORDS], neighbor, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    PipeBarrier<PIPE_V>();
    // Compare numeric float block IDs, not reinterpreted subnormal bit patterns.
    ArithProgression(offsets.ReinterpretCast<int32_t>(), static_cast<int32_t>(HIT_RUN_SENTINEL_WORDS - 1U), 1, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(offsets, offsets, HIT_RUN_OFFSET_SHIFT, count);
    PipeBarrier<PIPE_V>();
    Gather(neighbor, extended, offsets, 0U, count);
    PipeBarrier<PIPE_V>();
    Compare(beginMask, extended[HIT_RUN_SENTINEL_WORDS], neighbor, CMPMODE::NE, count);
    PipeBarrier<PIPE_V>();
    And(beginMask.ReinterpretCast<uint16_t>(), beginMask.ReinterpretCast<uint16_t>(),
        valid.ReinterpretCast<uint16_t>(), count / HIT_RUN_MASK_BITS_PER_HALFWORD);
    ArithProgression(offsets.ReinterpretCast<int32_t>(), static_cast<int32_t>(HIT_RUN_SENTINEL_WORDS + 1U), 1, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(offsets, offsets, HIT_RUN_OFFSET_SHIFT, count);
    PipeBarrier<PIPE_V>();
    Gather(neighbor, extended, offsets, 0U, count);
    PipeBarrier<PIPE_V>();
    Compare(endMask, extended[HIT_RUN_SENTINEL_WORDS], neighbor, CMPMODE::NE, count);
    PipeBarrier<PIPE_V>();
    And(endMask.ReinterpretCast<uint16_t>(), endMask.ReinterpretCast<uint16_t>(),
        valid.ReinterpretCast<uint16_t>(), count / HIT_RUN_MASK_BITS_PER_HALFWORD);
    PipeBarrier<PIPE_V>();
    GatherMaskParams params;
    params.repeatTimes = 1;
    params.src0BlockStride = 1;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 0;
    GatherMask(blocks.ReinterpretCast<uint32_t>(), blockIds,
               beginMask.ReinterpretCast<uint32_t>(), true, count, params, blockCount);
    ArithProgression(offsets.ReinterpretCast<int32_t>(), 0, 1, count);
    PipeBarrier<PIPE_V>();
    GatherMask(starts.ReinterpretCast<uint32_t>(), offsets,
               beginMask.ReinterpretCast<uint32_t>(), true, count, params, startCount);
    PipeBarrier<PIPE_V>();
    ArithProgression(offsets.ReinterpretCast<int32_t>(), 1, 1, count);
    PipeBarrier<PIPE_V>();
    GatherMask(ends.ReinterpretCast<uint32_t>(), offsets,
               endMask.ReinterpretCast<uint32_t>(), true, count, params, endCount);
    PipeBarrier<PIPE_V>();
}
} // namespace LIMAge
#endif
