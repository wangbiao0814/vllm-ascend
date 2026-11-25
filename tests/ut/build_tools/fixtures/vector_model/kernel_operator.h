// SPDX-License-Identifier: Apache-2.0
// CPU semantic model for the limited operations used by age_hit_runs_vector.h.
// This does not model device synchronization, instruction support or timing.
#pragma once
#include <cassert>
#include <cstdint>
#include <cstring>
#define __aicore__
namespace AscendC {
enum { PIPE_V = 0 };
enum class RoundMode { CAST_NONE };
enum class CMPMODE { GE, LT, NE };
enum class SELMODE { VSEL_TENSOR_TENSOR_MODE };
struct GatherMaskParams {
  uint8_t src0BlockStride;
  uint16_t repeatTimes;
  uint16_t src0RepeatStride;
  uint8_t src1RepeatStride;
};
template <typename T>
struct LocalTensor {
  uint8_t* base;
  uint32_t offset;
  uint32_t bytes;
  T GetValue(uint32_t i) const {
    assert((i + 1U) * sizeof(T) <= bytes);
    T value;
    std::memcpy(&value, base + offset + i * sizeof(T), sizeof(T));
    return value;
  }
  void SetValue(uint32_t i, T value) const {
    assert((i + 1U) * sizeof(T) <= bytes);
    std::memcpy(base + offset + i * sizeof(T), &value, sizeof(T));
  }
  LocalTensor<T> operator[](uint32_t i) const {
    assert(i * sizeof(T) <= bytes);
    return {base, offset + static_cast<uint32_t>(i * sizeof(T)), bytes - static_cast<uint32_t>(i * sizeof(T))};
  }
  template <typename U>
  LocalTensor<U> ReinterpretCast() const {
    return {base, offset, bytes};
  }
};
template <typename T>
void Aligned(LocalTensor<T> x) {
  assert(x.offset % 32U == 0U);
}
template <int P>
void PipeBarrier() {}
template <typename T>
void Duplicate(LocalTensor<T> dst, T value, uint32_t n) {
  Aligned(dst);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, value);
}
template <typename D, typename S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode, uint32_t n) {
  Aligned(dst);
  Aligned(src);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, static_cast<D>(src.GetValue(i)));
}
template <typename T, typename U, typename V>
void ArithProgression(LocalTensor<T> dst, U first, V step, uint32_t n) {
  Aligned(dst);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, static_cast<T>(first + i * step));
}
inline void ShiftRight(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, uint32_t shift, uint32_t n) {
  Aligned(dst);
  Aligned(src);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, src.GetValue(i) >> shift);
}
inline void ShiftLeft(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, uint32_t shift, uint32_t n) {
  Aligned(dst);
  Aligned(src);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, src.GetValue(i) << shift);
}
inline bool CompareValue(float a, float b, CMPMODE mode) {
  return mode == CMPMODE::GE ? a >= b : mode == CMPMODE::LT ? a < b : a != b;
}
inline void SetMask(LocalTensor<uint8_t> dst, uint32_t i, bool value) {
  uint8_t word = dst.GetValue(i / 8U);
  const uint8_t bit = static_cast<uint8_t>(1U << (i % 8U));
  dst.SetValue(i / 8U, value ? static_cast<uint8_t>(word | bit) : static_cast<uint8_t>(word & ~bit));
}
inline void CompareScalar(LocalTensor<uint8_t> dst, LocalTensor<float> src, float value, CMPMODE mode, uint32_t n) {
  Aligned(dst);
  Aligned(src);
  for (uint32_t i = 0; i < (n + 7U) / 8U; ++i) dst.SetValue(i, 0);
  for (uint32_t i = 0; i < n; ++i) SetMask(dst, i, CompareValue(src.GetValue(i), value, mode));
}
inline void Compare(LocalTensor<uint8_t> dst, LocalTensor<float> a, LocalTensor<float> b, CMPMODE mode, uint32_t n) {
  Aligned(dst);
  Aligned(a);
  Aligned(b);
  for (uint32_t i = 0; i < (n + 7U) / 8U; ++i) dst.SetValue(i, 0);
  for (uint32_t i = 0; i < n; ++i) SetMask(dst, i, CompareValue(a.GetValue(i), b.GetValue(i), mode));
}
inline void And(LocalTensor<uint16_t> dst, LocalTensor<uint16_t> a, LocalTensor<uint16_t> b, uint32_t n) {
  Aligned(dst);
  Aligned(a);
  Aligned(b);
  for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, a.GetValue(i) & b.GetValue(i));
}
inline void Select(LocalTensor<float> dst, LocalTensor<uint8_t> mask, LocalTensor<float> a, LocalTensor<float> b,
                   SELMODE, uint32_t n) {
  Aligned(dst);
  Aligned(mask);
  Aligned(a);
  Aligned(b);
  for (uint32_t i = 0; i < n; ++i)
    dst.SetValue(i, (mask.GetValue(i / 8U) >> (i % 8U)) & 1U ? a.GetValue(i) : b.GetValue(i));
}
inline void Gather(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<uint32_t> offsets, uint32_t base,
                   uint32_t n) {
  Aligned(dst);
  Aligned(src);
  Aligned(offsets);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t byte = base + offsets.GetValue(i);
    assert(byte % 4U == 0U);
    dst.SetValue(i, src.GetValue(byte / 4U));
  }
}
inline void GatherMask(LocalTensor<uint32_t> dst, LocalTensor<uint32_t> src, LocalTensor<uint32_t> mask, bool counter,
                       uint32_t n, const GatherMaskParams& p, uint64_t& retained) {
  Aligned(dst);
  Aligned(src);
  Aligned(mask);
  assert(counter && p.repeatTimes == 1 && p.src0BlockStride == 1 && p.src0RepeatStride == 8 && p.src1RepeatStride == 0);
  retained = 0;
  for (uint32_t i = 0; i < n; ++i)
    if ((mask.GetValue(i / 32U) >> (i % 32U)) & 1U) dst.SetValue(retained++, src.GetValue(i));
}
}  // namespace AscendC
