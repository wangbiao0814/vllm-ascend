// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vllm_ascend {

// Count selected stable-prefix occurrences (including duplicate selections).
// Plan rows may be strided; positive entries encode hits, negative ones misses.
inline void accumulate_sparse_plan_hits(int64_t* counts, const int64_t* requests,
                                       const int32_t* sources, const int32_t* prefixes,
                                       const int16_t* plan, int64_t stride,
                                       int64_t rows, int64_t topk) {
  int64_t accesses = 0;
  int64_t hits = 0;
  for (int64_t row = 0; row < rows; ++row) {
    if (requests[row] < 0) {
      continue;
    }
    for (int64_t col = 0; col < topk; ++col) {
      const auto token = sources[row * topk + col];
      if (token >= 0 && token < prefixes[row]) {
        ++accesses;
        hits += plan[row * stride + col] > 0;
      }
    }
  }
  counts[0] += accesses;
  counts[1] += hits;
}

}  // namespace vllm_ascend
