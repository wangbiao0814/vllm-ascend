# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU test: python <this file> -v. No torch/ACL required."""

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSparseKVHitRateStats(unittest.TestCase):
    def test_counts_warmup_reuse_reset_and_weighted_logging(self):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            self.skipTest("A C++17 compiler is required")
        include = Path(__file__).resolve().parents[5] / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"
        with tempfile.TemporaryDirectory(prefix="sfa-hit-rate-test-") as directory:
            source = Path(directory) / "test.cpp"
            binary = Path(directory) / "test"
            source.write_text(r'''
#include "hit_rate_stats.h"
#include <cassert>
int main() {
  SparseKVHitRateStats stats({"layer0", "layer1"}, "reuse_prioritized_lru", "test", 32, 1, 1);
  int32_t topk[] = {2, 2, -1, 3};
  int32_t slots[] = {7, 7, -1, 8};
  int32_t count[] = {1};
  int32_t misses[] = {3, -1, -1, -1};
  int64_t req[] = {1};
  const auto record = [&]() {
    stats.record(0, reinterpret_cast<uintptr_t>(topk), reinterpret_cast<uintptr_t>(slots),
                 reinterpret_cast<uintptr_t>(count), reinterpret_cast<uintptr_t>(misses),
                 reinterpret_cast<uintptr_t>(req), 1, 4);
  };
  record();
  stats.record_reused(1, 0);
  auto snapshot = stats.snapshot();
  assert(snapshot[0][0] == 1 && snapshot[0][1] == 0 && snapshot[0][2] == 0);
  count[0] = 0;
  record();  // unique tokens 2,3: two hits, not three
  stats.record_reused(1, 0);
  snapshot = stats.snapshot();
  assert(snapshot[0][1] == 2 && snapshot[1][1] == 2);
  topk[0] = 4; topk[1] = -1; topk[2] = 32; topk[3] = -1;
  count[0] = 1; misses[0] = 4;
  record();  // global 4 hits / 5 accesses = 80%, not mean(2/3, 2/2)
  snapshot = stats.snapshot();
  assert(snapshot[0][1] == 2 && snapshot[0][2] == 1);
  req[0] = 0;
  record();  // padded request contributes nothing
  stats.record_reused(1, 0);
  snapshot = stats.snapshot();
  assert(snapshot[0][2] == 1 && snapshot[1][2] == 0);
  stats.reset();
  for (auto entry : stats.snapshot()) assert(entry[0] == 0 && entry[1] == 0 && entry[2] == 0);
  req[0] = 1;
  record();  // reset also restarts warmup
  assert(stats.snapshot()[0][2] == 0);

  SparseKVHitRateStats requests({"l0", "l1"}, "lru", "test", 32, 0, 0, true);
  int32_t selections[] = {1,2,3,4}, positions[] = {0,1,0,1};
  int32_t counts[] = {1,0}, missing[] = {2,-1,-1,-1};
  int64_t ids[] = {11,22};
  requests.record(0, reinterpret_cast<uintptr_t>(selections), reinterpret_cast<uintptr_t>(positions),
                  reinterpret_cast<uintptr_t>(counts), reinterpret_cast<uintptr_t>(missing),
                  reinterpret_cast<uintptr_t>(ids), 2, 2);
  requests.record_reused(1, 0);
  requests.finish_request(11, "request-A");
  requests.finish_request(11, "request-A");  // duplicate completion must not print twice
  requests.finish_request(22, "request-B");
  requests.finish_request(99, "prefill-only");  // no decode state, no output

  SparseKVHitRateStats empty({"empty"}, "lru", "test", 32, 1, 0);
  topk[0] = -1; topk[2] = -1; count[0] = 0;
  empty.record(0, reinterpret_cast<uintptr_t>(topk), reinterpret_cast<uintptr_t>(slots),
               reinterpret_cast<uintptr_t>(count), reinterpret_cast<uintptr_t>(misses),
               reinterpret_cast<uintptr_t>(req), 1, 4);
  assert(empty.snapshot()[0][1] == 0 && empty.snapshot()[0][2] == 0);
}
''')
            subprocess.run(
                compiler + ["-std=c++17", "-O2", "-pthread", "-I", str(include), str(source), "-o", str(binary)],
                check=True, capture_output=True, text=True,
            )
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.count("request_id=request-A layer=ALL"), 1)
        self.assertIn("request_id=request-A layer=ALL hits=2 misses=2 hit_rate=50%", result.stdout)
        self.assertIn("request_id=request-B layer=ALL hits=4 misses=0 hit_rate=100%", result.stdout)
        self.assertNotIn("prefill-only", result.stdout)
        self.assertIn("overall_hits=4 overall_misses=1 overall_hit_rate=80%", result.stdout)
        self.assertIn("layer=layer0 updates=3 hits=2 misses=1", result.stdout)
        self.assertIn("layer=layer1 updates=2 hits=2 misses=0", result.stdout)
        self.assertIn("policy=lru layer=empty updates=1 hits=0 misses=0 hit_rate=0%", result.stdout)


if __name__ == "__main__":
    unittest.main()
