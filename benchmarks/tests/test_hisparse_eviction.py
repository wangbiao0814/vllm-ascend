# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.benchmark_hisparse_eviction import Reference, make_hit_rate_trace, make_trace
from benchmarks.benchmark_hisparse_eviction_ablation import (
    add_latency_deltas,
    expand_request_trace,
    load_trace,
    reference_stats,
)


class TestHiSparseEvictionReference(unittest.TestCase):
    def test_prefer_empty_slots_and_merge_by_age(self):
        states = Reference(5, 4)
        states.tags = [10, 11, 12, 13, -1]
        states.ages = [8, 5, 2, 0, 32767]
        plan, out = states.resolve([13, 14, 15, 14])
        self.assertEqual(plan, [(14, 4), (15, 0)])
        self.assertEqual(out, [3, 4, 0, 4])
        self.assertEqual(states.order, [1, 4, 0, 2, 3])
        self.assertEqual([states.ages[slot] for slot in states.order], [6, 4, 4, 3, 0])
        self.assertEqual(states.lookups, 3)

    def test_age_saturates_and_ties_preserve_queue_order(self):
        states = Reference(4, 32767)
        states.tags = [20, 21, 22, -1]
        states.ages = [32767, 32766, 3, 0]
        states.resolve([23, 22])
        self.assertEqual(states.order, [0, 1, 3, 2])
        self.assertEqual(states.ages, [32767, 32767, 0, 32767])

    def test_reused_rows_survive_the_scan(self):
        lru, age = Reference(40, None), Reference(40, 4)
        for step in make_trace("reuse-plus-scan", 16, 4, 8, 40):
            lru.resolve(step)
            age.resolve(step)
        self.assertEqual(lru.lookups, age.lookups)
        self.assertGreater(age.hit_rate, lru.hit_rate)


class TestAblationTrace(unittest.TestCase):
    def test_writeback_delta_uses_the_matching_control(self):
        row = {"resolver_us": 98.0, "combined_us": 145.0}
        control = {"resolver_us": 100.0, "combined_us": 150.0}
        add_latency_deltas(row, control, prefix="writeback_")
        self.assertEqual(row["writeback_resolver_delta_us"], -2.0)
        self.assertEqual(row["writeback_combined_delta_us"], -5.0)
        self.assertAlmostEqual(row["writeback_resolver_overhead_pct"], -2.0)

    def test_steady_hit_rate_excludes_priming_and_keeps_union_full(self):
        requests, queries, topk, rounds, hot_size = 3, 4, 8, 6, 40
        width = queries * topk
        for rate in (0, 0.5, 0.9, 1.0):
            trace = list(make_hit_rate_trace(rounds, queries, topk, rate))
            self.assertTrue(all(len(set(row)) == width for row in trace))
            batch, _ = expand_request_trace(trace, requests)
            for age in (None, 4, 32767):
                with self.subTest(rate=rate, age=age):
                    stats = reference_stats(batch, width, hot_size, age, priming_rounds=1)
                    expected_misses = (width - round(width * rate)) * rounds * requests
                    self.assertEqual(stats, (width * rounds * requests, expected_misses))

    def test_batch_uses_disjoint_host_ranges_and_preserves_padding(self):
        trace = [[1, -1, 2, 3], [0, -1, 2, 2]]
        expanded, host_size = expand_request_trace(trace, 3)
        self.assertEqual(host_size, 12)
        self.assertEqual(
            expanded, [[1, -1, 2, 3, 5, -1, 6, 7, 9, -1, 10, 11], [0, -1, 2, 2, 4, -1, 6, 6, 8, -1, 10, 10]]
        )

    def test_batch_rejects_host_rows_that_exceed_int32(self):
        with self.assertRaisesRegex(ValueError, "exceeds int32"):
            expand_request_trace([[2**30]], 2)

    def test_trace_supports_flat_and_per_query_rounds(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[[[1, -1], [2, 3]], [4, 5, 6, 7]]")
            self.assertEqual(load_trace(path, 2, 2), [[1, -1, 2, 3], [4, 5, 6, 7]])

    def test_trace_rejects_out_of_range_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text("[[1, 2147483647]]")
            with self.assertRaisesRegex(ValueError, "int32 IDs"):
                load_trace(path, 1, 2)


if __name__ == "__main__":
    unittest.main()
