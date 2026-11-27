# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from benchmarks.benchmark_hisparse_eviction import Reference, make_trace


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


if __name__ == "__main__":
    unittest.main()
