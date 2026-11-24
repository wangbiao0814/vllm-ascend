# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks for diagnostic workload specifications."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

from benchmarks.benchmark_lim_age import TOPK, parse_args, trace_spec


class TestDiagnosticSpec(unittest.TestCase):
    def test_diagnostic_cli_rejects_invalid_workloads(self):
        for argv in (
            ["bench", "--scenario", "all-miss", "--length", "8192", "--capacity", "8192"],
            ["bench", "--scenario", "trace", "--ablation", "all"],
        ):
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)
        with patch("sys.argv", ["bench", "--scenario", "all-hit", "--ablation", "all", "--dry-run"]):
            self.assertEqual(parse_args().ablation, "all")

    def test_all_hit_and_all_miss_residency(self):
        for capacity in (TOPK, 8192):
            for scenario in ("all-hit", "all-miss"):
                args = argparse.Namespace(
                    seed=7, scenario=scenario, length=16384, capacity=capacity, core_fraction=0.5, steps=8
                )
                spec = trace_spec(args)
                self.assertEqual(spec, trace_spec(args))
                core, resident = set(spec["core_sources"]), set(spec["initial_resident_sources"])
                self.assertEqual(len(core), TOPK)
                self.assertEqual(len(resident), capacity)
                self.assertTrue(spec["fixed_query"])
                self.assertTrue(all(0 <= source < args.length for source in resident | core))
                self.assertEqual(len(spec["query_seeds"]), args.steps)
                if scenario == "all-hit":
                    self.assertTrue(core <= resident)
                else:
                    self.assertTrue(core.isdisjoint(resident))


if __name__ == "__main__":
    unittest.main()
