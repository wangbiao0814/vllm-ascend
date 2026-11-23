# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; runnable directly without importing vLLM or torch_npu."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[5]


class TestLimAgePolicy(unittest.TestCase):
    def test_native_arithmetic_geometry_and_long_source_sort_keys(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required")
        source = r"""
#include "age_policy.h"
#include <cassert>
#include <cstring>
#include <cstdint>
int main() {
    using namespace LIMAge;
    assert(Increment(0) == 1);
    assert(Increment(MAX_AGE) == MAX_AGE);
    assert(Increment(MAX_AGE-1) == MAX_AGE);
    assert(ValidGeometry(8192,4096,8320,8320));
    assert(!ValidGeometry(8192,4096,8192,8320));
    assert(!ValidGeometry(8193,4096,8320,8320));
    assert(!ValidGeometry(8192,4097,8320,8320));
    assert(!ValidGeometry(8192,16384,8320,8320));
    // Production eviction codec embeds source_high4 in the low key bits.
    // Every integral age must remain ordered despite all 16 source tags.
    for (int age=0; age<MAX_AGE; ++age) {
        float current=float(age+1), next=float(age+2);
        uint32_t bits; std::memcpy(&bits,&current,sizeof(bits));
        bits=(bits & ~15U) | 15U;
        float highest; std::memcpy(&highest,&bits,sizeof(bits));
        assert(highest < next);
    }
}
"""
        include = ROOT / "csrc/attention/fused_lightning_indexer_age_manage/op_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / "test.cpp"
            cpp.write_text(source)
            binary = Path(tmp) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(include), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


class TestAgeConfigContract(unittest.TestCase):
    def test_validation_keeps_default_and_rejects_unsupported_values(self):
        module = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text())
        config = next(
            node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "SparseKVOffloadConfig"
        )
        method = next(
            node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "_validate_values"
        )
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "config", "exec"), namespace)
        validate = namespace["_validate_values"]
        base = dict(
            eviction_policy="score",
            probation_age=4,
            use_fused_overlap=False,
            topk_buffer_size=8192,
            dram_size_per_dp_GB=110,
            store_port_base=8500,
        )
        self.assertEqual(validate(SimpleNamespace(**base)).eviction_policy, "score")
        for updates in (
            {"eviction_policy": "wrong"},
            {"probation_age": -1},
            {"probation_age": 1 << 19},
            {"eviction_policy": "age", "use_fused_overlap": True},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                validate(SimpleNamespace(**(base | updates)))


if __name__ == "__main__":
    unittest.main()
