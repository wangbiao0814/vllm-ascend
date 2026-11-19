# SPDX-License-Identifier: Apache-2.0
"""Execute the production CPU counter helper without CANN or NPU libraries."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def counter(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).parents[5]
    tmp = tmp_path_factory.mktemp("cpu_hit_rate")
    source = tmp / "counter.cpp"
    source.write_text(
        '#include "sparse_plan_hit_rate.h"\n'
        'extern "C" void count(int64_t* counts, const int64_t* requests, '
        'const int32_t* sources, const int32_t* prefixes, const int16_t* plan, '
        'int64_t stride, int64_t rows, int64_t topk) { '
        'vllm_ascend::accumulate_sparse_plan_hits(counts, requests, sources, '
        'prefixes, plan, stride, rows, topk); }\n'
    )
    library = tmp / "counter.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(root / "csrc"), str(source), "-o", str(library)],
        check=True,
    )
    module = ctypes.CDLL(str(library))
    function = module.count
    function.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int16), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    function.restype = None
    return function


def test_cpu_plan_counts_strided_rows_and_replay_updates(counter):
    counts = (ctypes.c_int64 * 2)(0, 0)
    requests = (ctypes.c_int64 * 2)(5, -1)
    sources = (ctypes.c_int32 * 8)(1, 2, 10, -1, 1, 2, 3, 4)
    prefixes = (ctypes.c_int32 * 2)(10, 10)
    # Padded row stride; invalid requests and tail/padding are excluded.
    plan = (ctypes.c_int16 * 12)(1, -2, 3, 0, 99, 99, 1, 2, 3, 4, 99, 99)
    counter(counts, requests, sources, prefixes, plan, 6, 2, 4)
    assert list(counts) == [2, 1]
    plan[1] = 2
    counter(counts, requests, sources, prefixes, plan, 6, 2, 4)
    assert list(counts) == [4, 3]


def test_cpu_plan_counts_cold_misses_duplicates_and_empty(counter):
    counts = (ctypes.c_int64 * 2)(0, 0)
    requests = (ctypes.c_int64 * 1)(7)
    sources = (ctypes.c_int32 * 4)(1, 1, 2, -1)
    prefixes = (ctypes.c_int32 * 1)(3)
    plan = (ctypes.c_int16 * 4)(-1, -1, -2, 0)
    counter(counts, requests, sources, prefixes, plan, 4, 1, 4)
    assert list(counts) == [3, 0]
    counter(counts, requests, sources, prefixes, plan, 4, 0, 4)
    assert list(counts) == [3, 0]
