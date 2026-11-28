# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the cost of each HiSparse eviction optimization on A2/A3."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from benchmarks.benchmark_hisparse_eviction import SCENARIOS, NativeResolver, make_trace


@dataclass(frozen=True)
class Variant:
    with_age: bool = True
    lazy: bool = True
    grouped: bool = False
    reuse_ub: bool = False
    match_hash: bool = False
    vector_age: bool = False
    gather_mask: bool = False

    @property
    def options(self):
        from vllm_ascend.hisparse.cache_ops import (
            AGE_GROUP_CANDIDATES,
            AGE_REUSE_TAG_SCRATCH,
            AGE_VECTOR_GROUPING,
            AGE_VECTOR_INCREMENT,
        )

        return (
            (AGE_GROUP_CANDIDATES if self.grouped else 0)
            | (AGE_REUSE_TAG_SCRATCH if self.reuse_ub else 0)
            | (AGE_VECTOR_INCREMENT if self.vector_age else 0)
            | (AGE_VECTOR_GROUPING if self.gather_mask else 0)
        )


VARIANTS = {
    "lru": Variant(with_age=False, lazy=False),
    "lru-matched": Variant(with_age=False, lazy=False, match_hash=True),
    "age-original": Variant(lazy=False),
    "age-lazy": Variant(),
    "age-grouped": Variant(grouped=True),
    "age-reuse-ub-matched": Variant(grouped=True, reuse_ub=True, match_hash=True),
    "age-reuse-ub": Variant(grouped=True, reuse_ub=True),
    "age-vector-age": Variant(grouped=True, reuse_ub=True, vector_age=True),
    "age-gather-mask": Variant(grouped=True, reuse_ub=True, gather_mask=True),
    "age-vector-both": Variant(grouped=True, reuse_ub=True, vector_age=True, gather_mask=True),
}


def load_trace(path, queries, topk):
    """Load either flattened rounds or rounds containing one list per query."""
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("trace must contain at least one Decode round")
    trace = []
    for index, raw in enumerate(data):
        if not isinstance(raw, list):
            raise ValueError(f"round {index} must be a list")
        if len(raw) == queries and all(isinstance(row, list) and len(row) == topk for row in raw):
            raw = [token for row in raw for token in row]
        if len(raw) != queries * topk or any(type(x) is not int or not -1 <= x < 2**31 - 1 for x in raw):
            raise ValueError(f"round {index} must contain {queries * topk} int32 IDs >= -1")
        trace.append(raw)
    return trace


@contextmanager
def mapped_gather(resolver, memfabric):
    from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
    from vllm_ascend.hisparse.cache_ops import HiSparseNPUOps
    from vllm_ascend.hisparse.host_pool import allocate_mapped_host

    torch = resolver.torch
    host = allocate_mapped_host((resolver.host_size, C8_ROW_BYTES), torch.int8, memfabric=memfabric)
    mapping = host._hisparse_registration
    try:
        # Touch all pages outside the timed region, including for disjoint traces.
        host.fill_(1)
        hot = torch.empty(
            (resolver.hot_size // resolver.block, resolver.block, C8_ROW_BYTES),
            dtype=torch.int8,
            device=resolver.device,
        )
        ops = HiSparseNPUOps(mapping)

        def gather():
            ops.hisparse_gather_compact(host, hot, resolver.host, resolver.victims, resolver.counts)

        yield gather
    finally:
        torch.npu.synchronize()
        mapping.close()


def run_scenario(args, name, trace, hot_size, queries, probation, variants):
    resident_only = name == "resident-only"
    host_size = 256 if resident_only else max(1, max(map(max, trace)) + 1)
    # Discover the actual age-constrained hash size on this SoC. No hard-coded
    # UB budget or shape-specific 9408 in the matched variants.
    matched_hash = 0
    if any(VARIANTS[variant].match_hash for variant in variants):
        probe = NativeResolver(queries, args.topk, hot_size, host_size, probation)
        matched_hash = probe.hash_capacity
        del probe
    rows = []

    for variant_name in variants:
        variant = VARIANTS[variant_name]
        resolver = NativeResolver(
            queries,
            args.topk,
            hot_size,
            host_size,
            probation if variant.with_age else None,
            lazy_age=variant.lazy,
            age_options=variant.options,
            hash_limit=matched_hash if variant.match_hash else 0,
        )
        source = resident = None
        if resident_only:
            torch = resolver.torch
            source = torch.tensor([[1]], dtype=torch.int32, device=resolver.device)
            resident = torch.tensor([[2]], dtype=torch.int32, device=resolver.device)
            selection = [i % 128 for i in range(queries * args.topk)]
            indices = resolver.tensor(selection)
            inputs = [indices] * args.rounds
            resolver.reset()
            resolver.resolve(indices, source, resident)
            assert not resolver.counts.any().item()
            assert resolver.out.flatten().tolist() == [256 + i for i in selection]
            lookups = misses = 0
        else:
            inputs = [resolver.tensor(selection) for selection in trace]
            ref = resolver.verify(trace, inputs)
            lookups, misses = ref.lookups, ref.misses

        row = {
            "scenario": name,
            "probation_age": probation if variant.with_age else None,
            "variant": variant_name,
            "execution": args.execution,
            "queries": queries,
            "topk": args.topk,
            "hot_size": hot_size,
            "rounds": len(inputs),
            "hash_capacity": resolver.hash_capacity,
            "lookups": lookups,
            "misses": misses,
            "hit_rate": (lookups - misses) / lookups if lookups else None,
        }
        if args.measure in ("resolver", "both"):
            row["resolver_us"] = resolver.time_trace(
                inputs, args.repeats, args.execution, source=source, resident=resident
            )
        if args.measure in ("combined", "both"):
            with mapped_gather(resolver, args.memfabric) as gather:
                row["combined_us"] = resolver.time_trace(
                    inputs, args.repeats, args.execution, source=source, resident=resident, gather=gather
                )
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--measure", choices=("resolver", "combined", "both"), default="resolver")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "resident-only"), action="append")
    parser.add_argument("--trace", type=Path, help="JSON Top-K IDs with fixed query count and width per round")
    parser.add_argument("--output", type=Path, help="write the measurement rows as JSON")
    parser.add_argument("--mtp", type=int, default=3)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--hot-size", type=int)
    parser.add_argument("--probation-age", type=int, action="append", help="repeat to sweep multiple values")
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--memfabric", action="store_true", help="use MemFabric for the Host KV mapping")
    args = parser.parse_args()
    queries = args.mtp + 1
    hot_size = args.hot_size if args.hot_size is not None else (queries + 1) * args.topk
    ages = args.probation_age or [4]
    if (
        queries <= 0
        or args.topk <= 0
        or not queries * args.topk <= hot_size <= 32768
        or hot_size % 128
        or args.rounds <= 0
        or args.repeats <= 0
        or any(not 0 <= age <= 32767 for age in ages)
    ):
        parser.error("invalid resolver geometry, repeat count or age; Hot size must align to 128")
    if args.trace and args.scenario:
        parser.error("select --trace or --scenario")
    if args.trace:
        try:
            scenarios = [(str(args.trace), load_trace(args.trace, queries, args.topk))]
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        scenarios = [
            (
                name,
                None if name == "resident-only" else list(make_trace(name, args.rounds, queries, args.topk, hot_size)),
            )
            for name in (args.scenario or (*SCENARIOS, "resident-only"))
        ]
    variants = tuple(dict.fromkeys(args.variant or VARIANTS))
    results = []
    for name, trace in scenarios:
        for i, age in enumerate(ages):
            selected = [variant for variant in variants if i == 0 or VARIANTS[variant].with_age]
            results.extend(run_scenario(args, name, trace, hot_size, queries, age, selected))
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
