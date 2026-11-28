# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the cost of each HiSparse eviction optimization on A2/A3."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from benchmarks.benchmark_hisparse_eviction import SCENARIOS, NativeResolver, Reference, make_hit_rate_trace, make_trace


@dataclass(frozen=True)
class Variant:
    with_age: bool = True
    lazy: bool = True
    grouped: bool = False
    reuse_ub: bool = False
    match_hash: bool = False
    vector_age: bool = False
    gather_mask: bool = False
    dense_refill: bool = False
    async_writeback: bool = False

    @property
    def options(self):
        from vllm_ascend.hisparse.cache_ops import (
            AGE_ASYNC_WRITEBACK,
            AGE_DENSE_REFILL,
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
            | (AGE_DENSE_REFILL if self.dense_refill else 0)
            | (AGE_ASYNC_WRITEBACK if self.async_writeback else 0)
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
    "age-dense-refill": Variant(grouped=True, reuse_ub=True, dense_refill=True),
    "age-async-write": Variant(grouped=True, reuse_ub=True, async_writeback=True),
    "age-dense-refill-async": Variant(grouped=True, reuse_ub=True, dense_refill=True, async_writeback=True),
    "age-gather-mask": Variant(grouped=True, reuse_ub=True, gather_mask=True),
    "age-vector-both": Variant(grouped=True, reuse_ub=True, vector_age=True, gather_mask=True),
}


WRITEBACK_CONTROLS = {
    "age-async-write": "age-reuse-ub",
    "age-dense-refill-async": "age-dense-refill",
}


def add_latency_deltas(row, baseline, *, prefix=""):
    for metric in ("resolver", "combined"):
        key = f"{metric}_us"
        if key in baseline and key in row:
            row[f"{prefix}{metric}_delta_us"] = row[key] - baseline[key]
            row[f"{prefix}{metric}_overhead_pct"] = (row[key] / baseline[key] - 1) * 100


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


def expand_request_trace(trace, num_requests):
    """Clone the workload per request, using disjoint Host row ranges."""
    host_rows_per_request = max(1, max(map(max, trace)) + 1)
    host_size = host_rows_per_request * num_requests
    if host_size > 2**31 - 1:
        raise ValueError("batched Host row range exceeds int32")
    if num_requests == 1:
        return trace, host_size
    offsets = range(0, host_size, host_rows_per_request)
    return [
        [token + offset if token >= 0 else -1 for offset in offsets for token in selection] for selection in trace
    ], host_size


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
            (resolver.num_requests * (resolver.hot_size // resolver.block), resolver.block, C8_ROW_BYTES),
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


def run_scenario(args, name, trace, hot_size, queries, probation, variants, *, target_hit_rate=None):
    resident_only = name == "resident-only"
    priming_rounds = int(target_hit_rate is not None)
    host_size = (args.requests + 1) * 128
    if not resident_only:
        trace, host_size = expand_request_trace(trace, args.requests)

    # Discover the actual age-constrained hash size on this SoC.
    matched_hash = 0
    if args.device == "npu" and any(VARIANTS[variant].match_hash for variant in variants):
        probe = NativeResolver(queries, args.topk, hot_size, host_size, probation)
        matched_hash = probe.hash_capacity
        del probe
    rows = []

    for variant_name in variants:
        variant = VARIANTS[variant_name]
        age = probation if variant.with_age else None
        row = {
            "scenario": name,
            "probation_age": age,
            "variant": variant_name,
            "device": args.device,
            "execution": args.execution if args.device == "npu" else None,
            "requests": args.requests,
            "queries": queries,
            "total_queries": queries * args.requests,
            "topk": args.topk,
            "hot_size": hot_size,
            "rounds": args.rounds if resident_only else len(trace) - priming_rounds,
            "target_hit_rate": target_hit_rate,
        }
        if args.device == "cpu":
            lookups, misses = reference_stats(trace, queries * args.topk, hot_size, age, priming_rounds)
        else:
            resolver = NativeResolver(
                queries,
                args.topk,
                hot_size,
                host_size,
                age,
                lazy_age=variant.lazy,
                age_options=variant.options,
                hash_limit=matched_hash if variant.match_hash else 0,
                num_requests=args.requests,
            )
            row["hash_capacity"] = resolver.hash_capacity
            source = resident = None
            if resident_only:
                torch = resolver.torch
                pages = torch.arange(1, args.requests + 1, dtype=torch.int32, device=resolver.device)[:, None]
                source, resident = pages, pages + 1
                selection = [i % resolver.block for i in range(resolver.max_union_rows)]
                indices = resolver.tensor(selection * args.requests)
                inputs = [indices] * args.rounds
                resolver.reset()
                resolver.resolve(indices, source, resident)
                assert not resolver.counts.any().item()
                expected = [
                    (request + 2) * resolver.block + token for request in range(args.requests) for token in selection
                ]
                assert resolver.out.flatten().tolist() == expected
                lookups = misses = 0
            else:
                inputs = [resolver.tensor(selection) for selection in trace]
                lookups, misses = resolver.verify(trace, inputs, priming_rounds=priming_rounds)

            timed_inputs, priming_inputs = inputs[priming_rounds:], inputs[:priming_rounds]
            kwargs = dict(source=source, resident=resident, priming_inputs=priming_inputs)
            if args.measure in ("resolver", "both"):
                row["resolver_us"] = resolver.time_trace(timed_inputs, args.repeats, args.execution, **kwargs)
            if args.measure in ("combined", "both"):
                with mapped_gather(resolver, args.memfabric) as gather:
                    row["combined_us"] = resolver.time_trace(
                        timed_inputs, args.repeats, args.execution, gather=gather, **kwargs
                    )
        row.update(lookups=lookups, misses=misses, hit_rate=(lookups - misses) / lookups if lookups else None)
        if target_hit_rate is not None:
            union = queries * args.topk
            assert lookups == args.requests * args.rounds * union
            assert misses == args.requests * args.rounds * (union - round(union * target_hit_rate))
        rows.append(row)
    by_variant = {row["variant"]: row for row in rows}
    lru = by_variant.get("lru", {})
    for row in rows:
        if target_hit_rate is not None:
            add_latency_deltas(row, lru)
        control_name = WRITEBACK_CONTROLS.get(row["variant"])
        if control_name in by_variant:
            control = by_variant[control_name]
            assert all(row.get(key) == control.get(key) for key in ("hash_capacity", "lookups", "misses"))
            row["writeback_control"] = control_name
            add_latency_deltas(row, control, prefix="writeback_")
        print(json.dumps(row), flush=True)
    return rows


def reference_stats(trace, union, hot_size, probation_age, priming_rounds):
    requests = len(trace[0]) // union
    refs = [Reference(hot_size, probation_age) for _ in range(requests)]
    lookups = misses = 0
    for step, selection in enumerate(trace):
        for request, ref in enumerate(refs):
            start = request * union
            previous_lookups = ref.lookups
            plan, _ = ref.resolve(selection[start : start + union])
            if step >= priming_rounds:
                lookups += ref.lookups - previous_lookups
                misses += len(plan)
    return lookups, misses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("npu", "cpu"), default="npu", help="CPU validates hit-rate sweeps without timing"
    )
    parser.add_argument(
        "--hit-rate", type=float, action="append", help="sweep synthetic steady-state Hot hit fractions from 0 to 1"
    )
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--measure", choices=("resolver", "combined", "both"), default="resolver")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "resident-only"), action="append")
    parser.add_argument("--trace", type=Path, help="JSON Top-K IDs with fixed query count and width per round")
    parser.add_argument("--output", type=Path, help="write the measurement rows as JSON")
    parser.add_argument("--mtp", type=int, default=3)
    parser.add_argument("--requests", type=int, default=1, help="requests per resolver batch")
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
        or args.requests <= 0
        or queries * args.requests > 2**31 - 1
        or args.topk <= 0
        or not queries * args.topk <= hot_size <= 32768
        or hot_size % 128
        or args.rounds <= 0
        or args.repeats <= 0
        or any(not 0 <= age <= 32767 for age in ages)
    ):
        parser.error("invalid resolver geometry, repeat count or age; Hot size must align to 128")
    if sum(bool(option) for option in (args.trace, args.scenario, args.hit_rate)) > 1:
        parser.error("select --trace, --scenario or --hit-rate")
    if args.device == "cpu" and (not args.hit_rate or args.measure != "resolver"):
        parser.error("CPU mode validates --hit-rate without NPU or gather timing")
    if args.hit_rate:
        if any(not 0 <= rate <= 1 for rate in args.hit_rate):
            parser.error("--hit-rate values must be fractions between 0 and 1")
        scenarios = [
            ("hit-rate-sweep", list(make_hit_rate_trace(args.rounds, queries, args.topk, rate)), rate)
            for rate in dict.fromkeys(args.hit_rate)
        ]
    elif args.trace:
        try:
            scenarios = [(str(args.trace), load_trace(args.trace, queries, args.topk), None)]
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        scenarios = [
            (
                name,
                None if name == "resident-only" else list(make_trace(name, args.rounds, queries, args.topk, hot_size)),
                None,
            )
            for name in (args.scenario or (*SCENARIOS, "resident-only"))
        ]
    default_variants = ("lru", "age-reuse-ub", "age-vector-age") if args.hit_rate else VARIANTS
    variants = tuple(dict.fromkeys(args.variant or default_variants))
    results = []
    for name, trace, target_hit_rate in scenarios:
        for i, age in enumerate(ages):
            selected = [variant for variant in variants if i == 0 or VARIANTS[variant].with_age]
            results.extend(
                run_scenario(
                    args,
                    name,
                    trace,
                    hot_size,
                    queries,
                    age,
                    selected,
                    target_hit_rate=target_hit_rate,
                )
            )
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
