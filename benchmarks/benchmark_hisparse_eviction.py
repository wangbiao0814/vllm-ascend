# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare arch22 HiSparse eviction on deterministic Decode traces."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from statistics import median
from typing import NamedTuple

SCENARIOS = ("sliding-window", "reuse-plus-scan", "cyclic-overflow", "disjoint")


def make_trace(scenario, rounds, queries, topk, hot_size):
    """Return row-major Top-K selections for each Decode round."""
    union_size = queries * topk
    for step in range(rounds):
        if scenario == "sliding-window":
            start = step * max(1, topk // 4)
            yield [start + (row * (topk // 2)) + col for row in range(queries) for col in range(topk)]
        elif scenario == "reuse-plus-scan" and step % 4 < 2:
            yield [(row * (topk // 2) + col) % (2 * topk) for row in range(queries) for col in range(topk)]
        elif scenario == "cyclic-overflow":
            yield [(step * union_size + pos) % (hot_size + topk) for pos in range(union_size)]
        else:
            start = 2 * topk + step * union_size
            yield list(range(start, start + union_size))


def make_hit_rate_trace(rounds, queries, topk, hit_rate):
    """Yield a priming round, then full unions with a fixed number of Hot hits."""
    width = queries * topk
    hits = round(width * hit_rate)
    yield list(range(width))
    for step in range(rounds):
        reused = iter(range(hits))
        fresh = iter(range(width + step * (width - hits), width + (step + 1) * (width - hits)))
        yield [
            next(reused) if (pos + 1) * hits // width != pos * hits // width else next(fresh) for pos in range(width)
        ]


class Reference:
    """Track distinct Host-row lookups, hits, and the expected swap plan."""

    def __init__(self, hot_size, probation_age):
        self.probation_age = probation_age
        self.tags = [-1] * hot_size
        self.order = list(range(hot_size))
        self.ages = [0] * hot_size
        self.lookups = 0
        self.misses = 0

    def resolve(self, indices):
        wanted = dict.fromkeys(token for token in indices if token >= 0)
        if not wanted:
            if self.probation_age is not None:
                for slot, tag in enumerate(self.tags):
                    if tag >= 0:
                        self.ages[slot] = min(32767, self.ages[slot] + 1)
            return [], [-1] * len(indices)
        hits, available = [], []
        hot = {}
        for slot in self.order:
            tag = self.tags[slot]
            if tag in wanted:
                hits.append(slot)
                hot[tag] = min(hot.get(tag, slot), slot)
            else:
                available.append(slot)

        if self.probation_age is not None:
            available = [slot for slot in available if self.tags[slot] < 0] + [
                slot for slot in available if self.tags[slot] >= 0
            ]
        missing = wanted.keys() - hot.keys()
        refill = available[: len(missing)]
        survivors = available[len(missing) :]
        plan = list(zip((tag for tag in wanted if tag in missing), refill))
        for tag, slot in plan:
            self.tags[slot] = tag
            hot[tag] = slot

        if self.probation_age is not None:
            for slot in survivors:
                if self.tags[slot] >= 0:
                    self.ages[slot] = min(32767, self.ages[slot] + 1)
            for slot in hits:
                self.ages[slot] = 0
            for slot in refill:
                self.ages[slot] = self.probation_age
            split = next(
                (
                    i
                    for i, slot in enumerate(survivors)
                    if self.tags[slot] >= 0 and self.ages[slot] < self.probation_age
                ),
                len(survivors),
            )
        else:
            split = len(survivors)
        self.order = survivors[:split] + refill + survivors[split:] + hits
        self.lookups += len(wanted)
        self.misses += len(plan)
        return plan, [hot.get(token, -1) for token in indices]

    @property
    def hit_rate(self):
        return (self.lookups - self.misses) / self.lookups


class TraceStats(NamedTuple):
    lookups: int
    misses: int

    @property
    def hit_rate(self):
        return (self.lookups - self.misses) / self.lookups


class NativeResolver:
    """Call only the native resolver. Host payload copies are outside the timed region."""

    def __init__(
        self,
        queries,
        topk,
        hot_size,
        host_size,
        probation_age,
        *,
        lazy_age=False,
        age_options=0,
        hash_limit=0,
        num_requests=1,
    ):
        import torch
        import torch_npu  # noqa: F401

        from vllm_ascend.hisparse.cache_ops import AGE_REUSE_TAG_SCRATCH, PENDING_AGE_ROW_WIDTH
        from vllm_ascend.utils import bootstrap_custom_op_env

        # The extension reads and caches custom OPP paths when it is loaded.
        bootstrap_custom_op_env()
        if not hasattr(torch.ops._C_ascend, "hisparse_resolve_residency"):
            import vllm_ascend.vllm_ascend_C  # noqa: F401

        self.torch = torch
        self.op = torch.ops._C_ascend.hisparse_resolve_residency
        self.device = f"npu:{torch.npu.current_device()}"
        self.num_requests = num_requests
        self.queries_per_request = queries
        self.max_union_rows = queries * topk
        rows = num_requests * queries
        self.shape = (rows, topk)
        self.host_size = host_size
        self.block = 128
        self.hot_size = hot_size
        self.probation_age = probation_age
        self.age_options = age_options
        self.hash_limit = hash_limit
        self.hash_capacity = torch.ops._C_ascend.hisparse_resolver_validate_scratch(
            hot_size, self.max_union_rows, probation_age is not None, bool(age_options & AGE_REUSE_TAG_SCRATCH)
        )
        if hash_limit:
            self.hash_capacity = min(self.hash_capacity, hash_limit)
        self.pending_age = (
            torch.zeros((num_requests, PENDING_AGE_ROW_WIDTH), dtype=torch.int32, device=self.device)
            if lazy_age
            else None
        )
        self.requests = torch.arange(num_requests, dtype=torch.int32).repeat_interleave(queries).to(self.device)
        self.valid_rows = torch.tensor([rows], dtype=torch.int32, device=self.device)
        blocks = hot_size // self.block
        self.table = torch.arange(num_requests * blocks, dtype=torch.int32, device=self.device).view(
            num_requests, blocks
        )
        self.tags = torch.empty((num_requests, hot_size), dtype=torch.int32, device=self.device)
        self.order = torch.empty_like(self.tags, dtype=torch.int16)
        self.initial_order = torch.arange(hot_size, dtype=torch.int32, device=self.device).to(torch.int16)
        self.ages = torch.empty_like(self.order) if probation_age is not None else None
        self.out, self.resolved, self.host, self.victims = (
            torch.empty(self.shape, dtype=torch.int32, device=self.device) for _ in range(4)
        )
        self.counts = torch.empty(rows, dtype=torch.int32, device=self.device)
        self.errors = torch.empty((rows, 8), dtype=torch.int32, device=self.device)

    def reset(self):
        self.tags.fill_(-1)
        self.order.copy_(self.initial_order.expand_as(self.order))
        if self.ages is not None:
            self.ages.zero_()
        if self.pending_age is not None:
            self.pending_age.zero_()

    def tensor(self, values):
        return self.torch.tensor(values, dtype=self.torch.int32).view(self.shape).to(self.device)

    def resolve(self, logical, source=None, resident=None):
        self.op(
            logical,
            self.table,
            self.requests,
            None,
            source,
            resident,
            self.valid_rows,
            self.tags,
            self.order,
            self.out,
            self.resolved,
            self.errors,
            None,
            None,
            self.host,
            self.victims,
            self.counts,
            None,
            None,
            self.host_size,
            self.block,
            self.block if source is not None else 0,
            self.block if resident is not None else 0,
            0,
            self.block,
            self.max_union_rows,
            0,
            self.ages,
            4 if self.probation_age is None else self.probation_age,
            self.pending_age,
            self.age_options,
            self.hash_limit,
        )

    def verify(self, trace, inputs, *, priming_rounds=0):
        self.reset()
        refs = [Reference(self.hot_size, self.probation_age) for _ in range(self.num_requests)]
        priming = TraceStats(0, 0)
        for step, (selection, logical) in enumerate(zip(trace, inputs)):
            if step == priming_rounds:
                priming = TraceStats(sum(ref.lookups for ref in refs), sum(ref.misses for ref in refs))
            self.resolve(logical)
            assert not self.errors[:, 0].cpu().any(), f"resolver failed at step {step}"
            counts = self.counts.tolist()
            host = self.host.flatten().tolist()
            victims = self.victims.flatten().tolist()
            physical = self.out.flatten().tolist()
            tags, order = self.tags.tolist(), self.order.tolist()
            ages = self.ages.tolist() if self.ages is not None else None
            pending = self.pending_age[:, 0].tolist() if self.pending_age is not None else [0] * self.num_requests
            for request, ref in enumerate(refs):
                start = request * self.max_union_rows
                stop = start + self.max_union_rows
                first_row = request * self.queries_per_request
                plan, expected = ref.resolve(selection[start:stop])
                misses = sum(counts[first_row : first_row + self.queries_per_request])
                offset = request * self.hot_size
                where = (step, request)
                assert misses == len(plan), where
                assert list(zip(host[start : start + misses], victims[start : start + misses])) == [
                    (token, slot + offset) for token, slot in plan
                ], where
                assert physical[start:stop] == [slot + offset if slot >= 0 else -1 for slot in expected], where
                assert tags[request] == ref.tags, where
                assert order[request] == ref.order, where
                if ages is not None:
                    assert all(
                        min(32767, ages[request][i] + pending[request]) == ref.ages[i]
                        for i, tag in enumerate(ref.tags)
                        if tag >= 0
                    ), where
        return TraceStats(
            sum(ref.lookups for ref in refs) - priming.lookups,
            sum(ref.misses for ref in refs) - priming.misses,
        )

    def time_trace(
        self,
        inputs,
        repeats,
        graph_mode,
        *,
        source=None,
        resident=None,
        gather: Callable[[], None] | None = None,
        priming_inputs=(),
    ):
        def step(indices):
            self.resolve(indices, source, resident)
            if gather is not None:
                gather()

        def reset_and_prime():
            self.reset()
            for indices in priming_inputs:
                step(indices)

        torch = self.torch
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        graph = None
        samples = []
        try:
            with torch.npu.stream(stream):
                for _ in range(2):
                    reset_and_prime()
                    for indices in inputs:
                        step(indices)
                if graph_mode == "graph":
                    reset_and_prime()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        for indices in inputs:
                            step(indices)
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                for _ in range(repeats):
                    reset_and_prime()
                    start.record()
                    if graph is None:
                        for indices in inputs:
                            step(indices)
                    else:
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / len(inputs))
        finally:
            stream.synchronize()
            if graph is not None:
                graph.reset()
        return median(samples)


def print_latency(times, execution):
    baseline, updated = times
    print(
        f"  {execution} resolver median per round: LRU={baseline:.2f} us, "
        f"age={updated:.2f} us, delta={updated - baseline:+.2f} us ({(updated / baseline - 1) * 100:+.2f}%)"
    )


def benchmark_resident_only(args, queries, hot_size):
    print("resident-only: all selected rows hit Resident; zero Host lookups")
    if args.device == "cpu":
        return
    times = []
    for probation in (None, args.probation_age):
        resolver = NativeResolver(queries, args.topk, hot_size, 256, probation)
        tensor = resolver.torch.tensor
        dtype = resolver.torch.int32
        source = tensor([[1]], dtype=dtype, device=resolver.device)
        resident = tensor([[2]], dtype=dtype, device=resolver.device)
        selection = [i % 128 for i in range(queries * args.topk)]
        indices = resolver.tensor(selection)
        resolver.reset()
        resolver.resolve(indices, source, resident)
        assert not resolver.counts.any().item()
        assert resolver.out.flatten().tolist() == [256 + i for i in selection]
        times.append(
            resolver.time_trace([indices] * args.rounds, args.repeats, args.execution, source=source, resident=resident)
        )
    print_latency(times, args.execution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "resident-only"), action="append", default=None)
    parser.add_argument("--mtp", type=int, default=3)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--hot-size", type=int)
    parser.add_argument("--probation-age", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    queries = args.mtp + 1
    hot_size = args.hot_size if args.hot_size is not None else (queries + 1) * args.topk
    if (
        queries <= 0
        or args.topk <= 0
        or not queries * args.topk <= hot_size <= 32768
        or args.rounds <= 0
        or args.repeats <= 0
        or not 0 <= args.probation_age <= 32767
        or (args.device == "npu" and hot_size % 128)
    ):
        parser.error("invalid geometry, repeat count or probation age (NPU Hot size must be a multiple of 128)")
    for scenario in args.scenario or (*SCENARIOS, "resident-only"):
        if scenario == "resident-only":
            benchmark_resident_only(args, queries, hot_size)
            continue
        trace = list(make_trace(scenario, args.rounds, queries, args.topk, hot_size))
        stats, times = [], []
        for probation in (None, args.probation_age):
            if args.device == "npu":
                resolver = NativeResolver(queries, args.topk, hot_size, max(map(max, trace)) + 1, probation)
                inputs = [resolver.tensor(selection) for selection in trace]
                ref = resolver.verify(trace, inputs)
                times.append(resolver.time_trace(inputs, args.repeats, args.execution))
            else:
                ref = Reference(hot_size, probation)
                for selection in trace:
                    ref.resolve(selection)
            stats.append(ref)
        lru, age = stats
        print(
            f"{scenario}: distinct Host lookups={lru.lookups}; "
            f"LRU hit={lru.hit_rate:.2%} ({lru.misses} misses); age hit={age.hit_rate:.2%} ({age.misses} misses); "
            f"delta={(age.hit_rate - lru.hit_rate) * 100:+.2f} pp"
        )
        if times:
            print_latency(times, args.execution)


if __name__ == "__main__":
    main()
