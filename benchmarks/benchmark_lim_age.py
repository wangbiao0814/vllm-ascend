# SPDX-License-Identifier: Apache-2.0
"""Compare score/age LIM kernels on the same synthetic single-query trace."""

import argparse
import json
import random
import statistics
import tempfile
from pathlib import Path

TOPK = 2048
HEAD_DIM = 128
HEADS = 32
HIT_CLEAR_BREAKDOWN = (
    "no-increment-or-hit-clear",
    "no-increment-hit-buckets-only",
    "no-increment-hit-copy-only",
    "no-increment",
)
HIT_GROUP_BREAKDOWN = (
    "no-increment-or-hit-clear",
    "no-increment-hit-sort-only",
    "scalar-no-increment-hit-buckets-only",
    "no-increment-hit-buckets-only",
    "scalar-no-increment",
    "no-increment",
)
FINISH_BREAKDOWN = (
    "normal",
    "no-refill",
    "finish-check-only",
    "finish-count-only",
    "scalar-hit-runs",
    "scalar-no-refill",
    "scalar-finish-check-only",
    "scalar-finish-count-only",
)
PROBE_WORDS = 16
PROBE_MAGIC = 0x4C494D41
REPORT_PROBE = 128

ABLATIONS = {
    "normal": 0,
    "no-hit-clear": 1,
    "no-increment": 2,
    "no-increment-or-hit-clear": 3,
    "no-refill": 4,
    "no-age-writes": 7,
    "hit-buckets-only": 8,
    "hit-copy-only": 16,
    "no-increment-hit-buckets-only": 10,
    "no-increment-hit-copy-only": 18,
    "hit-sort-only": 32,
    "no-increment-hit-sort-only": 34,
    "finish-count-only": 256,
    "finish-check-only": 512,
    "scalar-no-refill": 68,
    "scalar-finish-count-only": 320,
    "scalar-finish-check-only": 576,
    "scalar-hit-runs": 64,
    "scalar-no-increment": 66,
    "scalar-no-increment-hit-buckets-only": 74,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["trace", "all-hit", "all-miss"], default="trace")
    parser.add_argument(
        "--ablation",
        choices=[*ABLATIONS, "all", "hit-clear-breakdown", "hit-group-breakdown", "finish-breakdown"],
        default="normal",
    )
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--latency-samples", type=int, default=4)
    parser.add_argument("--core-fraction", type=float, default=0.5)
    parser.add_argument("--correlation", type=float, default=0.95)
    parser.add_argument("--probation-age", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--mode", choices=["eager", "graph", "both"], default="both")
    parser.add_argument("--diagnostics", action="store_true", help="Untimed eager compaction/FinishAge probes")
    parser.add_argument("--probe-calls", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path, help="Profiler-only run; exports NPU kernel details and trace")
    parser.add_argument("--profile-calls", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("lim_age_results.json"))
    parser.add_argument("--dry-run", action="store_true", help="Validate the trace specification without torch/NPU")
    args = parser.parse_args()
    if not TOPK <= args.capacity <= min(args.length, 32640):
        parser.error("Require 2048 <= capacity <= min(length, 32640)")
    if args.scenario == "all-miss" and args.length < args.capacity + TOPK:
        parser.error("all-miss requires length >= capacity + 2048")
    if args.ablation not in ("normal", "scalar-hit-runs") and args.scenario == "trace":
        parser.error("Diagnostic ablations require all-hit or all-miss; they do not preserve the age policy")
    if args.length % 128 or args.capacity % 128:
        parser.error("length and capacity must be 128-aligned")
    if min(args.steps, args.repeats, args.latency_samples, args.probe_calls, args.profile_calls) < 1:
        parser.error("steps, repeats, latency-samples, probe-calls and profile-calls must be positive")
    if not 0 <= args.core_fraction <= 1 or not 0 <= args.correlation < 1:
        parser.error("Require core-fraction in [0,1] and correlation in [0,1)")
    if not 0 <= args.probation_age <= (1 << 19) - 1:
        parser.error("probation-age must be in [0,524287]")
    return args


def trace_spec(args):
    # Sample core sources across the history rather than concentrating in one scan block.
    rng = random.Random(args.seed)
    core_count = TOPK if args.scenario != "trace" else round(TOPK * args.core_fraction)
    core = sorted(rng.sample(range(args.length), core_count))
    core_set = set(core)
    background = [s for s in range(args.length) if s not in core_set]
    residents = (background if args.scenario == "all-miss" else core + background)[: args.capacity]
    return {
        "core_sources": core,
        "query_seeds": [rng.randrange(1 << 31) for _ in range(args.steps)],
        "query_shape": [1, HEADS, HEAD_DIM],
        "key_shape": [(args.length + 128) // 128, 128, 1, HEAD_DIM],
        "initial_resident_sources": residents if args.scenario != "trace" else list(range(args.capacity)),
        "fixed_query": args.scenario != "trace",
    }


def decode_probe(words):
    """Decode reserved output-tail counters; reject stale/uninstrumented binaries."""
    if len(words) != PROBE_WORDS or words[0] != PROBE_MAGIC or words[1] != 1:
        raise RuntimeError("Missing age diagnostic probe: rebuild the age operator and extension")
    paths = {0: "skipped", 1: "sort_only", 2: "scalar", 3: "vector", 4: "scalar_fallback", 7: "no_hits"}
    return {
        "raw_block_count": words[2],
        "raw_start_count": words[3],
        "raw_end_count": words[4],
        "fallback_count": words[5],
        "effective_runs": words[6],
        "path_code": words[7],
        "path": paths.get(words[7], "unknown"),
        "finish_count": words[8],
        "finish_phase": words[9],
    }


def collect_probes(torch, case, snapshot, invoke, make_outputs, flags, calls):
    cache_seed, age_seed, query = snapshot
    cache, ages = cache_seed.clone(), age_seed.clone()
    outputs = make_outputs(case)
    rows = []
    for _ in range(calls):
        cache.copy_(cache_seed)
        ages.copy_(age_seed)
        case.query.copy_(query)
        outputs[4][:, -PROBE_WORDS:].fill_(-1)
        invoke(cache=cache, ages=ages, outputs=outputs, flags=flags | REPORT_PROBE)
        torch.npu.synchronize()
        rows.append(decode_probe(outputs[4][0, -PROBE_WORDS:].cpu().tolist()))
    return {"mode": "eager", "calls": calls, "fallback_count": sum(r["fallback_count"] for r in rows), "rows": rows}


def collect_profile(torch, torch_npu, args, snapshot, cache, ages, case, launch, label):
    # Counter probes run separately: profile only the original variant, flags unchanged.
    args.profile_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=label + "-", dir=args.profile_dir)).resolve()
    cache_seed, age_seed, query = snapshot
    config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=config,
    ) as profiler:
        for call in range(args.profile_calls):
            cache.copy_(cache_seed)
            ages.copy_(age_seed)
            case.query.copy_(query)
            torch.npu.synchronize()
            with torch.profiler.record_function(f"LIM/{label}/call{call}"):
                launch()
                torch.npu.synchronize()
            profiler.step()
    return {
        "directory": str(directory),
        "calls": args.profile_calls,
        "kernel_details": [str(p) for p in directory.rglob("kernel_details.csv")],
        "traces": [str(p) for p in directory.rglob("trace_view.json")],
    }


def run(args, spec):
    # Deliberately import only for the real benchmark; dry-run works on a CPU-only Mac.
    import torch
    import torch_npu  # noqa: F401

    from tests.e2e.nightly.single_node.ops.singlecard_ops.test_fused_lightning_indexer_manage import (
        _assert_resident_bijection,
        _build_case,
        _call_op,
        _make_outputs,
    )

    instrumented = args.diagnostics or args.profile_dir is not None
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    case = _build_case(
        q_values=[1],
        states=[-1],
        offload_len=args.length,
        cache_tokens=args.capacity,
        dtype=dtype,
        seed=args.seed,
        random_block_table=False,
    )
    if not hasattr(torch.ops._C_ascend, "npu_fused_lightning_indexer_age_manage"):
        raise RuntimeError("Rebuild the extension with npu_fused_lightning_indexer_age_manage")
    generator = torch.Generator().manual_seed(args.seed)
    keys = torch.randn(args.length + 128, HEAD_DIM, generator=generator)
    keys[:, 0] = 0
    keys[spec["core_sources"], 0] = 32
    case.index_key_cache.copy_(keys.reshape(case.index_key_cache.shape).to(device="npu", dtype=dtype))
    case.index_weights.zero_()
    case.index_weights[:, 0] = 1
    queries = []
    previous = None
    for seed in spec["query_seeds"]:
        vector = torch.randn(HEAD_DIM - 1, generator=torch.Generator().manual_seed(seed))
        if previous is not None:
            vector = args.correlation * previous + (1 - args.correlation**2) ** 0.5 * vector
        previous = vector
        query = torch.cat((torch.ones(1), vector / vector.norm()))
        queries.append(query.expand(1, HEADS, HEAD_DIM).contiguous().to(device="npu", dtype=dtype))
    if args.scenario != "trace":
        # Exactly TOPK positive-score cores; every other key scores zero.
        # Fix Query to e0 so the selected set cannot drift between steps.
        fixed_query = torch.zeros_like(queries[0])
        fixed_query[:, :, 0] = 1
        queries = [fixed_query] * args.steps
        row = case.req_entries[0]
        case.cache_seed[row].fill_(-(1 << 31))
        resident_sources = torch.tensor(spec["initial_resident_sources"], dtype=torch.long, device="npu")
        case.cache_seed[row, resident_sources] = torch.arange(args.capacity, dtype=torch.int32, device="npu")
    samples = min(args.steps, args.latency_samples)
    sample_steps = {round(i * (args.steps - 1) / max(1, samples - 1)) for i in range(samples)}
    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    results = []
    selections = {}
    with torch.inference_mode():
        if args.ablation == "finish-breakdown":
            names = FINISH_BREAKDOWN
        elif args.ablation == "hit-group-breakdown":
            names = HIT_GROUP_BREAKDOWN
        elif args.ablation == "hit-clear-breakdown":
            names = HIT_CLEAR_BREAKDOWN
        else:
            names = ABLATIONS if args.ablation == "all" else [args.ablation]
        variants = [("score", "baseline")] + [("age", name) for name in names]
        for mode in modes:
            for policy, ablation in variants:
                flags = ABLATIONS.get(ablation, 0)
                cache = case.cache_seed.clone()
                ages = torch.full_like(cache, args.probation_age)
                outputs = _make_outputs(case)

                def invoke(policy=policy, cache=cache, outputs=outputs, ages=ages, flags=flags):
                    if policy == "score":
                        _call_op(case, cache, outputs)
                    else:
                        torch.ops._C_ascend.npu_fused_lightning_indexer_age_manage(
                            case.index_weights,
                            case.query_dequant_scale,
                            case.query,
                            case.index_key_dequant_scale,
                            case.index_key_cache,
                            case.index_block_table,
                            case.actual_seq_lengths_query,
                            case.actual_seq_lengths_key,
                            case.offload_seq_lengths_key,
                            case.num_cache_tokens,
                            case.request_state,
                            case.req_pool_entries,
                            cache,
                            *outputs,
                            ages,
                            args.probation_age,
                            flags,
                        )

                case.query.copy_(queries[0])
                for _ in range(3):
                    invoke()
                torch.npu.synchronize()
                graph = None
                if mode == "graph":
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        invoke()
                launch = graph.replay if graph is not None else invoke
                cache.copy_(case.cache_seed)
                ages.fill_(args.probation_age)
                torch.npu.synchronize()
                latency = []
                misses = []
                snapshot = None
                for step, query in enumerate(queries):
                    case.query.copy_(query)
                    if args.scenario == "all-miss":
                        # Reset outside timing so every call misses, not just the first one.
                        cache.copy_(case.cache_seed)
                        ages.fill_(args.probation_age)
                    if step in sample_steps:
                        # Restore the same pre-step state before each timed call.
                        saved_cache, saved_ages = cache.clone(), ages.clone()
                        snapshot = (saved_cache, saved_ages, query.clone())
                    if step in sample_steps and not instrumented:
                        events = []
                        for _ in range(args.repeats):
                            cache.copy_(saved_cache)
                            ages.copy_(saved_ages)
                            begin = torch.npu.Event(enable_timing=True)
                            end = torch.npu.Event(enable_timing=True)
                            begin.record()
                            launch()
                            end.record()
                            events.append((begin, end))
                        torch.npu.synchronize()
                        latency.extend(a.elapsed_time(b) * 1000 for a, b in events)
                    else:
                        launch()
                    # The last timed call also advances the trajectory by exactly one step.
                    torch.npu.synchronize()
                    sources = outputs[0][0, 0].cpu().long()
                    ordered = sources.sort().values.tolist()
                    key = (mode, step)
                    if policy == "score":
                        selections[key] = ordered
                    elif selections[key] != ordered:
                        raise AssertionError(f"TopK source sets differ at {mode=} {step=}")
                    count = int(outputs[2].cpu()[0])
                    assert 0 <= count <= TOPK
                    if args.scenario == "all-hit":
                        assert count == 0 and int(outputs[5].cpu()[0]) == 0, "All-hit setup produced misses"
                        assert ordered == spec["core_sources"], "All-hit TopK differs from planted core set"
                        assert torch.equal(cache, case.cache_seed), "All-hit call changed the cache map"
                    if args.scenario == "all-miss":
                        assert count == TOPK and int(outputs[5].cpu()[0]) == TOPK, "All-miss setup produced hits"
                        assert ordered == spec["core_sources"], "All-miss TopK differs from planted core set"
                    misses.append(count)
                    row = case.req_entries[0]
                    mapping = cache[row].cpu()
                    _assert_resident_bijection(mapping, args.length, args.capacity)
                    assert torch.equal(mapping[sources], outputs[1][0, 0].cpu())
                accesses = args.steps * TOPK
                sorted_latency = sorted(latency)
                mean = statistics.mean(latency) if latency else None
                median = statistics.median(latency) if latency else None
                p95 = sorted_latency[min(len(latency) - 1, int(len(latency) * 0.95))] if latency else None
                result = {
                    "mode": mode,
                    "policy": policy,
                    "ablation": ablation,
                    "diagnostic_flags": flags,
                    "policy_valid": (flags & ~64) == 0,
                    "hits": accesses - sum(misses),
                    "misses": sum(misses),
                    "accesses": accesses,
                    "hit_rate": 1 - sum(misses) / accesses,
                    "misses_per_step": misses,
                    "timing_scope": "not_measured" if instrumented else "event_interval",
                    "event_mean_us": mean,
                    "event_median_us": median,
                    "event_p95_us": p95,
                    "kernel_mean_us": mean,
                    "kernel_median_us": median,
                    "kernel_p95_us": p95,
                    "timed_calls": len(latency),
                }
                if args.profile_dir is not None:
                    result["profile"] = collect_profile(
                        torch, torch_npu, args, snapshot, cache, ages, case, launch, f"{mode}-{policy}-{ablation}"
                    )
                if args.diagnostics and policy == "age":
                    result["probe"] = collect_probes(
                        torch, case, snapshot, invoke, _make_outputs, flags, args.probe_calls
                    )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    return {
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "device": torch.npu.get_device_name(),
        "scope": "Event interval includes device waits/host submission gaps; input/reset copies are outside events. "
        "Legacy kernel_* fields alias event_*; use profiler kernel_details for actual kernel durations. "
        "Instrumented runs have no Event timing. Probes are eager and outside the profiler region.",
        "results": results,
    }


def main():
    args = parse_args()
    spec = trace_spec(args)
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "trace": spec}
    if not args.dry_run:
        report.update(run(args, spec))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
