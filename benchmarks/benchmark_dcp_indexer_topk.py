# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the single-node A3 TP16/DCP16 K2048 indexer candidate path."""

import argparse
import json
import math
import os
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

WORLD_SIZE = 16
TOPK = 2048
HEAD_DIM = 128
PAGE_SIZE = 128
DEFAULT_INTERLEAVE = 768
MAX_LOCAL_TOKENS = 1 << 20  # Pack's local ID limit.
MAX_GLOBAL_TOKENS = 1 << 24  # Merge's FP32 ID comparator limit.


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, nargs="+", default=[1, 16, 64], help="Total query rows per step")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[32768], help="Global KV tokens per request")
    parser.add_argument("--queries-per-request", type=int, default=1, help="Causal query suffix size per request")
    parser.add_argument("--heads", type=int, choices=[32, 64], default=64, help="Replicated indexer heads per rank")
    parser.add_argument("--interleave", type=int, default=DEFAULT_INTERLEAVE, help="Global tokens per DCP stripe")
    parser.add_argument("--input", choices=["qli", "synthetic"], default="qli", help="Run QLI or preload candidates")
    parser.add_argument("--id-mode", choices=["owner-local", "global", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile-dir", type=Path, help="Write a separate torch_npu trace per case and rank")
    parser.add_argument("--profile-steps", type=int, default=3)
    args = parser.parse_args()
    if min(args.queries_per_request, args.interleave, args.warmup, args.iterations, args.profile_steps) < 1:
        parser.error("queries-per-request, interleave, warmup, iterations and profile-steps must be positive")
    if any(rows < 1 or rows % args.queries_per_request for rows in args.queries):
        parser.error("each query count must be a positive multiple of queries-per-request")
    if WORLD_SIZE * args.interleave >= 1 << 31:
        parser.error("DCP stripe period must fit in INT32")
    for length in args.context_lengths:
        if not args.queries_per_request <= length <= MAX_GLOBAL_TOKENS:
            parser.error("context length must be between queries-per-request and 2**24")
        if local_length(length, 0, args.interleave) > MAX_LOCAL_TOKENS:
            parser.error("context length and interleave exceed the packer's 2**20 local token limit")
    return args


def local_length(length, rank, interleave):
    cycles, rest = divmod(length, WORLD_SIZE * interleave)
    return cycles * interleave + min(interleave, max(0, rest - rank * interleave))


def make_selector(args, rows, length, rank, device, device_operator):
    import torch

    from vllm_ascend.worker.utils import make_page_strided_cache_view

    q_count = args.queries_per_request
    requests = rows // q_count
    local_tokens = local_length(length, rank, args.interleave)
    pages = max(1, math.ceil(local_tokens / PAGE_SIZE))
    total_pages = requests * pages
    query_rng = torch.Generator().manual_seed(args.seed)
    key_rng = torch.Generator().manual_seed(args.seed + 1 + rank)

    q = torch.randint(-127, 128, (rows, args.heads, HEAD_DIM), generator=query_rng, dtype=torch.int8).to(device)
    weights = torch.randn((rows, args.heads), generator=query_rng).to(device=device, dtype=torch.bfloat16)
    q_scale = (torch.rand(rows * args.heads, generator=query_rng) * 0.01 + 0.005).to(device=device, dtype=torch.float16)
    key_shape = (total_pages, PAGE_SIZE, 1, HEAD_DIM)
    scale_shape = (total_pages, PAGE_SIZE, 1, 1)
    key_bytes = PAGE_SIZE * HEAD_DIM
    page_bytes = key_bytes + PAGE_SIZE * torch.empty((), dtype=torch.float16).element_size()
    # Model Runner V2's hybrid cache places K and scale regions in each page.
    backing = torch.empty(total_pages * page_bytes, dtype=torch.int8, device=device)
    key = make_page_strided_cache_view(backing, key_shape, torch.int8, page_bytes)
    key_scale = make_page_strided_cache_view(backing, scale_shape, torch.float16, page_bytes, offset_bytes=key_bytes)
    key.copy_(torch.randint(-127, 128, key_shape, generator=key_rng, dtype=torch.int8).to(device))
    key_scale.copy_((torch.rand(scale_shape, generator=key_rng) * 0.01 + 0.005).to(device=device, dtype=torch.float16))
    table = torch.randperm(total_pages, generator=key_rng, dtype=torch.int32).reshape(requests, pages).to(device)
    starts = torch.arange(0, rows + 1, q_count, dtype=torch.int32).to(device)
    local = torch.full((requests,), local_tokens, dtype=torch.int32, device=device)
    dcp = {
        "global_seq_lens": torch.full((requests,), length, dtype=torch.int32, device=device),
        "cp_world": WORLD_SIZE,
        "cp_rank": rank,
        "cp_interleave": args.interleave,
    }
    metadata = device_operator.build_indexer_local_metadata(
        starts, local, num_heads_q=args.heads, head_dim=HEAD_DIM, topk=TOPK, **dcp
    )

    def select():
        return device_operator.indexer_select_dcp_candidates(
            q, weights, key, key_scale, table, starts, local, q_scale, qli_metadata=metadata, topk=TOPK, **dcp
        )

    return select


def synthetic_candidates(args, rows, length, rank, device):
    import torch

    rng = torch.Generator().manual_seed(args.seed + rank)
    # Use unsorted scores with ties and the same invalid sentinel as normalized QLI.
    ids = torch.randperm(TOPK, dtype=torch.int32, generator=rng).expand(rows, 1, TOPK).clone()
    visible = [
        local_length(length - args.queries_per_request + row % args.queries_per_request + 1, rank, args.interleave)
        for row in range(rows)
    ]
    ids.masked_fill_(ids >= torch.tensor(visible).view(rows, 1, 1), -1)
    scores = torch.randint(0, TOPK, ids.shape, generator=rng).float()
    scores.masked_fill_(ids < 0, -torch.inf)
    return ids.to(device), scores.to(device)


def reference_topk(indices, scores, args, rows, length, group):
    """Build an independent CPU reference from candidate rows on every rank."""
    import torch
    import torch.distributed as dist

    from vllm_ascend.attention.context_parallel.sharded_indexer import gathered_candidate_row_tile

    tile = gathered_candidate_row_tile(rows, WORLD_SIZE, TOPK)
    sample = sorted({0, rows // 2, rows - 1, min(tile - 1, rows - 1), min(tile, rows - 1)})
    local = (indices.cpu()[sample, 0], scores.cpu()[sample, 0])
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local, group=group.cpu_group)
    expected = {"owner-local": [], "global": []}
    for sample_slot, row in enumerate(sample):
        visible = length - args.queries_per_request + row % args.queries_per_request + 1
        pairs = []
        for rank, (rank_ids, rank_scores) in enumerate(gathered):
            ids, values = rank_ids[sample_slot], rank_scores[sample_slot]
            valid = ids >= 0
            local_ids = ids[valid].tolist()
            local_visible = local_length(visible, rank, args.interleave)
            if (
                len(local_ids) != min(TOPK, local_visible)
                or len(local_ids) != len(set(local_ids))
                or any(idx >= local_visible for idx in local_ids)
                or not torch.isfinite(values[valid]).all()
                or not ((ids[~valid] == -1) & torch.isneginf(values[~valid])).all()
            ):
                raise RuntimeError(f"Invalid local candidates for query row {row}, DCP rank {rank}")
            for idx, score in zip(local_ids, values[valid].tolist()):
                block, offset = divmod(idx, args.interleave)
                global_id = (block * WORLD_SIZE + rank) * args.interleave + offset
                pairs.append((-score, global_id, rank, idx))
        winners = sorted(pairs)[:TOPK]
        global_ids = [gid for _, gid, _, _ in winners]
        owner_ids = [idx for _, _, rank, idx in winners if rank == group.rank_in_group]
        for mode, ids in (("global", global_ids), ("owner-local", owner_ids)):
            expected[mode].append(ids + [-1] * (TOPK - len(ids)))
    return sample, {mode: torch.tensor(ids, dtype=torch.int32) for mode, ids in expected.items()}


@contextmanager
def profile_stages(group):
    """Annotate the production call chain for the trace run only."""
    import torch

    from vllm_ascend.ops.triton import sharded_indexer as ops

    def annotate(fn, name):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with torch.profiler.record_function("dcp_indexer." + name):
                return fn(*args, **kwargs)

        return wrapped

    with ExitStack() as stack:
        for attr, label in (
            ("normalize_dcp_candidates", "normalize"),
            ("pack_local_candidates", "sort_pack"),
            ("merge_packed_candidates", "merge"),
        ):
            stack.enter_context(patch.object(ops, attr, annotate(getattr(ops, attr), label)))
        stack.enter_context(patch.object(group, "all_gather", annotate(group.all_gather, "all_gather")))
        yield


def time_case(invoke, group, args):
    import torch
    import torch.distributed as dist

    with torch.inference_mode():
        for _ in range(args.warmup):
            invoke()
        torch.npu.synchronize()

    samples = []
    for _ in range(args.iterations):
        dist.barrier(group=group.cpu_group)
        with torch.inference_mode():
            start = perf_counter_ns()
            invoke()
            torch.npu.synchronize()
            samples.append((perf_counter_ns() - start) / 1e6)

    # Summarize the slowest rank for each step. Keep barriers and reduction outside timing.
    slowest = torch.tensor(samples, dtype=torch.float64)
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX, group=group.cpu_group)
    return {
        "rank_max_p50_ms": torch.quantile(slowest, 0.50).item(),
        "rank_max_p95_ms": torch.quantile(slowest, 0.95).item(),
        "rank_max_samples_ms": slowest.tolist(),
    }


def trace_case(select, candidates, exchange, group, args, directory):
    import torch
    import torch.distributed as dist
    import torch_npu

    profiler = torch_npu.profiler
    dist.barrier(group=group.cpu_group)
    with (
        torch.inference_mode(),
        profile_stages(group),
        profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(directory), worker_name=f"rank{group.rank_in_group}"),
        ) as trace,
    ):
        for _ in range(args.profile_steps):
            with torch.profiler.record_function("dcp_indexer.step"):
                if select is not None:
                    with torch.profiler.record_function("dcp_indexer.local_qli"):
                        candidates = select()
                exchange(candidates)
            trace.step()
        torch.npu.synchronize()
    dist.barrier(group=group.cpu_group)


def run_cases(args, group, device, device_operator):
    import torch
    import torch.distributed as dist

    from vllm_ascend.attention.context_parallel.sharded_indexer import (
        PACKED_PAIR_BYTES,
        distributed_topk,
        gathered_candidate_row_tile,
    )

    modes = ("owner-local", "global") if args.id_mode == "both" else (args.id_mode,)
    rank = group.rank_in_group
    for rows in args.queries:
        for length in args.context_lengths:
            select = None
            if args.input == "qli":
                select = make_selector(args, rows, length, rank, device, device_operator)
                candidates = select()
            else:
                candidates = synthetic_candidates(args, rows, length, rank, device)
            sample, expected = reference_topk(*candidates, args, rows, length, group)

            for mode in modes:

                def exchange(pair, mode=mode):
                    return distributed_topk(*pair, group, TOPK, args.interleave, return_local=(mode == "owner-local"))

                def invoke(select=select, candidates=candidates, exchange=exchange):
                    pair = select() if select is not None else candidates
                    return exchange(pair)

                actual = exchange(candidates).cpu()[sample, 0]
                verified = torch.tensor(int(torch.equal(actual, expected[mode])), dtype=torch.int32)
                dist.all_reduce(verified, op=dist.ReduceOp.MIN, group=group.cpu_group)
                if not verified.item():
                    raise RuntimeError(f"Top-K reference mismatch for Q={rows}, KV={length}, mode={mode}, rank={rank}")

                latency = time_case(invoke, group, args)
                tile = gathered_candidate_row_tile(rows, WORLD_SIZE, TOPK)
                case = f"{args.input}-{mode}-q{rows}-kv{length}"
                directory = args.profile_dir / case if args.profile_dir is not None else None
                if directory is not None:
                    trace_case(select, candidates, exchange, group, args, directory)
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "case": case,
                                "input": args.input,
                                "id_mode": mode,
                                "queries": rows,
                                "queries_per_request": args.queries_per_request,
                                "global_context": length,
                                "local_context_by_rank": [
                                    local_length(length, r, args.interleave) for r in range(WORLD_SIZE)
                                ],
                                "replicated_heads": args.heads,
                                "kernel_page_size": PAGE_SIZE,
                                "interleave": args.interleave,
                                "topk": TOPK,
                                "world_size": WORLD_SIZE,
                                "seed": args.seed,
                                "warmup_steps": args.warmup,
                                "all_gathers_per_step": math.ceil(rows / tile),
                                "packed_input_bytes_per_rank": rows * TOPK * PACKED_PAIR_BYTES,
                                "checked_rows": sample,
                                "trace_dir": str(directory) if directory is not None else None,
                                **latency,
                            }
                        ),
                        flush=True,
                    )
                del invoke
            del select, candidates


def main():
    args = parse_args()
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != WORLD_SIZE
    ):
        raise SystemExit("Launch with torchrun --standalone --nproc-per-node=16 on a single A3 server.")

    # Keep NPU imports below CLI parsing so --help works without the Ascend runtime.
    import torch
    import torch.distributed as dist
    import torch_npu
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.device.device_config import check_ascend_device_type
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
    from vllm_ascend.utils import enable_custom_op

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", local_rank)
    torch_npu.npu.set_device(device)
    torch.set_num_threads(1)
    if current_platform.device_type != "npu":
        raise RuntimeError("Install and enable the vLLM Ascend platform plugin.")
    if not dist.is_backend_available("hccl"):
        raise RuntimeError("The torch_npu HCCL backend must be available.")
    check_ascend_device_type()
    if not get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
        raise RuntimeError("This benchmark requires the A3 native INT8 QLI hardware profile.")

    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=WORLD_SIZE,
            decode_context_parallel_size=WORLD_SIZE,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=args.interleave,
            distributed_executor_backend="external_launcher",
        )
    )
    with set_current_vllm_config(config), torch.no_grad():
        init_ascend_config(config)
        device_operator = None
        if args.input == "qli":
            if not enable_custom_op() or not all(
                hasattr(torch.ops._C_ascend, name)
                for name in ("npu_quant_lightning_indexer_v2", "npu_quant_lightning_indexer_v2_metadata")
            ):
                raise RuntimeError("Build and install the vLLM Ascend A3 QLI and QLI metadata custom operators.")
            # Initialize ops first: its eager imports reach DeviceOperator,
            # which itself imports Triton modules from the ops package.
            import vllm_ascend.ops  # noqa: F401
            from vllm_ascend.device.device_op import DeviceOperator

            device_operator = DeviceOperator

        try:
            init_distributed_environment(
                world_size=WORLD_SIZE, rank=int(os.environ["RANK"]), local_rank=local_rank, backend="hccl"
            )
            initialize_model_parallel(
                tensor_model_parallel_size=WORLD_SIZE,
                decode_context_model_parallel_size=WORLD_SIZE,
                prefill_context_model_parallel_size=1,
                backend="hccl",
            )
            run_cases(args, get_dcp_group(), device, device_operator)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
