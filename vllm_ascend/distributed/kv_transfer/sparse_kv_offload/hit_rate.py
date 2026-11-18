# SPDX-License-Identifier: Apache-2.0
"""Opt-in device counters. Snapshots and request-completion summaries synchronize with the host."""

import torch


class HitRateStats:
    """Count selected stable-prefix occurrences, including cold misses.

    One instance per worker; each layer owns a fixed-address int64 [access, hit]
    tensor. Updates are captured in ACLGraph and execute on every replay.
    """

    def __init__(self, use_fused_overlap: bool, layer_names, device):
        self.use_fused_overlap = use_fused_overlap
        self.layers = {name: torch.zeros(2, dtype=torch.int64, device=device) for name in layer_names}

    def record(self, layer: str, valid: torch.Tensor, hit: torch.Tensor) -> None:
        self.layers[layer].add_(torch.stack((valid.sum(), (valid & hit).sum())))

    def record_lim(self, layer, sources, misses, query_ends, states, prefix_lengths):
        rows = torch.arange(sources.shape[0], device=sources.device)
        requests = (rows[:, None] >= query_ends[None, :]).sum(dim=1)
        valid_rows = requests < query_ends.numel()
        requests = requests.clamp(max=query_ends.numel() - 1)
        offloaded = (states[requests] == -1) | (states[requests] == -2)
        sources = sources.reshape(sources.shape[0], -1)
        valid = valid_rows[:, None] & offloaded[:, None] & (sources >= 0) & (sources < prefix_lengths[requests, None])
        positions = torch.arange(sources.shape[1], device=sources.device)
        self.record(layer, valid, positions[None, :] >= misses[:, None])

    def record_overlap(self, layer, sources, plan, prefix_lengths, request_ids):
        sources = sources.reshape(sources.shape[0], -1)
        valid = (sources >= 0) & (sources < prefix_lengths[:, None]) & (request_ids[:, None] >= 0)
        # External plans encode hit as slot+1, miss as -(slot+1), padding as 0.
        # The planner publishes into registered host memory. Capture the H2D
        # copy after the planner stream dependency, never inspect it in Python.
        plan = plan.to(device=sources.device, non_blocking=True)
        self.record(layer, valid, plan > 0)

    def reset(self) -> None:
        # Call between workloads, with no concurrent generate/replay.
        for counts in self.layers.values():
            counts.zero_()

    def snapshot(self) -> dict:
        layers = {}
        # Batch all layers into one host transfer at measurement boundaries.
        values = torch.stack(list(self.layers.values())).cpu().tolist() if self.layers else []
        for layer, (accesses, hits) in zip(self.layers, values):
            layers[layer] = {
                "accesses": accesses,
                "hits": hits,
                "misses": accesses - hits,
                "hit_rate": hits / accesses if accesses else None,
            }
        accesses = sum(row["accesses"] for row in layers.values())
        hits = sum(row["hits"] for row in layers.values())
        return {
            "use_fused_overlap": self.use_fused_overlap,
            "accesses": accesses,
            "hits": hits,
            "misses": accesses - hits,
            "hit_rate": hits / accesses if accesses else None,
            "layers": layers,
        }

    def log_summary(self, logger, request_id: str) -> None:
        """Report all-layer cumulative totals; concurrent requests share counters."""
        result = self.snapshot()
        if not result["accesses"]:
            return
        logger.info(
            "[sparse_kv_hit_rate] request_finished=%s use_fused_overlap=%s "
            "scope=cumulative_all_layers hits=%d misses=%d accesses=%d hit_rate=%.6f",
            request_id,
            self.use_fused_overlap,
            result["hits"],
            result["misses"],
            result["accesses"],
            result["hit_rate"],
        )

        for layer, counts in result["layers"].items():
            rate = f"{counts['hit_rate']:.6f}" if counts["hit_rate"] is not None else "N/A"
            logger.info(
                "[sparse_kv_hit_rate] request_finished=%s use_fused_overlap=%s "
                "scope=cumulative_layer layer=%s hits=%d misses=%d accesses=%d hit_rate=%s",
                request_id,
                self.use_fused_overlap,
                layer,
                counts["hits"],
                counts["misses"],
                counts["accesses"],
                rate,
            )


def reset_hit_rate_stats(worker):
    """Worker callable for LLM.collective_rpc; invoke after warmup."""
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        get_sparse_kv_offload_manager,
    )

    stats = get_sparse_kv_offload_manager().hit_rate_stats
    if stats is None:
        raise RuntimeError("Enable sparse_kv_offload_config.collect_hit_rate first")
    stats.reset()
    torch.npu.synchronize()


def get_hit_rate_stats(worker):
    """Worker callable returning JSON-serializable per-layer and total counts."""
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        get_sparse_kv_offload_manager,
    )

    stats = get_sparse_kv_offload_manager().hit_rate_stats
    if stats is None:
        raise RuntimeError("Enable sparse_kv_offload_config.collect_hit_rate first")
    torch.npu.synchronize()
    return stats.snapshot()
