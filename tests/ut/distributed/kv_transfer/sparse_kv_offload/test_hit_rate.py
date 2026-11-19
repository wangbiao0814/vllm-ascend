# SPDX-License-Identifier: Apache-2.0
"""CPU tests runnable without importing the NPU/vLLM package."""

import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "hit_rate", Path(__file__).parents[5] / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/hit_rate.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HitRateStats = MODULE.HitRateStats


def test_lim_dynamic_rows_cold_dense_and_padding():
    stats = HitRateStats(False, ["layer"], "cpu")
    stats.record_lim(
        "layer",
        torch.tensor([[1, 2, 3], [4, 5, -1], [1, 2, 3], [6, 7, 8], [1, 2, 3]]),
        torch.tensor([1, 2, 0, 3, 0]),
        torch.tensor([2, 3, 4]),
        torch.tensor([-1, -3, -2]),
        torch.tensor([10, 10, 10]),
    )
    row = stats.snapshot()["layers"]["layer"]
    assert row == {"accesses": 8, "hits": 2, "misses": 6, "hit_rate": 0.25}
    pointer = stats.layers["layer"].data_ptr()
    stats.reset()
    assert stats.layers["layer"].data_ptr() == pointer
    assert stats.snapshot()["hit_rate"] is None


def test_overlap_excludes_tail_invalid_and_inactive_requests():
    stats = HitRateStats(True, ["layer"], "cpu")
    sources = torch.tensor([[1, 2, 10, -1], [1, 2, 3, 4]])
    plan = torch.tensor([[1, -2, 3, 0], [1, 2, 3, 4]], dtype=torch.int16)
    for _ in range(3):
        stats.record_overlap("layer", sources, plan, torch.tensor([10, 10]), torch.tensor([5, -1]))
    result = stats.snapshot()
    assert result["accesses"] == 6
    assert result["hits"] == 3
    assert result["hit_rate"] == 0.5


def test_aclgraph_replay_reads_updated_host_plan():
    import pytest

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU")
    stats = HitRateStats(True, ["layer"], "npu")
    sources = torch.tensor([[1, 2, 10, -1]], device="npu")
    prefix = torch.tensor([10], device="npu")
    request_ids = torch.tensor([5], device="npu")
    plan = torch.tensor([[1, -2, 3, 0]], dtype=torch.int16, pin_memory=True)
    for _ in range(3):
        stats.record_overlap("layer", sources, plan, prefix, request_ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        stats.record_overlap("layer", sources, plan, prefix, request_ids)
    stats.reset()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    plan[0, 1] = 2
    graph.replay()
    torch.npu.synchronize()
    assert stats.snapshot()["accesses"] == 4
    assert stats.snapshot()["hits"] == 3


def test_aclgraph_lim_replay_uses_dynamic_state_and_counts():
    import pytest

    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU")
    stats = HitRateStats(False, ["layer"], "npu")
    sources = torch.tensor([[1, 2, 3], [4, 5, 6]], device="npu")
    misses = torch.tensor([1, 2], device="npu")
    ends = torch.tensor([1, 2], device="npu")
    states = torch.tensor([-1, -3], device="npu")
    prefix = torch.tensor([10, 10], device="npu")
    for _ in range(3):
        stats.record_lim("layer", sources, misses, ends, states, prefix)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        stats.record_lim("layer", sources, misses, ends, states, prefix)
    stats.reset()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    states.fill_(-1)
    misses.zero_()
    graph.replay()
    torch.npu.synchronize()
    assert stats.snapshot()["accesses"] == 9
    assert stats.snapshot()["hits"] == 8


def test_completion_summary_aggregates_layers_without_reset():
    from unittest.mock import Mock

    stats = HitRateStats(False, ["layer0", "layer1", "unused"], "cpu")
    stats.layers["layer0"].copy_(torch.tensor([10, 6]))
    stats.layers["layer1"].copy_(torch.tensor([30, 9]))
    logger = Mock()
    stats.log_summary(logger, "request-1")
    assert logger.info.call_count == 4
    args = logger.info.call_args_list[0].args
    assert args[1:] == ("request-1", False, 15, 25, 40, 0.375)
    assert logger.info.call_args_list[1].args[1:] == ("request-1", False, "layer0", 6, 4, 10, "0.600000")
    assert logger.info.call_args_list[2].args[1:] == ("request-1", False, "layer1", 9, 21, 30, "0.300000")
    assert logger.info.call_args_list[3].args[1:] == ("request-1", False, "unused", 0, 0, 0, "N/A")
    assert stats.snapshot()["misses"] == 25
    stats.log_summary(logger, "request-2")
    assert logger.info.call_args_list[4].args[1:] == ("request-2", False, 15, 25, 40, 0.375)
    stats.reset()
    logger.reset_mock()
    stats.log_summary(logger, "empty")
    logger.info.assert_not_called()


def test_completion_reporting_requires_scheduler_finished_ids():
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock

    root = Path(__file__).parents[5] / "vllm_ascend"
    manager_tree = ast.parse(
        (root / "distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py").read_text()
    )
    method = next(
        node
        for node in ast.walk(manager_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "report_hit_rate_on_requests_finished"
    )
    npu = Mock()
    logger = Mock()
    namespace = {"torch_npu": SimpleNamespace(npu=npu), "logger": logger}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "report", "exec"), namespace)
    report = namespace[method.name]
    stats = Mock()
    manager = SimpleNamespace(hit_rate_stats=stats, tp_rank=0)
    report(manager, set())  # deferred/unscheduled, no terminal scheduler notification
    stats.log_summary.assert_not_called()
    npu.synchronize.assert_not_called()
    report(manager, {"done-b", "done-a"})
    stats.log_summary.assert_called_once_with(logger, "done-a,done-b")
    stats.reset_mock()
    manager.tp_rank = 1
    report(manager, {"done"})
    stats.log_summary.assert_not_called()
    runner_tree = ast.parse((root / "worker/v2/model_runner.py").read_text())
    execute = next(
        node for node in ast.walk(runner_tree) if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    assert "scheduler_output.finished_req_ids" in ast.unparse(execute)
    assert "not dummy_run" in ast.unparse(execute)
    # Removal can also mean deferral; it must never report completion itself.
    state_tree = ast.parse((root / "worker/v2/model_states/default.py").read_text())
    remove = next(
        node for node in ast.walk(state_tree) if isinstance(node, ast.FunctionDef) and node.name == "remove_request"
    )
    assert "report_hit_rate" not in ast.unparse(remove)


def test_shared_indexer_layers_are_counted_at_attention_consumption():
    import ast

    source = Path(__file__).parents[5] / "vllm_ascend/attention/sfa_kv_offload.py"
    tree = ast.parse(source.read_text())
    methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "record_lim" not in ast.unparse(methods["_lim_select"])
    assert "record_lim" in ast.unparse(methods["_copy_sfa_attention"])
    assert "hit_rate_stats is not None and (not self.skip_topk)" not in ast.unparse(
        methods["_execute_fused_overlap_offload_decode"]
    )
