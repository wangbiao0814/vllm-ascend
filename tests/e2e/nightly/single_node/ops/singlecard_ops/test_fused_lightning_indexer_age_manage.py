# SPDX-License-Identifier: Apache-2.0
"""Real NPU tests: selection parity, age lifecycle, victim ranking and graph replay."""

import pytest
import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_fused_lightning_indexer_manage import (
    BLOCK_SIZE,
    MAX_CACHE_TOKENS,
    TOPK,
    _assert_resident_bijection,
    _build_case,
    _make_outputs,
    _native_topk,
)
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Age LIM requires A2/A3"
)
AGE_MAX = (1 << 19) - 1


def call_age(case, cache, ages, outputs, probation, diagnostic_flags=0):
    assert hasattr(torch.ops._C_ascend, "npu_fused_lightning_indexer_age_manage"), "Rebuild the Ascend extension"
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
        probation,
        diagnostic_flags,
    )


def check_age_step(case, cache, ages, outputs, before_cache, before_ages, probation):
    torch.npu.synchronize()
    result = [tensor.cpu() for tensor in outputs]
    actual_cache, actual_age = cache.cpu(), ages.cpu()
    reference = _native_topk(case).cpu()
    for batch, row in enumerate(case.req_entries):
        sources = result[0][batch, 0].long()
        assert torch.equal(sources.sort().values, reference[batch].long().sort().values)
        initial = before_cache[row]
        _assert_resident_bijection(actual_cache[row], case.offload_key[batch], case.cache_tokens[batch])
        if case.states[batch] == -2:
            expected = torch.zeros_like(before_ages[row])
            expected[actual_cache[row] >= 0] = probation
            assert result[5][batch] == case.cache_tokens[batch]
        else:
            expected = before_ages[row].clone()
            old_resident = initial >= 0
            expected[: case.offload_key[batch]] = torch.where(
                old_resident[: case.offload_key[batch]],
                (expected[: case.offload_key[batch]].clamp(0, AGE_MAX - 1) + 1),
                0,
            )
            hits = sources[initial[sources] >= 0]
            misses = sources[initial[sources] < 0]
            expected[hits] = 0
            expected[misses] = probation
            assert result[2][batch] == misses.numel()
            assert torch.equal(sources[: misses.numel()], misses.sort().values)
        assert torch.equal(actual_age[row], expected)
        assert torch.equal(actual_cache[row, sources], result[1][batch, 0])
    unused = sorted(set(range(cache.shape[0])) - set(case.req_entries))
    assert torch.equal(actual_age[unused], before_ages[unused])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("probation", [0, 4, AGE_MAX])
@torch.inference_mode()
def test_age_first_fill_and_steady(dtype, probation):
    case = _build_case(q_values=[1, 1], states=[-2, -1], offload_len=8192, cache_tokens=4096, dtype=dtype)
    cache = case.cache_seed.clone()
    ages = torch.arange(cache.shape[1], dtype=torch.int32, device="npu").expand_as(cache).contiguous()
    before_cache, before_ages = cache.cpu(), ages.cpu()
    outputs = _make_outputs(case)
    call_age(case, cache, ages, outputs, probation)
    check_age_step(case, cache, ages, outputs, before_cache, before_ages, probation)


@torch.inference_mode()
def test_age_aclgraph_replay_updates_persistent_state():
    case = _build_case(q_values=[1], states=[-2], offload_len=8192, cache_tokens=4096)
    cache = case.cache_seed.clone()
    ages = torch.zeros_like(cache)
    outputs = _make_outputs(case)
    for _ in range(3):
        call_age(case, cache, ages, outputs, 4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        call_age(case, cache, ages, outputs, 4)
    cache.copy_(case.cache_seed)
    ages.zero_()
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    for _ in range(2):
        case.states = [-1]
        case.request_state.fill_(-1)
        before_cache, before_ages = cache.cpu(), ages.cpu()
        graph.replay()
        check_age_step(case, cache, ages, outputs, before_cache, before_ages, 4)
        assert outputs[5].cpu().item() == 0


@torch.inference_mode()
def test_age_identity_transition_and_long_sources():
    case = _build_case(q_values=[1], states=[-1], offload_len=(1 << 17) + BLOCK_SIZE, cache_tokens=8192)
    cache = (
        torch.arange(case.cache_seed.shape[1], dtype=torch.int32, device="npu").expand_as(case.cache_seed).contiguous()
    )
    ages = torch.full_like(cache, AGE_MAX)
    outputs = _make_outputs(case)
    call_age(case, cache, ages, outputs, 4)
    torch.npu.synchronize()
    row = case.req_entries[0]
    _assert_resident_bijection(cache[row].cpu(), case.offload_key[0], case.cache_tokens[0])
    assert ages[row].max().cpu().item() <= 5
    assert torch.equal(outputs[0][0, 0].sort().values.cpu(), _native_topk(case)[0].sort().values.cpu())
    assert MAX_CACHE_TOKENS > TOPK  # same public slot capacity contract


def scan_seed(length, row):
    value = (length ^ ((row + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


@torch.inference_mode()
def test_age_selects_oldest_inside_original_scanned_blocks():
    length, budget, chunk = 16384, 8192, 512
    case = _build_case(q_values=[1], states=[-1], offload_len=length, cache_tokens=budget)
    selected = set(_native_topk(case)[0].cpu().tolist())
    missing = len([s for s in selected if s >= budget])
    assert missing > 0
    expected = []
    accumulated = []
    chunks = length // chunk
    start_chunk = scan_seed(length, case.req_entries[0]) % chunks
    for step in range(chunks):
        begin = ((start_chunk + step) % chunks) * chunk
        candidates = [s for s in range(begin, min(begin + chunk, budget)) if s not in selected]
        if missing <= chunk:
            accumulated.extend(candidates)
            if len(accumulated) >= missing:
                expected = sorted(accumulated, reverse=True)[:missing]
                break
        else:
            expected.extend(sorted(candidates, reverse=True)[: missing - len(expected)])
            if len(expected) == missing:
                break
    cache = case.cache_seed.clone()
    ages = torch.arange(cache.shape[1], dtype=torch.int32, device="npu").expand_as(cache).contiguous()
    outputs = _make_outputs(case)
    call_age(case, cache, ages, outputs, 4)
    torch.npu.synchronize()
    after = cache[case.req_entries[0], :budget].cpu()
    actual = torch.nonzero(after < 0).flatten().tolist()
    assert set(actual) == set(expected)


@torch.inference_mode()
def test_age_rejects_invalid_contract_and_preserves_invalid_dynamic_row():
    case = _build_case(q_values=[1], states=[-1], offload_len=8192, cache_tokens=4096)
    cache = case.cache_seed.clone()
    ages = torch.zeros_like(cache)
    outputs = _make_outputs(case)
    with pytest.raises(RuntimeError, match="probation_age"):
        call_age(case, cache, ages, outputs, AGE_MAX + 1)
    case.actual_seq_lengths_query.fill_(2)  # T=B=1, invalid device-side end offset
    before_cache, before_ages = cache.cpu(), ages.cpu()
    call_age(case, cache, ages, outputs, 4)
    torch.npu.synchronize()
    assert torch.equal(cache.cpu(), before_cache)
    assert torch.equal(ages.cpu(), before_ages)


@pytest.mark.parametrize("flags", range(128))
@torch.inference_mode()
def test_age_diagnostic_flags_control_state_updates(flags):
    # Each bit disables only its target stage; the cache plan must remain valid.
    case = _build_case(q_values=[1], states=[-1], offload_len=8192, cache_tokens=4096)
    cache = case.cache_seed.clone()
    initial = cache.cpu()
    ages = torch.full_like(cache, 7)
    outputs = _make_outputs(case)
    call_age(case, cache, ages, outputs, 4, diagnostic_flags=flags)
    torch.npu.synchronize()
    expected = torch.full_like(initial, 7)
    row = case.req_entries[0]
    prefix = case.offload_key[0]
    if not flags & 2:
        expected[row, :prefix] = torch.where(initial[row, :prefix] >= 0, 8, 0)
    sources = outputs[0][0, 0].cpu().long()
    hits = sources[initial[row, sources] >= 0]
    misses = sources[initial[row, sources] < 0]
    assert hits.numel() > 0 and misses.numel() > 0
    if not flags & (1 | 8 | 16 | 32):
        expected[row, hits] = 0
    if not flags & 4:
        expected[row, misses] = 4
    assert torch.equal(ages.cpu(), expected)
    assert int(outputs[2].cpu()[0]) == misses.numel()
    assert torch.equal(cache[row].cpu()[sources], outputs[1][0, 0].cpu())
    _assert_resident_bijection(cache[row].cpu(), prefix, case.cache_tokens[0])


@pytest.mark.parametrize("flags", [-1, 128])
@torch.inference_mode()
def test_age_rejects_invalid_diagnostic_flags(flags):
    case = _build_case(q_values=[1], states=[-1], offload_len=8192, cache_tokens=4096)
    with pytest.raises(RuntimeError, match="diagnostic_flags"):
        call_age(case, case.cache_seed.clone(), torch.zeros_like(case.cache_seed), _make_outputs(case), 4, flags)


@torch.inference_mode()
def test_age_buffered_hit_clear_all_hit_partial_block():
    case = _build_case(q_values=[1], states=[-1], offload_len=8320, cache_tokens=4096)
    selected = _native_topk(case)[0].cpu().long()
    selected_set = set(selected.tolist())
    fillers = [s for s in range(case.offload_key[0]) if s not in selected_set][: case.cache_tokens[0] - TOPK]
    residents = torch.cat((selected, torch.tensor(fillers, dtype=torch.long))).to("npu")
    cache = case.cache_seed.clone()
    row = case.req_entries[0]
    cache[row].fill_(-(1 << 31))
    cache[row, residents] = torch.arange(case.cache_tokens[0], dtype=torch.int32, device="npu")
    ages = torch.arange(cache.shape[1], dtype=torch.int32, device="npu").expand_as(cache).contiguous()
    before_cache, before_ages = cache.cpu(), ages.cpu()
    outputs = _make_outputs(case)
    call_age(case, cache, ages, outputs, 4)
    check_age_step(case, cache, ages, outputs, before_cache, before_ages, 4)
    assert int(outputs[2].cpu()[0]) == 0
    assert torch.equal(cache.cpu(), before_cache)


@pytest.mark.parametrize("length", [8320, (1 << 17) + BLOCK_SIZE])
@torch.inference_mode()
def test_age_vector_and_scalar_grouping_match(length):
    case = _build_case(q_values=[1], states=[-1], offload_len=length, cache_tokens=4096)
    results = []
    for flags in (0, 64):
        cache = case.cache_seed.clone()
        ages = torch.arange(cache.shape[1], dtype=torch.int32, device="npu").expand_as(cache).contiguous()
        outputs = _make_outputs(case)
        before_cache, before_ages = cache.cpu(), ages.cpu()
        call_age(case, cache, ages, outputs, 4, diagnostic_flags=flags)
        check_age_step(case, cache, ages, outputs, before_cache, before_ages, 4)
        results.append((cache.cpu(), ages.cpu(), [tensor.cpu() for tensor in outputs]))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    left, right = results[0][2], results[1][2]
    for index in (0, 1, 2, 5):
        assert torch.equal(left[index], right[index])
    count = int(left[5][0])
    for index in (3, 4):
        assert torch.equal(left[index][0, :count], right[index][0, :count])
