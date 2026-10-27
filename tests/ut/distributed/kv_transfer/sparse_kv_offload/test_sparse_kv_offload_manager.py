import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_host_device_memory_usage_ratio,
    resolve_lru_workspace_threads,
)


class TestLRUWorkspaceThreads(unittest.TestCase):
    def test_thread_budget_is_limited_by_available_cpus(self):
        self.assertEqual(resolve_lru_workspace_threads(16, 8), 8)
        self.assertEqual(resolve_lru_workspace_threads(8, 16), 8)

    def test_thread_budget_requires_positive_values(self):
        for configured_threads, available_cpus in ((0, 8), (-1, 8), (8, 0), (8, -1)):
            with self.assertRaises(ValueError):
                resolve_lru_workspace_threads(configured_threads, available_cpus)


class TestHostDeviceMemoryRatio(unittest.TestCase):
    def gen_kv_cache_specs(
        self,
        spec_type: type[KVCacheSpec],
        num: int,
        name_suffix: str,
        **kwargs,
    ):
        kv_cache_specs = {}
        for layer_id in range(num):
            name = f"model.layers.{layer_id}.{name_suffix}"
            spec = spec_type(**kwargs)
            kv_cache_specs[name] = spec
        return kv_cache_specs

    def test_normal_case(self):
        # same number of mla & indexer cache, no c8
        # gold = kv_dim / idx_dim
        #      = 576 / 128
        #      = 4.5
        hd_memory_ratio_gold = 4.5
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            4,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_indexer_share(self):
        # multi layer share one indexer (GLM5.2), no c8
        # gold = (kv_num * kv_dim) / (idx_num * idx_dim)
        #      = (4 * 576) / (1 * 128)
        #      = 18
        hd_memory_ratio_gold = 18
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_li_c8(self):
        # indxer c8 cache
        # gold = (kv_num * kv_dim * bf16) / (idx_num * (idx_dim * int8 + scale_dim * fp16))
        #      = (4 * 576 * 2) / (1 * (128 * 1 + 1 * 2))
        #      = 35.44615384615385
        hd_memory_ratio_gold = 35.44615384615385
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            cache_dtype_str="bfloat16",
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_no_offload(self):
        # no store_on_host specs
        # gold = 0 (don't need host memory)
        hd_memory_ratio_gold = 0
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=False,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)


class TestResidentPlannerDispatch(unittest.TestCase):
    def test_skip_topk_reuses_source_layout_without_second_age_update(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager._get_offload_layer_id = MagicMock(return_value=1)
        manager._onload_topk_kv_cpu = MagicMock()
        manager.max_num_topk_rows = 2
        manager.mtp_layer_id = -1
        manager.tp_size = 1
        manager.gvas_k_bases, manager.gvas_v_bases = [100, 200], [1000, 1100]
        manager.addr_k_bases, manager.addr_v_bases = [2000, 2100], [3000, 3100]
        manager.gvas_buffer_npu = MagicMock()
        manager.addr_buffer_npu = MagicMock()
        manager.size_buffer_npu = MagicMock()
        manager.num_tokens_buffer_npu = MagicMock()
        manager.topk_buffers_k = [MagicMock()]
        manager.lru_current_slots_cpu = MagicMock()
        current_slots = MagicMock()
        module = SparseKVOffloadManager.__module__
        with patch(f"{module}.offload", create=True) as offload:
            manager.onload_topk_kv(
                "layer.1", 2, 2, MagicMock(), MagicMock(), current_slots, MagicMock(), MagicMock(), skip_topk=True,
            )
        manager._onload_topk_kv_cpu.assert_not_called()
        offload.sparse_copy.assert_called_once()
        current_slots.__getitem__.return_value.copy_.assert_called_once_with(
            manager.lru_current_slots_cpu.__getitem__.return_value, non_blocking=False,
        )

    def test_callback_selects_planner_and_preserves_address_inputs(self):
        for policy in ("lru", "reuse_prioritized_lru"):
            with self.subTest(policy=policy):
                manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
                manager.eviction_policy = policy
                manager.probation_age = 4
                manager.topk = 2
                manager.topk_buffer_size = 4
                manager.max_model_len = 128
                manager.lru_workspace_threads = 1
                manager.lru_stamps_ptrs = [200, 201]
                manager.lru_age_workspace_ptr = 300
                manager.sparse_kv_offload_cpp = MagicMock()
                # Real callback tuple layout: counts, miss tensors, pointers,
                # block table, address metadata, scratch, outputs, layer id.
                args = (2, "counts", "tokens", "slots", *range(10, 20), "blocks", *range(21, 37), 1)
                manager._onload_topk_kv_cpu(args)
                common = (*range(10, 20), *range(28, 33), 2, 2, 4, 128, 1, 1)
                if policy == "lru":
                    manager.sparse_kv_offload_cpp.lru_resident_compact.assert_called_once_with(*common)
                    manager.sparse_kv_offload_cpp.reuse_prioritized_lru_resident_compact.assert_not_called()
                else:
                    manager.sparse_kv_offload_cpp.reuse_prioritized_lru_resident_compact.assert_called_once_with(
                        *common, 201, 300, 4,
                    )
                    manager.sparse_kv_offload_cpp.lru_resident_compact.assert_not_called()
                manager.sparse_kv_offload_cpp.compute_lru_resident_addrs.assert_called_once_with(
                    "counts", "tokens", "slots", "blocks", *range(21, 28), 4, 1, *range(33, 37),
                )


if __name__ == "__main__":
    unittest.main()
