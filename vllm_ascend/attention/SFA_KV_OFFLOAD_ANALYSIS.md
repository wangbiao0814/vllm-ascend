# SFA KV Offload 调用逻辑与 LRU 策略分析

整理日期：2026-10-23。本文汇总本次对话对当前工作区源码的静态分析，示例按代码逻辑推演，未运行 NPU 实测。

## 1. 核心结论

CPU 保存完整的主 KV，NPU 保留 indexer cache 和有限容量的 resident KV buffer。每轮 decode 由 indexer 选出 top-k，仅将 resident 中缺失的 token 从 CPU 回载到 NPU，再对 resident buffer 执行一次 SFA。

主要特点：

- 通过继承复用普通 SFA 的模型计算流程，改写 KV 写入和 attention 数据来源。
- decode 新生成的主 KV 直接 D2H 写 CPU，不经过完整的 NPU paged 主 cache。
- top-k 的逻辑 token 索引被转换成 resident slot 索引，再传给 attention 算子。
- resident buffer 可以跨轮保留本轮未访问的历史 token。
- LRU 状态按“层 × query row”维护，使用请求身份变化重置和推测后缀失效机制。
- LRU 按整组 top-k 更新，更新顺序为“未访问槽 → 新加载槽 → 命中槽”，不是严格按单次访问先后更新的标准 LRU。

## 2. 源码导航

以下链接指向本次分析所在工作区；函数名可用于代码变动后重新定位。

| 文件 | 主要职责 |
| --- | --- |
| [sfa_kv_offload.py](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/attention/sfa_kv_offload.py) | offload metadata、KV 生成与写入、resident attention 路由 |
| [sfa_v1.py](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/attention/sfa_v1.py) | backend 选择、父类 forward、indexer、输出投影 |
| [sparse_kv_offload_manager.py](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py) | CPU 池、resident buffer、D2H/H2D、LRU 调用与图模式支持 |
| [sparse_kv_offload.cpp](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp) | CPU LRU 状态更新、miss 拷贝地址生成 |
| [device_op.py](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/device/device_op.py:542) | SFA 算子参数组织与调用 |
| [ascend_config.py](/Users/wangbiao/Documents/Codes/vllm-ascend/vllm_ascend/ascend_config.py:1042) | offload 配置及适用范围校验 |

## 3. 存储内容：这里的 K/V 是 MLA 缓存

manager 中的 `k/v` 是缓存对的命名，不应直接对应普通 MHA 中独立投影得到的 K 和 V。

| 存储对象 | 实际内容 | 用途 |
| --- | --- | --- |
| `k_caches_cpu` / `topk_buffers_k` | `k_nope`，维度 `kv_lora_rank` | 压缩 latent，作为 attention 的 key 和 value |
| `v_caches_cpu` / `topk_buffers_v` | `k_pe`，维度 `qk_rope_head_dim` | RoPE key |
| NPU indexer cache | indexer 的历史 key 等数据 | 选择 top-k |

实际 SFA 算子参数为：

```python
key = resident_k
value = resident_k
key_rope = resident_v
```

attention 输出之后，父类继续执行 `_v_up_proj()` 和 `o_proj()`。

完整主 KV 位于 CPU；用于选 top-k 的 indexer cache 仍驻留 NPU。因此，offload 并没有把所有类型的 KV 一并移到 CPU。

## 4. 启用入口与调用链

### 4.1 Backend 选择

当 `sparse_kv_offload_config.enabled=True` 时：

- `AscendSFABackend.get_builder_cls()` 返回 `AscendSFAKVOffloadMetadataBuilder`。
- `AscendSFABackend.get_impl_cls()` 返回 `AscendSFAKVOffloadImpl`。

实现类强制选择 `PreprocessType.NATIVE`，禁用 `sfa_prolog_v3/mlapo`，使主 KV 经过子类控制的生成和 offload 流程。

### 4.2 Metadata

`AscendSFAKVOffloadMetadataBuilder` 在父类 metadata 基础上补充：

| 字段 | 含义 |
| --- | --- |
| `num_decodes` | decode 请求数量 |
| `num_prefills` | prefill 请求数量 |
| `num_decode_tokens` | decode query token 总数，推测解码时可大于请求数 |
| `req_ids_tensor` | 请求身份，用于发现 resident row 被其他请求占用 |
| `token_to_req` | 每个 query token 属于哪个请求 |

纯 PD consumer 上，已收到 prompt KV 的单 token 边界步可能仍被调度器标记为 prefill。builder 使用 `treat_short_extends_as_decodes` 将符合条件的短扩展归入 decode；真正的长 prefill 不因此变成 decode。

### 4.3 主调用顺序

```text
AscendSFAKVOffloadImpl.forward()
  └─ 记录当前 layer_name
  └─ AscendSFAImpl.forward()
       ├─ Native Q/KV 投影与归一化
       ├─ indexer_select_pre_process()
       ├─ wait_for_kv_layer_from_connector()
       ├─ self.exec_kv()                         # offload 子类覆盖
       │    ├─ _compute_kv_only()
       │    └─ manager.offload_new_kv()          # 当前 token D2H
       ├─ 计算 ql_nope、q_pe
       ├─ indexer_select_post_process()         # 或复用已有 top-k
       ├─ self._execute_sparse_flash_attention_process()
       │    ├─ 准备逐 query 的 top-k、请求身份、稳定前缀
       │    ├─ manager.onload_topk_kv()
       │    │    ├─ _onload_topk_kv_cpu()
       │    │    │    ├─ lru_resident_compact()
       │    │    │    └─ compute_lru_resident_addrs()
       │    │    ├─ 上传拷贝描述符
       │    │    ├─ sparse_copy()               # 仅 miss H2D
       │    │    └─ 上传 current_slots
       │    └─ DeviceOperator 执行 resident SFA
       ├─ _v_up_proj()
       └─ o_proj()
```

父类负责模型计算主流程；子类通过覆盖 `exec_kv()` 和 `_execute_sparse_flash_attention_process()` 替换主 KV 的数据路径。`forward()` 用 `try/finally` 管理临时的当前层名称。

## 5. 一轮 decode 的数据流

### 5.1 当前 token 的主 KV 写入 CPU

`_is_decode_only()` 要求 attention 状态为 `DecodeOnly` 或 `SpecDecoding`，同时没有 prefill 且 decode 请求数大于零。

进入此分支后，`_compute_kv_only()`：

1. 将 `kv_no_split` 拆成 latent 部分和 RoPE 部分。
2. 对 latent 部分执行 RMSNorm，得到 `k_nope`。
3. 对 RoPE 部分执行旋转位置编码，得到 `k_pe`。

然后调用：

```python
manager.offload_new_kv(
    slot_mapping=slots,
    k_cache_cpu=k_cache_cpu,
    v_cache_cpu=v_cache_cpu,
    k_cache_npu=None,
    v_cache_npu=None,
    k=k_nope,
    v=k_pe,
    has_prefill=False,
    capturing=self._in_graph_runtime(),
)
```

`slots` 是 CPU paged cache 的物理 token 槽位。manager 根据源 tensor 行地址和目标 CPU slot 地址生成 D2H 描述符，再调用 `offload.sparse_copy()`。

TP 场景下，新生成的 decode KV 在各 rank 上重复，只有 TP0 写入 CPU 池。共享池地址会广播给其他 rank；H2D 前通过 NPU broadcast 保证 TP0 的 D2H 顺序。

### 5.2 准备 top-k

offload attention 路径会：

- 根据 `token_to_req` 展开请求的序列长度、请求 ID 和稳定前缀。
- 将不满足有效范围的 top-k 转成 `-1`。
- 把 `[tokens, 1, topk]` 压成 `[tokens, topk]`，其他不符合二维形状的输入报错。
- 按 query 行获取 resident KV 视图和固定页表。

`-1` 对应无效项，不参与正常回载。

### 5.3 从逻辑 token 转换到 resident slot

例如：

```text
indexer 输出逻辑 token： [5, 17, 80]
LRU 输出 resident slot： [3,  0,  6]
```

表示逻辑 token 5、17、80 的 KV 分别位于本行 resident slot 3、0、6。

hit 保留原槽位和数据，miss 才生成 H2D 拷贝。

### 5.4 miss 的地址计算

设 `C=topk_buffer_size`，对于 query 行 `row`、逻辑 token `token` 和分配槽 `slot`：

```python
logical_block = token // block_size
offset = token % block_size
physical_block = original_block_table[row, logical_block]

src = cpu_base + (physical_block * block_size + offset) * token_bytes
dst = resident_base + (row * C + slot) * token_bytes
```

K latent 和 RoPE key 分别使用自己的基地址和单 token 字节数。

这里存在两套不同寻址规则：CPU 源数据依赖请求原始 block table；NPU 目的数据依赖 resident row/slot。

### 5.5 Resident attention

算子使用 resident KV、resident slot 索引和 resident block table：

```python
DeviceOperator.execute_sparse_flash_attention_process(
    self,
    ql_nope[:num_decode_tokens],
    q_pe[:num_decode_tokens],
    (resident_k, resident_v),
    resident_slot_indices.unsqueeze(1),
    attn_metadata,
    resident_query_lens,
    resident_seq_lens,
    block_table=resident_block_table,
)
```

每个 query 行被组织成独立序列：

```text
resident_query_lens = [1, 2, 3, ...]   # 累计 query 长度
resident_seq_lens   = [C, C, C, ...]   # 每行 resident 容量
```

每行占用 `C / block_size` 个 resident page。buffer 只做 view，不会为了本轮 top-k 重新压紧全部 KV。

虽然 resident 中还留有本轮未选中的历史 token，算子仅通过本轮 `current_slots` 访问选中数据。

## 6. LRU 状态与管理粒度

### 6.1 关键数据结构

| 状态 | 主要形状 | 含义 |
| --- | --- | --- |
| `lru_slot_to_token_cpu_list[layer]` | `[rows, C]` | resident slot 保存哪个逻辑 token，`-1` 表示无效 |
| `lru_slots_cpu_list[layer]` | `[rows, C]` | 从优先淘汰到最后淘汰的 slot 队列 |
| `lru_last_req_ids_cpu_list[layer]` | `[rows]` | 上次占用该行的请求身份 |
| `lru_current_slots_cpu` | `[rows, K]` | 本轮 top-k 对应的 resident slot |
| `lru_miss_count_cpu_list[layer]` | `[rows]` | 本轮实际分配的 miss 数量 |
| `lru_miss_tokens_cpu_list[layer]` | `[rows, K]` | 待回载的逻辑 token |
| `lru_miss_slots_cpu_list[layer]` | `[rows, K]` | miss 的目标 resident slot |

其中 `K` 来自模型 `index_topk`。临时输出和工作区可复用，持久缓存身份与顺序按层保存。

### 6.2 按“层 × query row”复用

普通 decode 通常一请求一个 query，表现为每请求一行。推测解码一请求有多个 query 时，每个 query 使用独立 resident row。

由此可知：

- 同一请求多个 query 行之间没有这一级 resident KV 共享。
- 请求从一个 batch row 移到另一个 row 时，没有按请求 ID 迁移原 resident 内容。
- 某 row 的请求 ID 变化时，清空整行的映射并重置顺序。
- 在通常重新计算 top-k 的路径，各层独立更新自己的 resident 状态。

请求身份由调度器 request ID 的 Adler-32 值生成。这里依赖身份值检测行归属变化，没有建立按 request ID 查找和迁移缓存的全局池。

## 7. LRU 更新算法

核心函数是 C++ 的 `process_one_lru_resident_row()`，由 `lru_resident_compact()` 调用。

每行执行：

1. 将本轮 `current_slots`、miss token 和 miss slot 输出初始化为无效。
2. 若请求身份改变，重置整行。
3. 标记本轮有效 top-k token。
4. 从旧 LRU 队头扫描到队尾：先使不稳定后缀失效；本轮需要的有效槽加入 `hit_slots`；其他槽加入 `evictable_slots`。
5. 按 top-k 位置顺序收集尚无 slot 的有效 token，形成 miss。
6. 从 `evictable_slots` 头部依次取槽给 miss，更新 `slot_to_token` 和 `current_slots`。
7. 重建下一轮队列：

```text
优先淘汰                                             最后淘汰
[剩余 evictable slots] + [本轮新加载 miss slots] + [本轮 hit slots]
```

### 7.1 与严格逐访问 LRU 的差异

- 先保护本轮所有 hit，再为 miss 分配槽，不会因处理前面的 miss 而淘汰后面的 hit。
- 未访问槽维持旧队列的相对顺序。
- hit 槽维持旧队列中的相对顺序，不按本轮 top-k 顺序重排。
- hit 槽排在本轮新加载槽后面，获得更低的淘汰优先级。
- 不统计访问频次，也不按 attention 分数加权淘汰。
- 空槽与未访问有效槽统一进入 `evictable_slots`，按队列顺序分配；没有独立的空槽绝对优先策略。

因此，更准确的描述是“按本轮 top-k 集合批量更新、保护本轮命中的 LRU 变体”。

## 8. 逐轮示例

设 `C=4`、`K=2`，同一请求始终占用同一 query row；选中的 token 都位于稳定前缀内。`-` 表示无效槽，LRU 队头优先淘汰。

```text
初始 slot_to_token = [-, -, -, -]
初始 lru_slots     = [0, 1, 2, 3]
```

| 轮次 | 逻辑 top-k | hit | miss → slot | 更新后 slot_to_token | 更新后 LRU | 本轮 SFA slot 索引 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `[2,5]` | 无 | `2→0, 5→1` | `[2,5,-,-]` | `[2,3,0,1]` | `[0,1]` |
| 2 | `[5,7]` | `5→1` | `7→2` | `[2,5,7,-]` | `[3,0,2,1]` | `[1,2]` |
| 3 | `[2,8]` | `2→0` | `8→3` | `[2,5,7,8]` | `[2,1,3,0]` | `[0,3]` |
| 4 | `[9,10]` | 无 | `9→2, 10→1` | `[2,10,9,8]` | `[3,0,2,1]` | `[2,1]` |

第 3 轮展开：

```text
旧 LRU       = [3, 0, 2, 1]
hit slots    = [0]          # token 2
evictable    = [3, 2, 1]
miss token 8 → slot 3

新 LRU       = [2, 1] + [3] + [0]
               未访问    新加载  命中
```

第 4 轮按新队头覆盖 slot 2 和 slot 1，即淘汰 token 7 和 token 5。

这也说明：token 2 在第 2 轮未被选中仍可保留，第 3 轮再次入选时可以直接命中。`C>K` 的额外容量正是用于这种跨轮保留。

## 9. 推测解码与稳定前缀失效

目标文件根据请求累计 query 长度还原本轮 query 数量，然后计算：

```python
stable_prefix_len = max(seq_len - query_len, 0)
```

该值经 `token_to_req` 展开到每个 query 行。C++ 扫描 resident 时执行：

```python
if cached_token >= stable_prefix_len:
    slot_to_token[slot] = -1
```

例如 `seq_len=103`、`query_len=3`：

- `[0,99]` 是稳定历史，可复用 resident 副本。
- `[100,102]` 属于可能被推测解码重算或覆盖的后缀，旧副本失效。
- 这些 token 若再次被 top-k 选中，必须从 CPU 池加载最新内容。

逻辑位置相同不保证 KV 内容相同，这一规则防止命中旧数据。普通单 token decode 也沿用该规则。

## 10. 特殊路径

### 10.1 Prefill 与 mixed

仅 `keep_device_kv_cache=True` 的 PD 同机调试场景支持：

- `exec_kv()` 先调用父类写完整 NPU 主 cache，再把对应 slot D2H 提交到 CPU 池。
- 纯 prefill 直接调用父类 SFA。
- mixed 的 decode 部分走 resident attention，prefill 部分仍访问完整 NPU cache。
- prefill 的累计 query 长度会减去 decode 的累计偏移，最后拼接两部分输出。

默认关闭完整 NPU 主 cache 的 D 节点收到真正的 prefill/mixed batch 时会报错。

### 10.2 skip_topk

父类复用已有的 top-k 索引；manager 的 `skip_topk` 分支还会：

- 复用前一层的 miss 拷贝描述符和 resident slot 映射。
- 用本层与前层的基地址差调整 CPU 源地址和 NPU 目标地址。
- 按调整后的描述符回载本层对应 KV。
- 跳过本层独立 LRU 计算。

此路径依赖相邻层的映射和布局兼容，并校验 K/V 地址偏移一致；不能理解成每层都重新跑一次 LRU。

### 10.3 图模式

`_in_graph_runtime()` 同时考虑 capture 和运行时 graph mode。在图路径中，CPU LRU 经当前 stream 的 `_launch_host_func()` 执行，再上传拷贝描述符和 slot 映射。

CPU 决策与双向元数据传输仍然存在，图模式并未将 LRU 移到 NPU 上。

## 11. 配置与适用范围

| 项目 | 当前实现 |
| --- | --- |
| `enabled` | 默认关闭 |
| `topk_buffer_size` | 默认 4096，表示每个 resident query row 的 token 容量 |
| `topk` | 从模型 `index_topk` 获取 |
| 容量约束 | `C>0`、`C>=K`、`C % block_size == 0` |
| `dram_size_per_dp_GB` | 默认 128 |
| `keep_device_kv_cache` | 默认 false；true 用于同机调试，仍分配完整 NPU 主 cache |
| 默认运行角色 | PD 分离的 KV consumer / D 节点 |
| 主 SFA cache | BF16，当前不支持主 SFA C8 |
| Indexer cache | 可以使用 sparse LI C8 |
| Context parallel | 当前不支持 |
| Pipeline parallel | 当前不支持 |
| Model runner v2 | 当前不支持 |
| 压缩模型配置 | 当前拒绝带 `compress_ratios` 的配置 |

resident 最大行数为：

```python
decode_width = 1 + num_speculative_tokens  # 未启用推测解码时为 1
max_num_topk_rows = min(max_num_batched_tokens, max_num_seqs * decode_width)
```

## 12. 性能分析与后续核查重点

以下为源码推导的成本模型和分析建议，并非实测结论。

### 12.1 数据搬运与内存

设单行本轮 miss 数为 `M`，latent 与 RoPE 维度分别为 `D_latent` 和 `D_rope`，BF16 每元素 2 字节，则主要 KV H2D 数据量约为：

```text
H2D bytes ≈ M × (D_latent + D_rope) × 2
```

新 query 的 D2H 另外发生，top-k、请求元数据、拷贝描述符、slot 映射等传输另计。

每层预分配 resident KV 的 NPU 内存约为：

```text
resident bytes ≈ max_num_topk_rows × C × (D_latent + D_rope) × 2
```

增大 `C` 可保留更多历史 token，也会增加逐层 resident 内存和 CPU 扫描成本。

### 12.2 CPU LRU 成本

每行主要为标记 top-k、扫描 `C` 个槽和处理 miss，约 `O(C+K)`。通过 epoch 标记复用工作区，通常不需要每轮清空整个最大序列长度数组。

manager 配置最多 8 个 LRU 工作线程，实际 C++ 线程数受当前行数和工作区线程数限制；单行走单线程。

### 12.3 影响收益的关键因素

- 相邻轮次 top-k 的重合度，以及未访问 token 之后再次入选的概率。
- 请求能否持续占用同一 resident row；batch 重排可能损失历史复用。
- 推测后缀失效导致的重新加载量。
- 多 query 独立 resident row 带来的重复存储与回载。
- CPU LRU、元数据搬运、TP 同步和 sparse copy 各自的时间占比。

当前 token 先写 CPU；若被本轮 top-k 选中，再按 miss 路径 H2D。当前代码没有将新生成 KV 直接插入 resident buffer 的快速路径。

### 12.4 建议验证场景

- 连续相同 top-k、全 miss、部分重合、无效 top-k 项。
- 同一行切换请求，以及同一请求发生 batch row 迁移。
- 推测后缀在同一逻辑位置被改写后的重新加载。
- mixed batch 的累计 query 长度重定位。
- `skip_topk` 层复用描述符与 slot 映射的正确性。
- eager 与 graph 模式、单 TP 与多 TP 的结果和传输顺序一致性。

分析中默认 top-k 有效 token 不重复。C++ 的 token 标记和 miss 收集方式不应直接视为对重复索引的完整去重保证；如需覆盖这一输入，应单独验证。

## 13. 已查看的相关测试

- [test_sfa_kv_offload.py](/Users/wangbiao/Documents/Codes/vllm-ascend/tests/ut/attention/test_sfa_kv_offload.py)：PD consumer 角色识别、边界 token 分类与长 prefill 分类。
- [test_sfa_v1.py](/Users/wangbiao/Documents/Codes/vllm-ascend/tests/ut/attention/a2/test_sfa_v1.py)：backend 选择及相关 SFA 路径。
- [test_sparse_kv_offload_manager.py](/Users/wangbiao/Documents/Codes/vllm-ascend/tests/ut/distributed/kv_transfer/sparse_kv_offload/test_sparse_kv_offload_manager.py)：host/device 内存比例等逻辑。

本次只整理分析文档，没有执行上述测试。逐轮 LRU 示例是对当前 C++ 实现的静态推演，不代表 NPU 运行验证。
