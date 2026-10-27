# SFA KV Offload：Reuse Prioritized LRU 拟修改与实现方案

整理日期：2026-10-26。源码基线：当前工作区 `64ba45045`。

本文保留原始改造设计与算法参考代码。2026-10-27 已接入 CPU C++ 策略及配置开关，实际实现与验证范围见第 13 节；尚未通过 NPU 验证。

## 1. 可行性结论

可以实现。实现范围确定为现有 CPU C++ 元数据路径：Reuse Prioritized LRU 的 Hit/Miss 判定、增龄、victim 选择、有序合并及映射更新全部在 `sparse_kv_offload.cpp` 中完成，Python manager 负责配置、状态与工作区分配、参数传递及调用。复用当前 resident attention 和 miss-only H2D 搬运。

材料第 4–9 页将该方法称为 **Probation-LRU**。本文用 Reuse Prioritized LRU 指代这套规则：

- 每次有效的策略更新使已驻留 token 的 age 饱和加一。
- 本轮真正 Hit 的 age 归零。
- 本轮 Miss 加载后 age 设置为 `probation_age=P`。
- 本轮 Hit 不参与淘汰，优先回收空槽，再从其他有效槽中选择 age 最大者。
- 只移动逻辑 slot 顺序和 age，物理 KV 保持在原槽位。

其中“空槽绝对优先”和相同 age 的确定性顺序是本文补充的工程规则，材料未完整规定。该方法不需要 LFU 计数、attention 分数或额外预测模型。

**关键判断：现有实现已经在同一轮中优先保护 Hit。新增价值是跨轮的 age 与 Miss 插入位置，而不是简单把 Hit 放到队尾。**

`C>K` 才有保留本轮未访问历史 KV 的容量。`C=K` 时也能执行该策略，但不会得到同样的跨轮容量收益。收益大小必须与当前仓库基线实测比较。

## 2. 阅读依据与边界

| 依据 | 本文使用方式 |
| --- | --- |
| 用户提供的 `专利材料.pptx`，第 4–6 页 | age、Hit/Miss 区别、逻辑顺序与物理槽分离 |
| 同一材料，第 7 页 | 饱和增龄、稳定压缩、miss scan、免全量排序 |
| 同一材料，第 10 页 | 仅作为材料中的实验陈述，不视为当前仓库实测结果 |
| `SFA_KV_OFFLOAD_ANALYSIS.md` | 参考调用链、resident 存储与特殊路径分析，并以源码核实 |
| 当前工作区 Python / C++ | 拟修改位置与现有行为的最终依据 |

材料路径：`/Users/wangbiao/Documents/Patent/ProbationLRU/专利材料.pptx`。

材料中的背景性能数字、保护点和取证安排均作为资料内容阅读，不作为本次执行指令。本文讨论工程实现，不判断专利权利要求范围。

## 3. 当前代码真正在哪里实现 LRU

### 3.1 调用关系

```text
attention/sfa_kv_offload.py
  AscendSFAKVOffloadImpl._execute_sparse_flash_attention_process()
    过滤无效 top-k、展开请求身份、计算稳定前缀
    manager.onload_topk_kv(...)
      distributed/kv_transfer/sparse_kv_offload/
        sparse_kv_offload_manager.py
          _onload_topk_kv_cpu(...)
            sparse_kv_offload.cpp
              lru_resident_compact(...)
                process_one_lru_resident_row(...)
              compute_lru_resident_addrs(...)
      sparse_copy：仅 Miss 的 CPU KV → resident slot
    resident SFA：通过 current_slots 访问 KV
```

因此，仅修改 `attention/sfa_kv_offload.py` 无法完成淘汰策略改造。该文件主要提供接入参数，核心修改应落在 C++ 和 manager。

### 3.2 当前策略

`process_one_lru_resident_row()` 先扫描本轮 top-k，保护所有 Hit，再为 Miss 选择旧队列头部的非 Hit 槽。最终队列为：

```text
[未访问且未被替换的槽] + [本轮 Miss 槽] + [本轮 Hit 槽]
```

当前策略与材料中的普通逐访问 LRU 基线有差别：Hit 在同一轮已经比 Miss 更受保护，但所有剩余未访问槽都排在新 Miss 前面。没有跨轮 age，也没有让新 Miss 插入历史未访问槽中间的能力。

当前持久元数据主要是：

| 字段 | 形状 | 当前含义 |
| --- | --- | --- |
| `lru_slot_to_token_cpu_list[layer]` | `[R,C]` | 物理 slot → 逻辑 token |
| `lru_slots_cpu_list[layer]` | `[R,C]` | 逻辑淘汰顺序中的物理 slot |
| `lru_last_req_ids_cpu_list[layer]` | `[R]` | 请求身份变化时重置该行 |
| `lru_current_slots_cpu` | `[R,K]` | 本轮 attention 的 slot 索引，临时共享输出 |

`R=max_num_topk_rows`，`C=topk_buffer_size`，`K=index_topk`。状态实际按 **layer × query row** 维护。推测解码一个请求可能对应多行，不能直接改成材料所述的 layer × request 共享池。

当前 `token_mark/token_pos` 是每线程 epoch 工作区，用于查询本轮 top-k，不是跨轮持久化的 token → slot 映射。不能直接把它当作材料中的 `slot_map`。

### 3.3 当前图模式

当前图路径借助 `_launch_host_func()` 进入 Python callback，再调用 C++ 执行 CPU 决策，然后上传拷贝描述符和 `current_slots`。本方案保留这条路径。callback 只负责调用 C++ 和组织已有流程，不用 Python 循环执行逐行或逐 token 的 LRU 决策。

## 4. 拟修改文件与职责

下表保留原始计划修改清单；当前实现与验证进度见第 13 节。

| 文件或位置 | 拟修改内容 | 范围 |
| --- | --- | --- |
| `vllm_ascend/ascend_config.py`，`SparseKVOffloadConfig` | 增加策略选择、`probation_age` 及校验 | 本方案 |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`，初始化状态处 | 分配各层 age tensor、注册指针和每线程 scratch | 本方案 |
| 同文件，`onload_topk_kv()` / `_onload_topk_kv_cpu()` | 将 age 和 P 传给新的 C++ 入口，保留老策略分支 | 本方案 |
| `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp` | 新增 age 状态机、稳定分流与有序合并，保留旧入口 | 本方案 |
| `vllm_ascend/attention/sfa_kv_offload.py` | 保留稳定前缀、top-k 过滤及 resident attention 接口，必要时补充契约注释 | 无必要的行为改动 |
| `tests/ut/distributed/kv_transfer/sparse_kv_offload/` | 新增算法参考、C++ 输出对照与生命周期回归测试 | 本方案 |
| `tests/ut/test_ascend_config.py` | 新参数默认值和非法参数测试 | 本方案 |
| `tests/e2e/` 的 offload 集成测试及 `tests/e2e/nightly/` 基准 | NPU 精度、图回放、TP 和性能对比 | 集成验证 |

不需要增加模型文件、新 patch 或 model runner 行为。优先使用 manager 已有生命周期，避免扩大模型适配范围。

## 5. 配置与元数据设计

### 5.1 配置

建议通过现有 `additional_config.sparse_kv_offload_config` 增加字段，不增加环境变量：

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "eviction_policy": "reuse_prioritized_lru",
    "probation_age": 4,
    "topk_buffer_size": 4096
  }
}
```

该片段只展示新增策略参数，启动仍须满足已有 PD consumer 等约束。

- `eviction_policy`：仅接受 `lru` 和 `reuse_prioritized_lru`，默认 `lru`，保持当前行为。
- `probation_age`：整数，建议初始实验值为 4，来自材料示例，不代表最优值。
- 定义 `MAX_RESIDENT_AGE = INT32_MAX - 1`，要求 `0 <= P <= MAX_RESIDENT_AGE`，增龄采用饱和加法。
- `P=0` 合法，但此时 Hit 与新 Miss 的 age 相同，并不保证退化成当前实现。
- 保留 `C>=K`、正容量、block 对齐和线程数校验。初始化后不动态修改策略和 P。

### 5.2 新增状态

```python
# 计划字段；仅启用新策略时分配，与 lru_slots 按逻辑位置对齐。
self.lru_stamps_cpu_list[layer]  # int32 [R, C], pinned CPU memory
self.lru_stamps_ptrs[layer]     # 上述 tensor 的 data_ptr
```

对于逻辑位置 `i`：

```text
s = lru_slots[row, i]
lru_stamps[row, i] = 物理槽 s 当前 token 的 age
slot_to_token[row, s] = 物理槽 s 当前 token
```

重排时 `(slot, age)` 必须同步移动，禁止将 stamp 错按物理 slot 下标读取。空槽 age 约定为 0，但空槽优先级由有效性决定，不依赖这个数值。

新增 CPU 持久状态约占 `4 × L × R × C` 字节。例：`L=60,R=5,C=4096` 时约 4.69 MiB。实际按最大行数及 offload 层数分配，不能只按当前 batch 估算。

每线程 scratch 保存 free、survivor、hit 和新 Miss 的 slot/age，按 `O(C+K)` 预分配，可使用固定 tensor 的分段。OpenMP 每线程独享，禁止所有行共用可写 scratch。现有 `3C` slot 工作区不能在未重新核算长度的情况下直接塞入所有 age 数组。

本方案不需要新增 `[L,R,max_model_len]` 的持久 slot_map：仍用当前 epoch 标记加 resident 扫描实现 `O(C+K)` 更新。减少侵入，也避免额外的大型映射。

## 6. 每行状态转移

### 6.1 不变量

每次更新完成后必须满足：

1. `lru_slots` 是 `[0,C)` 的完整排列，物理槽不重复。
2. 队列结构为 `[free] + [有效槽按 age 降序]`，相同 age 保持规定的稳定顺序。
3. 每个有效逻辑 token 至多占一个物理槽，本轮各有效 top-k 位置都有对应 slot。
4. 所有本轮 Hit 都保持原物理槽，本轮 Miss 使用不同的非 Hit 槽。
5. age 取值始终在 `[0,MAX_RESIDENT_AGE]`。
6. SFA 启动前，Miss 的 K/V 两部分都已写入相应 resident 槽。

### 6.2 更新步骤

1. **身份重置**：请求 ID 改变时，清空 `slot_to_token`，顺序恢复 `[0..C-1]`，age 清零。沿用现有身份生成机制，不在本次新增请求迁移池。
2. **后缀失效**：对 `token>=stable_prefix_len` 的旧 resident 条目先失效，使其成为 free。即使同一逻辑位置本轮再次入选，也必须按 Miss 回载最新 KV。
3. **输入契约**：有效 token 必须在 `[0,max_model_len)` 内，忽略 `-1` 等无效项。有效 top-k 需唯一，按原始 top-k 位置组织输出。
4. **增龄与分流**：扫描旧逻辑队列，收集 free、non-hit、hit。所有有效条目先饱和增龄，Hit 再归零。non-hit 保留原有顺序。
5. **分配 Miss**：按 top-k 原始位置生成 Miss，先用 free，再依次用 age 最大的 non-hit。不得逐个处理 Miss 时淘汰尚未判定的 Hit。
6. **插入与合并**：余下 non-hit 已按 age 降序排列；新 Miss 的 age 都是 P，将两者线性合并，然后追加 age=0 的 Hit，前面保留未分配 free。
7. **输出**：更新正向物理映射、顺序和 age，生成现有格式的 `current_slots/miss_count/miss_tokens/miss_slots`，沿用地址计算与 `sparse_copy`。

相同 age 的确定性规则：既有 non-hit 排在新 Miss 前；新 Miss 按 top-k 顺序；Hit 按旧逻辑队列顺序。`P=0` 时仍按这一分组规则拼接。free 保持扫描顺序。

这里“一轮”指该 layer/query row 实际执行一次元数据更新，包括 MTP 路径的实际调用；不是所有层共享一个全局 decode 时间。真实有效行即使本轮全是无效 top-k，也执行增龄，不生成搬运；图 padding 行应通过有效行范围或 active mask 排除，不能作为真实更新。

### 6.3 为什么无需完整排序

若上轮有效区按 age 降序，统一饱和加一仍保持顺序。删去失效槽和 Hit 是稳定过滤，剩余 non-hit 仍有序。删去分配给 Miss 的最老前缀也保持顺序。

因此，将“剩余 non-hit 有序流”与“age=P 的新 Miss 流”合并，再追加归零 Hit 即可。复杂度为每行 `O(C+K)`，而不是每轮 `O(C log C)` 排序。

仅做 `[剩余 non-hit]+[Miss]+[Hit]` 不够：当部分 non-hit 的 age 小于 P 时，新 Miss 必须插到它们前面。否则数组头部不再代表最老候选。

### 6.4 重复 top-k 和容量失败

当前 C++ 的 `token_mark` 只记 token 的首个位置，但收集 Miss 会扫描所有位置；重复 token 不能直接假设已被正确去重。

本方案采用“有效 top-k 唯一”的接口契约。新策略入口应在修改持久状态前利用已有标记扫描检测重复，返回明确错误，由 host callback 外围按运行时支持的方式报告，不能静默输出部分结果或让异常穿越 C 回调边界。若后续要求支持重复项，需要单独定义 attention 对重复索引的语义，再决定是否允许多个输出位置引用同一 slot。

在唯一性和 `C>=K` 下，必有 `miss_count <= C-hit_count`。仍应校验容量；失败不得像 `min(miss,evictable)` 一样留下有效 top-k 的 `current_slots=-1`，也不得部分提交状态。元数据结构损坏属于失败，不作为合法输入继续运行。

## 7. 算法参考实现

以下是单行 Python 参考实现，用于解释和作为 UT oracle，不在生产 decode 热路径执行。输入 top-k 使用唯一的有效 token；实际调用方继续负责按 seq_len 过滤。`previous_req_id` 应随返回状态一起持久化。

```python
MAX_RESIDENT_AGE = (1 << 31) - 2

def reuse_prioritized_step(
    slot_to_token, lru_slots, lru_stamps, topk,
    req_id, previous_req_id, stable_prefix_len, max_token, probation_age,
):
    capacity = len(slot_to_token)
    assert capacity > 0
    assert len(lru_slots) == len(lru_stamps) == capacity
    assert sorted(lru_slots) == list(range(capacity))
    assert 0 <= probation_age <= MAX_RESIDENT_AGE
    selected = [(pos, t) for pos, t in enumerate(topk) if 0 <= t < max_token]
    assert len({t for _, t in selected}) == len(selected)
    assert len(selected) <= capacity
    wanted = {t for _, t in selected}
    mapping = list(slot_to_token)
    order = list(lru_slots)
    stamps = list(lru_stamps)
    if req_id != previous_req_id:
        mapping = [-1] * capacity
        order = list(range(capacity))
        stamps = [0] * capacity

    free, survivors, hits = [], [], []
    hit_lookup = {}
    stable_prefix_len = min(max(stable_prefix_len, 0), max_token)
    for slot, age in zip(order, stamps):
        token = mapping[slot]
        if token < 0 or token >= stable_prefix_len:
            mapping[slot] = -1
            free.append((slot, 0))
        elif token in wanted:
            hits.append((slot, 0))
            hit_lookup[token] = slot
        else:
            survivors.append((slot, min(age + 1, MAX_RESIDENT_AGE)))

    current = [-1] * len(topk)
    missing = []
    for pos, token in selected:
        if token in hit_lookup:
            current[pos] = hit_lookup[token]
        else:
            missing.append((pos, token))
    assert len(missing) <= len(free) + len(survivors)
    used_free = min(len(missing), len(free))
    used_old = len(missing) - used_free
    targets = free[:used_free] + survivors[:used_old]
    survivors = survivors[used_old:]
    misses, inserted = [], []
    for (pos, token), (slot, _) in zip(missing, targets):
        mapping[slot] = token
        current[pos] = slot
        misses.append((token, slot))
        inserted.append((slot, probation_age))

    # Stable merge: existing survivors with equal age precede new misses.
    split = 0
    while split < len(survivors) and survivors[split][1] >= probation_age:
        split += 1
    pairs = (
        free[used_free:] + survivors[:split] + inserted
        + survivors[split:] + hits
    )
    return (
        mapping, [s for s, _ in pairs], [a for _, a in pairs],
        current, misses, req_id,
    )
```

参考实现使用 Python 容器保证可读性；生产 C++ 应使用现有 epoch 工作区和固定 scratch，不在每行中创建 `set/dict/vector` 或进行动态分配。输入不变量的昂贵检查放在测试或调试验证中，生产保留必要的范围与容量保护。

## 8. C++ 与 manager 的接线方式

建议保留 `lru_resident_compact()` 原有 ABI，新增独立入口：

```text
reuse_prioritized_lru_resident_compact(
    <现有 lru_resident_compact 的输入输出及工作区指针>,
    lru_stamps_ptr,
    age_and_merge_workspace_ptr,
    probation_age
)
```

上面是拟议签名示意，不是已存在的 API。新入口复用 row-level OpenMP 调度，在每个线程内部执行第 6 节状态机。新线程工作区必须按 `thread_id` 偏移，age 持久数组则按 `row*C` 偏移。

manager 的具体接线顺序：

1. 仅新策略分配 age tensor 与 scratch，保存张量引用，避免裸指针失效。
2. 在 `onload_topk_kv()` 构造 callback 参数时加入对应 layer 的 age 指针、scratch 指针和 P。
3. 同步修改 `_onload_topk_kv_cpu()` 解包；按 `eviction_policy` 选择旧入口或新入口。
4. 保持 `compute_lru_resident_addrs()` 的输入输出格式不变，仍按 token 经 block table 定位 CPU 源地址。
5. 保持 H2D 目标为 `(row*C+slot)`，SFA 仍读取 `current_slots`，无需增加 hit D2D 拷贝或 selected-KV buffer。
6. 保持 eager 的直接 CPU 调用和图模式的 host callback 顺序；新策略不得另加 `.item()`、`.cpu()` 或每 token 同步。

CPU 扫描时可以先写元数据再排入 H2D，但这只是当前顺序执行链的逻辑提交：在复制完成前不能让其他流消费该映射。搬运错误应使该调用失败，重试前清空受影响状态，不能复用“映射已更新、KV 未写完”的条目。

### 8.1 C++ 热路径的性能要求

LRU 决策应在 C++ 中实现。现有 `lru_resident_compact()` 和地址计算已经采用 C++，新策略直接沿用这个边界，避免在 Python 中逐 token 判定、排序或更新状态。第 7 节 Python 代码仅用于测试对照。

- 一次 C++ 调用处理本次全部有效行，按现有 OpenMP 方式并行处理行，避免 Python 逐行跨语言调用。
- 每行复用 epoch 标记，按 `O(C+K)` 扫描、稳定分流和有序合并，不使用全量排序。
- 初始化时分配连续 int32 状态及每线程工作区，热路径不进行容器扩容或堆分配。
- 请求重置、后缀失效、Hit/Miss 判定、age 更新、victim 分配和映射更新均在 C++ 完成。地址计算继续复用 `compute_lru_resident_addrs()`。
- 沿用 `lru_max_threads` 限制线程数，小行数避免过度并行；分别测量单线程与多线程收益。
- 沿用现有元数据传输，不为 age 新增每轮 CPU/NPU 往返，age 持久保存在 CPU。

C++ 能减少解释器、对象分配和调用开销，但端到端性能仍受 top-k 元数据 D2H、host callback 调度、描述符上传及 KV H2D 影响。应同时记录 C++ 决策耗时和整步 decode 延迟，通过性能测试判断新增 age 管理开销能否被 Miss 减少抵消。

## 9. 生命周期与特殊路径

### 9.1 请求变化、batch 变化与 MTP

保留当前按 row 的请求身份比较。row 换请求时 age 必须与映射一起清零，batch 缩小时不更新非活动行。请求迁移到另一 row 时沿用现有冷启动行为，不引入跨 row 数据搬运。

MTP 后缀先失效再判断 Hit。这一点优先于任何 age 保护规则，否则可能返回相同 token 位置对应的旧 KV。

### 9.2 skip_topk

当前 `skip_topk=True` 的层复用前层 miss 描述符和 `current_slots`，仅调整层间地址基址，不独立执行 CPU LRU。不能在这些层额外运行一次 age 更新，同时又继续复用前层 victim。

本方案把连续复用相同 top-k 和布局的层视为一个策略决策组：仅源层持久 age 有效，后续层沿用其 slot 布局和 victim，并回载各自层的 KV。每组每轮只更新一次 age。

需要验证模型的复用分组在初始化后固定，且 skip 层不会在运行中切回独立决策。若不能证明这一点，新策略应在初始化时拒绝这种动态切换配置，或者补充完整状态同步后再支持。不得直接启用某 skip 层从未同步过的私有 age/mapping。

### 9.3 TP 与数据顺序

保留 TP0 写新 KV、其他 TP rank 回载前现有广播同步。各 rank 必须采用相同 P、有效 top-k 和确定性相同 age 规则，产生相同 slot 布局。

本方案不改变当前 CPU 池 GVA、BF16 MLA 数据布局、indexer cache 或 prefill 路由。独立验证 K latent 与 RoPE 部分均复制正确。

## 10. 逐轮算例与预期差异

设 `C=3,K=1,P=4`，初始物理槽为 `[B,C,A]`。初始化逻辑队列 `[B,C,A]`、对应 age `[7,6,5]`，接着依次访问 `[A]、[X]、[Y]、[Z]、[A]`，这些位置始终在稳定前缀内。

| 访问 | 当前仓库队列，淘汰端在左 | 新策略有效队列，age 降序 | 新策略 Miss |
| --- | --- | --- | --- |
| A | B,C,A | B:8,C:7,A:0 | 无 |
| X | C,A,X | C:8,X:4,A:1 | X 替换 B |
| Y | A,X,Y | X:5,Y:4,A:2 | Y 替换 C |
| Z | X,Y,Z | Y:5,Z:4,A:3 | Z 替换 X |
| A | Y,Z,A | Y:6,Z:5,A:0 | 无 |

最后一次 A 在当前策略中为 Miss，新策略中为 Hit。这个例子也适用于当前仓库的批量保护策略，因为每轮只有一个有效访问。A 如果继续长期不被访问，age 仍会超过 P 并最终被淘汰，不是永久保护。

## 11. 验证计划与验收标准

### 11.1 算法 UT

新增参考实现与 C++ 实现的逐轮比较，比较 slot 映射、顺序、age、Miss 列表和 `current_slots`，不能只比较命中率。

| 用例 | 必须验证的结果 |
| --- | --- |
| 第 10 节抗扫描序列 | 最后 A 命中，新 Miss 按 P 插入 |
| 冷启动 / 全空槽 | 按 free 顺序分配，无有效缓存被提前淘汰 |
| 混合空槽与有效槽 | 即使空槽位于旧逻辑顺序后部，也先用空槽 |
| 全 Hit | 无 H2D，Hit age 清零，slot 保持不变 |
| 全 Miss / C=K | 本轮全量可访问，slot 不重复 |
| 多 Hit、多 Miss | 本轮 Hit 绝不被前面的 Miss 覆盖 |
| non-hit age 小于 P | 新 Miss 插到较年轻的历史槽前，而非统一放尾部 |
| 相同 age / P=0 | 符合确定性 tie 规则，TP 输出一致 |
| age 到达上限 | 饱和增龄，无 int32 回绕 |
| req_id 变化 | 映射、顺序、age 一起重置 |
| MTP 后缀失效 | 旧 KV 即使 age=0 也必须失效并重新回载 |
| 无效 top-k / 全无效 | 无效项 slot=-1，真实行增龄，Miss=0 |
| 重复 token / 超容量 | 修改状态前报告失败，不返回半成品 |
| 多线程与单线程 | 完整输出相同，无 scratch 串行污染 |
| 多层 / 多行 | 各独立决策组状态隔离 |
| skip_topk 决策组 | 只更新源层策略，后层搬运布局一致 |
| 策略为 lru | 保持旧入口与旧输出 |

随机差分测试覆盖容量、K、P、身份切换、后缀回退和无效输入。C++ 指针入口必须在有 torch_npu/ACL 的开发环境编译测试；纯 Python oracle 通过不等于 C++ 或 NPU 通过。

### 11.2 NPU 集成

同一模型与固定输入，分别运行当前策略与新策略：

1. 比较每轮 top-k、resident 中相应 KV 与 CPU 源 KV，核验数据及索引对应关系。
2. 比较 eager 和图回放的 attention 输出、最终生成结果，使用项目已有精度容差。
3. 覆盖普通 decode、MTP 接受/拒绝、请求完成和替换、batch 伸缩、TP>1、固定 skip_topk 分组。
4. 覆盖纯 prefill 与 mixed 的已有支持场景，确认 decode 改造不影响原路由。
5. 对图模式反复回放，确认捕获阶段的测试执行没有将状态错误带入正式请求。

缓存策略只影响搬运与驻留，不应改变相同 top-k 对应的实际 KV 或 attention 语义。出现输出偏差先作为一致性问题排查。

### 11.3 性能与实验口径

以当前仓库策略为基线，保持模型、top-k、C、上下文长度、batch、TP、图模式、MTP、输入和预热一致。容量 factor 仅作实验口径，当前配置直接指定 C，且仍需 block 对齐。

建议对比 `C/K=1,2,3` 及 `P=0,1,2,4,8,16`，这些是参数扫描点，不是生产推荐值。分别记录：

- 有效 token Hit Rate，按层报告，并以有效访问总数加权汇总，排除 padding。
- Miss token 数及主 KV H2D Bytes/Token，区分主 KV 流量与元数据流量。
- 元数据计算时间、host callback 时间和搬运等待占比。
- decode 延迟及尾延迟、吞吐，分小 batch 图模式与大 batch 场景。
- CPU pinned memory、NPU HBM 占用和各线程工作区大小。

单行主 KV 搬运量估算为 `M × (D_latent+D_rope) × 2` 字节，M 为该行 Miss 数，数据类型为当前 BF16。跨层/行/rank 汇总时需明确统计范围。

材料第 10 页给出的 `85.02% → 89.03%` 是材料中的 factor=3 实验陈述。它没有证明当前仓库 CPU 路径、更换数据集或不同 batch 会获得同样的 4.01 个百分点收益。

验收要求：状态机与 oracle 一致、KV/attention 精度通过、旧策略无回归；新增策略的性能收益或退化按真实实验记录。命中率提高不能替代吞吐验证，也不预设一定有正收益。

## 12. 建议实施顺序与本次交付状态

1. 落实配置及单行参考状态机，用逐轮样例和随机序列固定语义。
2. 实现 C++ age 更新与有序合并，接入 manager，保留旧策略回退入口。
3. 完成 C++ 差分、图回放、MTP、TP 和 skip_topk 生命周期验证。
4. 在 Ascend NPU 上完成精度和参数扫描，依据端到端结果决定 P。

原方案阶段仅交付 Markdown 设计。当前已完成的生产代码、配置项和测试见第 13 节；第 7 节仍为 Python 算法参考，不是运行时接口。

### 12.1 本次文档内算法核对记录

已从第 7 节提取 Python 代码直接执行：

- 第 10 节抗扫描序列与表中状态一致，最后访问 A 为 Hit。
- 固定随机种子 42，100 组、每组 100 轮，共 10,000 次状态转移，检查 slot 排列、有效区 age 降序、token 唯一性和本轮 token/slot 对应关系，均通过。
- 这些检查只覆盖参考算法，不代表 C++ 差分测试或 NPU 集成验证完成。
- 尝试运行 `bash format.sh ci`，因本机未安装 `pre-commit` 在依赖检查阶段退出，完整仓库格式检查未执行。

## 13. CPU C++ 实现记录（2026-10-27）

已增加 `sparse_kv_offload_config.eviction_policy`，取值为 `lru`（默认）或 `reuse_prioritized_lru`，以及整数 `probation_age`（默认 4）。Python manager 仅为新策略分配 age 状态和线程工作区，并在现有 callback 中选择 C++ 入口。

生产 CPU 决策整理到 `distributed/kv_transfer/sparse_kv_offload/lru_resident.h`，由原 `sparse_kv_offload.cpp` 包含并注册 Python 入口。原有 LRU 算法与入口保留。这样测试能直接编译并调用生产算法，无需 torch/ACL/NPU。新策略每行使用既有 3C 个 slot scratch 和新增 C 个 age scratch，不进行动态分配或全量排序。

与第 6.4 节的原始提案不同，实现支持重复 top-k：同轮同一 token 仅分配和回载一次，各原始 attention 位置返回同一物理 slot。同轮重复不会将首次 Miss 提升为 Hit；第 7 节参考代码仍以唯一有效 token 为前提，实际测试另用支持重复索引的排序 oracle。

`skip_topk` 是 attention 实例初始化时固定的选项，复用层继续使用源层的描述符和布局，不另行调用策略或更新 age。现有 CPU host callback、H2D 地址计算、resident attention 和 TP 同步路径保持不变。

已新增测试：

- `test_lru_resident.py`：直接编译生产 C++，与排序 oracle 随机对照，并检查旧 LRU 与新策略的抗扫描差异。
- `TestSparseKVOffloadEvictionConfig`：默认值、策略选择、P 边界及非法配置。
- `TestResidentPlannerDispatch`：两种策略的 callback 分发、层 age 指针及地址计算输入。

本机生产 C++ 测试通过，构造的 C=4096、K=2048 扫描序列累计 Miss 从 10240 降到 8192。本机没有 OpenMP 开发库，运行的是串行 fallback；多线程执行仍需在带 OpenMP 的环境验证。本机没有 vLLM，配置和 callback 测试通过抽取真实定义隔离执行，完整 UT 仍需在开发环境运行。NPU 精度、图回放、TP、PD 及端到端吞吐尚未验证。
