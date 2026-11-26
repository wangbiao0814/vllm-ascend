# Score versus age LIM benchmark

Run from the repository root, in the NPU environment with both custom operators
compiled. This standalone program imports operator test helpers without loading
pytest conftests. It requires torch, torch-npu, pytest and the matching vLLM stack.
It does not use the model engine or its hit-rate collector.

```bash
python -m benchmarks.benchmark_lim_age --mode both \
  --length 16384 --capacity 8192 --steps 64 \
  --core-fraction 0.5 --correlation 0.95 --probation-age 4 \
  --output lim_age_results.json
```

Compare different fixed-core fractions using identical seeds and parameters:

```bash
for fraction in 0 0.25 0.5 0.75 0.9; do
  python -m benchmarks.benchmark_lim_age --core-fraction "$fraction" \
    --output "lim_age_core_${fraction}.json"
done
```

The workload uses one request, one Query per call (MTP=0), 32 heads, FP16/BF16
and a fixed historical prefix. A seeded, scattered set of
`round(2048 * core_fraction)` sources gets a strong fixed key component.
Remaining key components are Gaussian. Queries follow a correlated Gaussian
process controlled by `correlation`; lowering it reduces temporal locality.
Only the first index head contributes scores. This is a controlled synthetic
workload, not a model trace or a reproduction of model quality/performance.
The core parameter describes the generated high-score set, not a measured hit
rate; actual selected Source sets are checked between the two operators.

Both policies start from the same cache map: Sources `[0, capacity)` resident
in Slots of the same ID. Ages start at P. Each policy then keeps its own cache
state across the trace. Warmup and graph capture mutations are reset before
measurement. First-fill is excluded; the initially missing accesses of the
steady trace are included in hit-rate totals.

For sampled steps, each timed repetition restores exactly the same pre-step
cache/age state. Only the last repetition's state is kept, so additional timing
repetitions do not artificially improve the hit rate. NPU events bracket the
operator invocation or graph replay. Query copies, state restores, CPU checks,
JSON output, KV copying and Sparse Attention are outside these intervals.
Counters and output checks run after synchronization, outside capture.
The resulting Event interval is not an isolated kernel duration: device waits
and gaps while Python submits work can appear between the recorded events.
It does not measure end-to-end decode latency. Use the device profiler described
below to distinguish kernel execution from those gaps. Device thermal state and clock variation still
require repeated runs and reversed run order for rigorous measurements.

Each JSON contains hardware/software identifiers, trace seeds/core IDs and:

- `event_mean_us`, `event_median_us`, `event_p95_us`: sampled Event intervals.
  The legacy `kernel_*` keys remain aliases for compatibility, not isolated
  kernel durations. `timing_scope` identifies whether timing was performed.
- `hits`, `misses`, `hit_rate`, `misses_per_step`: actual TopK occurrence counts
  from the operators, not a Python approximation of the eviction policy.
- `policy` and `mode`: score/age and eager/graph respectively.

Compare age/score latency within the same mode and configuration. A lower miss
rate does not by itself establish a latency or throughput improvement; evaluate
KV-copy and full-model decode separately. This script does not predict speedup.

CPU-only specification validation (does not execute either kernel):

```bash
python -m benchmarks.benchmark_lim_age --dry-run --output /tmp/lim_age_trace.json
```

NPU compilation and execution of this benchmark remain to be verified on server.

## All-hit diagnostic

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit --mode both \
  --length 16384 --capacity 8192 --steps 32 --repeats 30 \
  --latency-samples 8 --output lim_age_all_hit.json
```

This uses exactly 2048 scattered positive-score core sources, zero-score
background sources, and a fixed unit Query in the first dimension. All core
sources are resident before the first measured call; the remaining slots contain
filler sources. Both policies start with identical mappings. The Query/Key
shapes, full prefix scan and age updates are retained. `core-fraction` and
`correlation` do not control this diagnostic. No first-fill call is measured.

Every measured step asserts zero route/request misses, the exact planted TopK
set, and an unchanged cache map. Both policies must report `hit_rate=1.0` and
`misses=0`; a failure stops the run rather than silently reporting a mixed case.
This excludes ordinary victim selection and refill work, while retaining age
increment and 2048 Hit-age clears. It does not isolate increment from Hit clears;
further ablations/profiling are needed to attribute their individual costs.
The full attention matrix arithmetic still runs, but the data distribution is
intentionally artificial. Results diagnose maintenance overhead rather than
representative model throughput.

## Diagnostic ablations (requires rebuilding the extension)

The age operator now accepts an optional trailing `diagnostic_flags=0` integer.
Default zero keeps the full age policy. Bit 1 skips Hit-age clearing, bit 2
skips scan-time increment, and bit 4 skips Miss-age assignment to P. Values
0 through 1023 are supported (see the probe and FinishAge flags below). These are low-level experimental operator arguments,
not model additional-config keys. Skip flags intentionally break age policy
semantics and must not be used to draw conclusions about production hit rates.
The score operator is unchanged. Initialization, score protection, sorting, and
existing global barriers remain enabled in all ablations.

Rebuild the Ascend extension once, then run all variants without rebuilding:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit --ablation all \
  --mode both --length 16384 --capacity 8192 --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_age_all_hit_ablation.json

python -m benchmarks.benchmark_lim_age --scenario all-miss --ablation all \
  --mode both --length 16384 --capacity 8192 --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_age_all_miss_ablation.json
```

| Variant | Flags | Increment | Hit clear | Miss P assignment |
| --- | --- | --- | --- | --- |
| `normal` | 0 | on | on | on |
| `no-hit-clear` | 1 | on | off | on |
| `no-increment` | 2 | off | on | on |
| `no-increment-or-hit-clear` | 3 | off | off | on |
| `no-refill` | 4 | on | on | off |
| `no-age-writes` | 7 | off | off | off |

Use `--ablation no-hit-clear` etc. to run a single age variant plus score.
Default `--ablation normal` retains the original benchmark behavior. Nonzero
flags are restricted to the controlled all-hit/all-miss scenarios in this CLI.
JSON rows include `ablation`, `diagnostic_flags`, and `policy_valid` (false for
skip flags; mask 64 retains the policy). Compare medians in the same mode; timings are not necessarily
additive because removing work can change overlap and synchronization behavior.

For all-hit, normal versus no-hit-clear estimates the effect of Hit clearing;
normal versus no-increment estimates increment overhead. Flags 3/7 retain age
initialization checks and barriers but skip both steady-state updates involved
in all-hit. `no-refill` also skips FinishAge metadata checks and the count
read, even when there are no misses. Use `finish-breakdown` below to distinguish
these from actual Miss P assignment. For all-miss, normal versus no-refill
includes both Miss P assignment and these checks/count-read costs;
no-hit-clear should retain almost the same work because there are no hits.

All-miss uses the same fixed positive-score core construction as all-hit, but
initial residents are selected only from background sources. It requires
`length >= capacity + 2048`. The initial map and ages are restored **before
every trace step and every timing repetition**, outside the timed interval.
Every step asserts exactly 2048 route/request misses. This measures repeated
independent all-miss calls, not a naturally evolving all-miss workload.
Victim ordering may change between diagnostic variants because age semantics
are deliberately altered. All-hit variants still require an unchanged map.

Validation commands:

```bash
python -m unittest discover \
  -s tests/ut/distributed/kv_transfer/sparse_kv_offload \
  -p test_lim_ablation_benchmark.py
python -m pytest --noconftest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_age_manage.py \
  -k diagnostic_flags
```

CPU workload checks and existing native age arithmetic checks have been run
locally. The diagnostic NPU tests cover the original 128 flag combinations, selected new masks, and expected
age state changes, but CANN compilation and NPU execution remain unverified
locally. This branch is based on the previously pushed single-query version;
it does not include unrelated uncommitted multi-query work from another checkout.

### ACLNN ABI regeneration fix

The diagnostic binding uses the versioned public ACLNN V2 entry points. The
original single-attribute public ABI remains available and forwards with flags
zero. Forwarders include the generated inner header instead of manually declaring
its ABI, so stale generated signatures now fail during compilation.

The opbuild generation command now depends on the operator definitions and the
generator library. Previously an incremental build could rebuild that library
without regenerating existing ACLNN source/header outputs. A CPU CMake regression
test reproduces this failure before the fix and verifies regeneration afterward.
This explains a possible `topkSrcIdsOut != nullptr` failure when a new diagnostic
integer was interpreted as an output pointer; actual runtime library selection
must still match the rebuilt package.

After updating this branch, rebuild in the same server environment and launch a
new Python process:

```bash
python3 -m pip install -v -e . --no-deps --no-build-isolation
```

Inspect the build log for regenerated inner ACLNN sources. If V2 API lookup
fails, inspect the loaded vendor-library paths for an older installation.
The CPU tests validate generation dependencies, forwarding argument positions,
legacy flags-zero forwarding, and compile-time rejection of a stale header;
they do not replace CANN compilation or NPU runtime validation.

### Buffered Hit clear

The latest normal age path groups Hit Source IDs into temporary 512-source
buckets in existing UB buffers. Each touched age block is loaded once, its Hit
positions are set to zero in UB, and the block is written back once by DMA.
Untouched blocks are not copied by this stage. Hit clear no longer performs one
GM scalar write/cache-line flush per Hit; Miss assignment still uses the original
path. Increment, candidate selection, sorting, initialization and diagnostic flag
meanings are unchanged. No persistent state or Slot-to-Source reverse map is added.

Bucket construction and local clearing still use scalar UB operations in this
first implementation. It is not a hardware Scatter implementation. The existing
buffers provide 4096 bucket heads for the maximum 2^21-source capacity and 2048
linked-list nodes. Work is proportional to the number of heads initialized,
Hits and touched blocks, rather than scanning all Hits separately for every block.
At 16384-source capacity, at most 32 blocks are touched. For very long histories
with scattered Hits, additional block DMA can be substantial; speedup is not
established until NPU measurements are available.

Rebuild the extension and run the same all-hit benchmark. Compare normal age
against `no-hit-clear` in the new binary, and compare normal against the previous
binary with identical parameters. All-hit must still report zero Misses and
unchanged mappings. Existing graph, mixed Hit/Miss, diagnostic-flag and long-Source
tests remain applicable; a new partial-block all-hit NPU regression checks the
last block and preservation of non-Hit ages. The native CPU regression exercises
the shared bucket/clear helpers with invalid IDs, repeated Hits, Miss-prefix
exclusion, empty Hit sets, partial tails, and the maximum Source capacity.

### Hit-clear stage breakdown

The diagnostic flags now also support mask value 8 (build Hit buckets only,
then return before age block DMA) and 16 (build buckets and DMA-read/write each
block, but skip local zeroing). Both deliberately leave Hit ages uncleared.
Bit 1 still skips the entire Hit-clear function; if combined with 8/16 it takes
precedence. Bit 8 takes precedence over 16. No new ABI parameter is added;
rebuild the extension for the new flags and range checks.

For the current all-hit investigation run:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation hit-clear-breakdown --mode both \
  --length 16384 --capacity 8192 --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_hit_clear_breakdown.json
```

This runs score plus four age variants with increment disabled throughout:

| Age variant | Flags | Build buckets | Read/write age blocks | Clear locally |
| --- | --- | --- | --- | --- |
| `no-increment-or-hit-clear` | 3 | off | off | off |
| `no-increment-hit-buckets-only` | 10 | on | off | off |
| `no-increment-hit-copy-only` | 18 | on | on | off |
| `no-increment` | 2 | on | on | on |

For measurements in the same mode and metric, compare flags 10 minus 3 for the
bucket-stage effect, 18 minus 10 for block DMA/synchronization, and 2 minus 18
for local clearing/list traversal. These differences are estimates, not isolated
hardware instruction costs: removing work can change pipeline overlap, cache
behavior and compiler scheduling. All variants retain zero Misses and an
unchanged slot map in all-hit; ages intentionally differ. Kernel profiling is
still needed to validate attribution.

Use `--ablation hit-buckets-only` or `--ablation hit-copy-only` to retain increment
while testing either new mode. `--ablation all` includes both new modes and their
no-increment combinations. JSON rows record the exact flags. Existing flag
values 0 through 7 and the V2 ACLNN parameter order are unchanged.

### Sorted Source runs instead of scalar linked lists

The latest Hit-clear implementation uses the existing AscendC `SortAll` path
on 2048 float Source keys and full uint32 Source payloads. The Miss prefix is
masked to -1 in the local copy before sorting; public TopK outputs are untouched.
Sources up to the supported 2^21 capacity are exactly representable as float
keys, and payload extraction retains the full Source ID without score tag bits.

One linear scan records boundaries of consecutive 512-source blocks. Metadata
is written only on a block transition, replacing per-Hit head/next writes. Local
clearing walks contiguous sorted IDs and no longer loads a `next` pointer for
each Hit. Age blocks still use one DMA read and one DMA write each. Existing UB
buffers are reused, with no persistent state, reverse map or ABI change.

Historical benchmark labels containing `buckets` are retained for compatibility,
but their grouping stage now measures vector sorting plus run construction.
Flags 8/10 stop after this stage; 16/18 include DMA but skip the linear local
clear. Run the same `hit-clear-breakdown` command and compare against the previous
binary under identical eager settings. Compare full-normal latency as well;
a smaller stage difference alone does not establish an overall improvement.

A fixed 2048-element sort is performed for every nonempty Hit set. This targets
the measured high-Hit case and may cost more for a very small Hit set; all-Miss
still returns before sorting. Sorting performance, compiler behavior and NPU
correctness need hardware validation. CPU tests exercise the actual run/clear
helpers against an elementwise oracle, including long IDs, duplicate Hits,
invalid IDs, Miss-prefix exclusion, partial blocks and maximum Source capacity;
they substitute `std::sort` for the device sort and cannot verify that SDK stage.

The age CMake definition now also declares the original LIM directory as a
shared kernel dependency. Its headers are staged and included in prepared-input
cache identity, including age-only builds, to prevent missing or stale shared
headers. A CPU CMake test verifies staging through the real source-copy helper.
This is source dependency staging, not installation of the score operator API;
a benchmark comparing score and age still requires both APIs installed.

### Sort-only and vector versus scalar run grouping

The default grouping stage now computes block IDs and predecessor/successor
boundaries with vector operations, then compacts block IDs and start/end offsets.
The previous per-Source scalar run builder is retained with mask value 64.
Mask value 32 stops after sorting and extracting Source payloads, before either
run builder. Existing masks and the V2 argument list are unchanged. Rebuild once
for the expanded [0,1023] flag range and new kernel logic.

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation hit-group-breakdown --mode eager \
  --length 16384 --capacity 8192 --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_hit_group_breakdown.json
```

This runs score and these age variants, all with increment disabled:

| Age variant | Flags | Work after sorting |
| --- | --- | --- |
| `no-increment-or-hit-clear` | 3 | skips sorting and the entire clear stage |
| `no-increment-hit-sort-only` | 34 | none |
| `scalar-no-increment-hit-buckets-only` | 74 | scalar run grouping only |
| `no-increment-hit-buckets-only` | 10 | vector run grouping only |
| `scalar-no-increment` | 66 | scalar grouping plus DMA and local clearing |
| `no-increment` | 2 | vector grouping plus DMA and local clearing |

Subtract flags 3 from 34 to estimate sorting/extraction cost. Compare 74 against
10 to compare grouping implementations, and 66 against 2 to compare complete
clear paths in the same binary. Final full-normal (`--ablation normal`) timings
remain necessary. Differences are observational effects, not additive hardware
instruction costs. Graph event timing remains unsuitable for fine attribution
until checked against device profiling.

Use `--ablation scalar-hit-runs` (flags 64) for the old grouping algorithm with
full age semantics and increment enabled; it is also allowed with trace mode.
Only flags 0/64 represent full age policies. All other named variants disable
state updates. `policy_valid` reflects this distinction. Flag 1 skips the whole
clear function; otherwise 32 stops after sort, otherwise 64 selects scalar
instead of vector grouping, and 8/16 keep their prior stage meanings.

Vector grouping reuses the dead sort-input buffer as scratch (4*2048+320 B32
words). Masks have separate 256-byte regions, vector operands have aligned starts,
and prefix/suffix sentinels permit aligned neighbor Gather bases. Invalid IDs
are excluded before comparing neighbors, including out-of-range IDs in a partial
last block. Full Source payloads and external TopK order remain unchanged.
Invalid or inconsistent compaction counts fall back to scalar grouping; profiling
is needed to confirm the fast path on the target SDK. No persistent state, reverse
map or additional UB allocation is introduced.

Implementation references: [GatherMask counter mode](https://www.hiascend.com/doc_center/source/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0071.html)
and the repository's existing Compare/GatherMask compaction paths. A CPU semantic
model executes the actual vector helper and checks every block/start/end against
the scalar builder across 105 cases. It also checks local operand alignment and
scratch bounds. It does not model instruction scheduling, target SDK support or
NPU synchronization. NPU tests cover all 128 flag combinations, invalid masks,
long Source IDs, partial blocks and graph state updates; these remain unrun
locally. CANN compilation and hardware speedup require server verification.

## Untimed compaction probes and FinishAge breakdown

Rebuild the age extension and kernel once for flags [0,1023]. The V2 ABI argument
list is unchanged. Normal flags zero do not export probe counters. These changes
are experimental diagnostics; CANN compilation and NPU execution need server
validation.

First inspect vector run compaction and fallback using identical saved inputs:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation hit-group-breakdown --mode eager --steps 8 \
  --diagnostics --probe-calls 3 --output lim_age_probes.json
```

Each age result contains `probe.rows` with `raw_block_count`, `raw_start_count`,
`raw_end_count`, `effective_runs`, `path` and `fallback_count`. Vector compaction
should produce equal positive counts; `scalar_fallback` identifies a failed count
check followed by the scalar builder. Counts of -1 mean the vector stage did not
run (for example, scalar grouping or sort-only). `fallback_count` at probe level
is the sum across calls, not across all benchmark steps. Each probe resets the
last sampled pre-step snapshot, runs eagerly, and is outside the timing/profiler
pass. A result whose outer mode is graph still labels its probe mode as eager.

Mask 128 exports 16 INT32 counters into the unused final 16 cells of each request's
`miss_dst_slots` output row. The valid Miss plan is unchanged: the largest valid
plan is 32640 entries and the row has 32768. Magic/version validation catches an
uninstrumented binary. No new persistent state or output argument is required.
Use this bit only for untimed probes; it adds a final DMA and synchronization.

`finish-breakdown` runs score plus these age variants, without changing increment
or Hit-clear settings:

| Variant | Flags | FinishAge metadata checks | Count read | Miss P writes |
| --- | --- | --- | --- | --- |
| `normal` | 0 | yes | yes | when count > 0 |
| `no-refill` | 4 | no | no | no |
| `finish-check-only` | 512 | yes | no | no |
| `finish-count-only` | 256 | yes | yes | no |
| `scalar-hit-runs` | 64 | yes | yes | when count > 0 |
| `scalar-no-refill` | 68 | no | no | no |
| `scalar-finish-check-only` | 576 | yes | no | no |
| `scalar-finish-count-only` | 320 | yes | yes | no |

Skip-refill takes precedence over check-only, which takes precedence over
count-only. With probes enabled, `finish_phase` is 0 if skipped/not applicable,
1 after metadata validation, 2 after reading count, and 3 before assigning P to
Miss sources. `finish_count` is -1 when unread. In all-hit, normal should report
count 0 / phase 2 and make no P writes. A nonzero count here would contradict the
setup and needs investigation.

Run ordinary timing separately, without either instrumentation option:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation finish-breakdown --mode eager --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_age_finish_timing.json
```

## Device profiler pass

Use the [torch-npu profiler](https://github.com/Ascend/pytorch/blob/master/docs/en/developer_notes/ascend_pytorch_profiler_user_guide.md)
to obtain actual kernel durations and the CPU/device timeline:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation finish-breakdown --mode eager --steps 8 \
  --profile-dir lim_age_profiles --profile-calls 3 \
  --diagnostics --probe-calls 3 --output lim_age_profile_report.json
```

For graph comparison, repeat with `--mode graph` and another output filename.
Each variant gets its own unique directory. JSON includes paths to exported
`kernel_details.csv` and `trace_view.json`. The profiler uses CPU/NPU activities,
Level1 and Text export. The normal kernel/graph is launched without the probe bit;
counter collection happens after the profiler closes. Each launch restores the
same saved pre-step state. Restores appear in the raw profile outside
`LIM/<mode>-<policy>-<ablation>/callN` record-function regions; compare LIM kernels,
not all device activity totals. Graph profiles may contain multiple tasks, so
inspect the timeline as well as individual task durations.

If either `--diagnostics` or `--profile-dir` is present, Event measurement is
entirely disabled: `timed_calls=0`, timing values are null and
`timing_scope=not_measured`. Profiler overhead must not be compared to the regular
benchmark Event intervals. Share the JSON and the exported CSV/trace files for
further attribution; JSON timing fields alone cannot resolve device stalls.

## Age update optimization comparison

Rebuild the age extension/kernel. Default grouping now uses the scalar run
builder. The previous vector builder remains available with mask 1024;
mask 64 explicitly selects scalar grouping and takes precedence over 1024.
The diagnostic mask range is now [0,8191].

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation age-update-breakdown --mode eager --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_age_update_comparison.json
```

The score baseline is followed by these full-policy variants:

| Variant | Flags | Age update |
| --- | --- | --- |
| `normal` | 0 | Scalar grouping, original local scalar clears and scan increment |
| `range-hit-clear` | 2048 | Scalar grouping; clear truly consecutive Hits with aligned vector interiors |
| `merged-age-update` | 6144 | Defer increment until final TopK; increment and clear in one block pass |
| `vector-hit-runs` | 1024 | Previous vector grouping, original scalar clears and scan increment |

Range clear uses scalar edges and 32-byte-aligned vector interiors. Gaps between
Hits are preserved. Merged mode disables scan-time increment, then walks the
entire valid history before eviction, using the old map to distinguish resident
entries. It increments resident ages with saturation, zeros nonresident ages,
overrides actual Hits to zero, and writes each block once. All-miss still updates
all blocks; first-fill initialization and final Miss P assignment are unchanged.
No reverse map or persistent state is added. The merged pass uses existing UB
scratch. It may lose scan/computation overlap, so speedup requires measurement.

Merged mode cannot combine with skip-increment, skip-Hit-clear, grouping-only,
copy-only or sort-only flags; the binding and tiler reject these combinations.
Other diagnostic bits retain their meaning. Historical breakdowns without mask
1024 now use scalar grouping. Use the explicit variants above for scalar/vector
comparisons. Add `--diagnostics --profile-dir lim_age_update_profiles` for an
untimed profiler pass; report actual kernel durations separately from Event times.
CPU helper/model checks do not establish CANN compatibility or NPU performance.

## Double-buffer merged update experiment

`merged-double-buffer` (flags 14336 = 6144 + 8192) preserves the merged update
policy and uses two disjoint banks of the existing sort scratch. Each bank owns
an age tile, resident map tile, float values, sign/zero values and compare mask.
Two banks occupy 20 KiB within the existing 64-KiB buffer; no new UB buffer or
persistent state is allocated. A compile-time bound checks this layout.

The first block is loaded before the loop. After waiting for the current block,
the next block's reads are queued before current-block computation. Writeback is
not waited immediately: each bank has a separately allocated MTE3-to-MTE2 event,
consumed before that bank is reloaded. The final drain consumes both pending
events, waits for scratch reuse and releases the event IDs before eviction.
This offers overlap; actual overlap and speedup require NPU profiling.

Rebuild the age extension/kernel, then compare the single- and double-buffer
versions together:

```bash
python -m benchmarks.benchmark_lim_age --scenario all-hit \
  --ablation age-update-breakdown --mode eager --steps 32 \
  --repeats 30 --latency-samples 8 --output lim_age_double_buffer.json
```

The breakdown now includes `merged-double-buffer`. Run the same command with
`--scenario all-miss` to check the no-Hit path. For NPU correctness/graph tests:

```bash
python -m pytest --noconftest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_age_manage.py \
  -k 'update_variants or grouping_match or all_hit_partial_block or probe_preserves or merged_all_miss'
```

The mask range is [0,16383]; bit 8192 requires merged update (4096). Existing
merged-stage compatibility checks still apply. This is benchmark/operator-only;
no model config option or default change is introduced. Local CPU checks cannot
validate hardware events, CANN compilation or device performance.
