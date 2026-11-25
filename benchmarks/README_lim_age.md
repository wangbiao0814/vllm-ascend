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
The resulting latency measures LIM device work, not end-to-end decode latency
or Python launch overhead. Device thermal state and clock variation still
require repeated runs and reversed run order for rigorous measurements.

Each JSON contains hardware/software identifiers, trace seeds/core IDs and:

- `kernel_mean_us`, `kernel_median_us`, `kernel_p95_us`: sampled device latency.
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
0 through 31 are supported. These are low-level experimental operator arguments,
not model additional-config keys. Nonzero values intentionally break age policy
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
nonzero flags). Compare medians in the same mode; timings are not necessarily
additive because removing work can change overlap and synchronization behavior.

For all-hit, normal versus no-hit-clear estimates the effect of Hit clearing;
normal versus no-increment estimates increment overhead. Flags 3/7 retain age
initialization checks and barriers but skip both steady-state updates involved
in all-hit. `no-refill` should retain all-hit work because there are no misses.
For all-miss, normal versus no-refill isolates the effect of Miss P assignment;
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
locally. The diagnostic NPU tests cover all 32 flag combinations and expected
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
