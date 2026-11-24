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
