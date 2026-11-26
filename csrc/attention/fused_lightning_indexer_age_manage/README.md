# FusedLightningIndexerAgeManage (experimental)

This opt-in A2/A3 FP16/BF16 operator reuses the production
FusedLightningIndexerManage QK, TopK, payload decoding and Miss management.
It changes eviction ranking to descending resident age. It supports **one Query
per request (MTP=0)**. C8 Indexer and `use_fused_overlap=true` are rejected.
The default remains the original score policy.

## Behavior

- `source_ages_pool`: contiguous INT32 NPU tensor, identical shape and Source
  indexing to `cache_slots_pool`; aliases an output and persists across replay.
- Only the even AIV increments each Source block during the complete Indexer
  scan, reusing the existing pre-score UB scratch. Only resident entries age.
- After TopK publication and synchronization, the request owner clears true Hit
  ages. Local/provisional TopK never clears ages.
- Original hashed block scan, stop condition, score-threshold protection and
  threshold-tie fallback are retained. Within normal candidate sorting, larger
  age wins. The exceptional threshold-tie fallback retains its original Source
  traversal order; it is not a full age sort of tied candidates.
- First-fill and newly installed Miss tokens get `probation_age` (P). Identity
  transition initializes the retained prefix to P before scan-time increment.
- Ages saturate at 524287. This bound preserves integer-age ordering when the
  existing long-Source codec adds its four source tag bits to the float key.
  Equal ages may use the existing hardware sort tie order.
- State reset follows request-state/Slot-map rebuilding; callers must signal
  first fill for reused pool rows rather than silently reusing stale mapping.
- No Slot-to-Source reverse map, candidate multiplier or new cursor is added.
  Score workspace and the score-based original operator remain available.

Persistent overhead is four bytes per Source per pool row per LIM state owner;
for example, a 131072-Source row adds 512 KiB. Sharing layers follow the existing
LIM owner. This is not four bytes per HBM resident slot. The extra age DMA and
scalar hit/refill cache-line flushes require hardware performance validation.

## Configuration

Add to the Decode service's existing additional config:

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": false,
    "eviction_policy": "age",
    "probation_age": 4,
    "collect_hit_rate": true
  }
}
```

Keep the deployment's other required options. Use `eviction_policy="score"` for
the original baseline, and do not enable speculative decoding for the age run.
The new extension must be rebuilt; Python-only updates are insufficient.

## Server validation

Build using the repository's normal matching CANN, torch-npu and vLLM environment.
Ensure the custom-op build includes `fused_lightning_indexer_age_manage` (both
A2/A3 build allowlists include it). Run:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_age_manage.py
```

Tests cover FP16/BF16, mixed first-fill/steady requests, P=0/4/MAX, Hit/Miss age
updates, unchanged unused pool rows, victim ordering within the original scan,
identity-to-sparse transition, long Source IDs, illegal device metadata and
ACLGraph replay with first-fill-to-steady switching. These tests have **not**
been run on NPU locally. Kernel compilation and model accuracy/performance need
server verification; this directory is an experimental first implementation.

For end-to-end comparison, keep prompts, cache budget, batch size and all other
settings fixed. Compare per-layer hit/Miss summaries and Decode latency, with
collection disabled for final latency measurements. Hit-rate improvement alone
does not establish a throughput improvement.

## Diagnostic-only maintenance flags

The optional trailing operator argument `diagnostic_flags=0` supports a bitmask:
1 skips Hit clear, 2 skips scan-time increment, and 4 skips Miss assignment to P.
The default zero preserves normal behavior. Nonzero flags intentionally change
the age policy; use only for controlled performance ablations, never production
inference or hit-rate quality comparisons. Rebuild the extension after this API
addition. No model configuration switch or environment variable is added.
See [benchmark instructions](../../../benchmarks/README_lim_age.md) for all-hit,
all-miss and eager/graph commands. The existing score operator is unchanged.

Hit clearing now uses temporary Source-block lists in existing UB storage and
read/modify/write DMA for each touched 512-source block. Local scalar clears
replace per-Hit GM writes and cache-line flushes. Increment and Miss updates are
unchanged; rebuild before comparing the existing all-hit diagnostic timings.
No NPU speedup has been verified locally.

For Hit-clear stage diagnostics, mask 8 runs bucket construction only and mask
16 retains bucket construction plus block DMA but skips local clearing. Flags
range from 0 through 1023; bit 1 (skip entire Hit clear) takes precedence, then bit
8 (buckets only). Combine either new mode with bit 2 to disable increment. These
are not valid production age policies. `--ablation hit-clear-breakdown` in the
benchmark runs the four comparable no-increment variants. The V2 ABI argument
list is unchanged; rebuilding is required for flag validation and kernel logic.

Hit grouping now uses vector Source sorting followed by contiguous block runs,
replacing scalar head/next list construction and pointer chasing. Miss-prefix
entries are masked in a local copy; external TopK order and outputs are unchanged.
Existing diagnostic flags retain their meaning, with `buckets` referring to the
new sorting/grouping stage. The shared original LIM kernel directory is declared
as a source/cache dependency so changed headers are staged before age compilation.
NPU correctness and speedup of this update are pending server verification.

Vector run grouping is now the default. Diagnostic mask 32 runs sorting only;
mask 64 selects the previous scalar run builder without disabling age updates.
Other skip bits retain their meaning, with range [0,1023]. The existing V2 argument
list is unchanged. Use benchmark `--ablation hit-group-breakdown` for matched
sort-only, scalar/vector grouping-only and complete-clear comparisons. Masks,
neighbor Gather and compacted metadata reuse existing UB buffers. Compilation,
NPU correctness and performance remain to be verified on server.

Untimed diagnostic mask 128 publishes compaction counts, scalar fallback and
FinishAge progress into the final 16 unused `miss_dst_slots` cells per request;
valid Miss entries are unchanged. Default flags zero perform no probe export.
Mask 512 stops FinishAge after request validation; mask 256 stops after its
count read. Skip-refill (4) takes precedence, followed by 512, then 256. These
masks intentionally omit P assignment. The benchmark provides `finish-breakdown`,
`--diagnostics` and `--profile-dir` to separate checks/count reads, verify fallback
and collect actual NPU kernel timelines. See the benchmark guide for commands and
the output-tail layout. Instrumented runs do not report Event latency.

The default now uses scalar run grouping. Mask 1024 restores vector grouping;
64 forces scalar grouping. Mask 2048 clears consecutive Hit ranges using aligned
vector interiors and scalar edges. Mask 4096 moves resident increment after TopK
and merges it with Hit clear in a full block pass before eviction. The merged
pass also uses range clear, even without 2048. It reuses existing UB scratch and
requires complete increment/Hit-clear stages (no bits 1,2,8,16,32). Mask range is
[0,8191]. Benchmark `age-update-breakdown` compares the three optimization steps
and the former vector path. NPU compilation, correctness and speedup await server
validation.

Experimental flag 8192 enables two-bank prefetch/deferred writeback for merged
updates; it requires bit 4096. The new maximum mask is 16383. Benchmark
`merged-double-buffer` selects 14336 and retains the single-buffer variant as a
control. Banks reuse disjoint regions of sort scratch; independently allocated
MTE3-to-MTE2 events guard bank reuse, and both banks are drained before eviction.
No model config routing/default is changed. CANN/NPU validation is pending.
