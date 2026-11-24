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
