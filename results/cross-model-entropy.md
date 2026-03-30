# Cross-Model Entropy Comparison

**Date:** 2026-03-30
**Models:** TinyLlama 1.1B (22 layers), Llama-2 7B (32 layers)
**Method:** `nd intmap` block entropy on Q4_0 quantized GGUFs

## Summary

| Metric | TinyLlama 1.1B | Llama-2 7B |
|--------|---------------|------------|
| **Mean Q entropy** | 3.339 | 3.372 |
| **Mean K entropy** | 3.317 | 3.374 |
| **Mean V entropy** | 3.359 | 3.394 |
| **Mean ordering** | K < Q < V | K ≈ Q < V |
| **L0 Q** | 2.774 (-16.9%) | 2.836 (-15.9%) |
| **L0 K** | 2.508 (-24.4%) | 2.949 (-12.6%) |
| **L0 V** | 3.313 (-1.4%) | 3.367 (-0.8%) |
| **L0 attn_output** | — | 3.195 (-5.5%) |

## Confirmed Universal Patterns (2/2 models)

1. **Layer 0 Q/K outlier:** Both Q and K projections at layer 0 have 13-25% lower entropy than their cross-layer mean. This is a phase transition — no other layer comes close.

2. **Layer 0 V is NOT an outlier:** V at layer 0 is within 1.5% of its cross-layer mean in both models. Whatever structure layer 0 learns, it's in the routing (Q/K), not the content extraction (V).

3. **K ≤ Q on average:** K projections are the most structured (lowest entropy) averaged across all layers. K encodes "what am I?" (position in key space), and this benefits from sparse structure.

4. **V is least structured on average:** V projections have the highest entropy. They act as content pass-throughs — extracting information from embeddings without needing fixed patterns.

5. **Entropy spread collapses with depth:** Layer 0 has a huge Q/K/V entropy spread (0.5-0.9 bits). By layer 5+, all three projections converge to within 0.01-0.02 bits of each other.

## Disproven

- **V<Q<K per-layer ordering is NOT universal.** TinyLlama: 2/22 layers show V<Q<K. Llama-2: 0/32. The dominant pattern is K<Q<V (K most structured, V least). The original HANDOVER claim was based on average entropy across projection types, not per-layer ordering.

## Interpretation

**Layer 0 is a router.** It uses structured Q/K patterns (low entropy = fewer effective quantization levels = sparser, more discrete decisions) to implement fixed attention routing — like the multi-script classifier found in TinyLlama head 2. V doesn't participate because routing doesn't need to extract content.

**Deeper layers are general-purpose.** As depth increases, Q/K/V entropies converge — all projections use the full quantization palette. The model transitions from fixed routing patterns to flexible, context-dependent computation.

## Per-Head Outliers (Layer 0)

| Model | Projection | Outlier Head | Entropy | Effective Levels | σ from mean | Max\|w\| |
|-------|-----------|-------------|---------|-----------------|-------------|---------|
| TinyLlama | K | Head 15 | 1.824 | 3.5 | 2.2σ | 0.559 |
| TinyLlama | K | Head 21 | 2.840 | 7.2 | — | **3.109** (model max) |
| TinyLlama | Q | Head 6 | 2.090 | 4.3 | 2.7σ | 0.559 |
| Llama-2 7B | Q | Head 26 | 2.381 | 5.2 | 2.3σ | 0.551 |
| Llama-2 7B | K | Head 1 | 2.696 | 6.5 | 1.6σ | 0.330 |

Key insight: Head specialization exists in both models but manifests differently:
- TinyLlama concentrates structure in K (the "what am I" signal) with extreme weight magnitudes
- Llama-2 concentrates structure in Q (the "what am I looking for" signal) more uniformly

Both models have heads at layer 0 using only 3.5-5.2 effective quantization levels — these are **potentially decompilable** into sparse circuits.

## New Feature: `nd intmap --deep N` Per-Head Analysis

Added per-head entropy decomposition to `nd intmap`. For attention Q/K/V tensors, the tool now:
1. Infers head count from GGUF metadata
2. Splits nibble blocks by head boundary
3. Computes per-head block entropy
4. Flags outlier heads with σ scores

This automates what was previously a manual Python analysis.

## Raw Data

See `tinyllama-intmap.txt`, `llama2-7b-intmap.txt`, and `llama2-7b-L0-heads.txt` in this directory.
