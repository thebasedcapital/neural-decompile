# Handover: Neural Decompiler — Where We Are and Where to Go

## What This Project Is

`nd` is a Rust CLI that decompiles neural network weights into readable code. It started with small RNNs (parity checkers, mod-3 adders) where we achieved 13/13 perfect decompilations and 6 Kani formal proofs. Then we added transformer support and GGUF loading.

On 2026-03-30 we built `nd intmap` and used it to produce what we believe is the **first full decompilation of an attention head from a production LLM** — TinyLlama 1.1B layer 0 KV head 2's K projection is a multi-script language classifier, 99.5% sparse, with the model's largest weight (-3.109) concentrated on a Unicode feature detector.

## Current State

### The Toolchain (15 commands)

```
nd decompile    — weights → Python/Rust/table code
nd verify       — check decompiled FSM against test cases
nd stats        — weight statistics
nd trace        — watch hidden states evolve on an input
nd diagnose     — forensic failure analysis
nd compare      — detect complement circuits
nd taxonomy     — cluster circuit families
nd slice        — isolate minimal active subcircuit
nd xray         — full analysis (stats+slice+decomposition+traces)
nd evolve       — animated training evolution across snapshots
nd diff         — semantic weight delta between circuits
nd patch        — compile Python code back to weight matrices
nd intmap       — scan GGUF for integer structure and block entropy
nd layers       — list GGUF tensors
nd extract      — pull a single tensor as JSON
```

Binary at `~/.local/bin/nd`. Source at `~/Projects/neural-decompile/`. Research (training scripts, experiments) at `~/Projects/research/mips-normalizers/`.

### RNN Results — Complete

- 13/13 algorithmic tasks decompiled perfectly
- STE training (3-phase quantization-aware) cracks every task
- L1 + direct quantization beats the MIPS 5-normalizer chain (Michaud & Tegmark 2024) — normalizers destroy integer alignment
- 6 Kani formal proofs: correctness proven for ALL inputs, not just test cases
- Complement discovery: contains_11 and no_consecutive_1 are structural complements

### Transformer Results — In Progress

- Full transformer decompile/verify/trace/slice/xray pipeline works
- Parity_transformer verified at 100%
- GGUF loading with Q4_0/Q8_0/F16/BF16/F32 dequantization

### TinyLlama intmap Discovery (2026-03-30)

This is the fresh work. Key findings:

**1. Layer 0 K projection is a 7.3σ entropy outlier.**
Block entropy 2.508 bits vs 4.17 mean for layers 1-21. This isn't gradual — it's a phase transition between layer 0 and everything else.

**2. V < Q < K entropy ordering is universal across all 22 layers.**
V projections average 2.54 bits (most structured), Q = 3.30, K = 4.10. V carries output content and naturally learns sparser representations. This might be a structural invariant of trained transformers.

**3. Head 2 of layer 0 K is a multi-script classifier.**
The circuit reads ~7 specific embedding dimensions (out of 2048) to classify tokens by script type:

| Embedding dim | What it detects |
|--------------|-----------------|
| 411 | Unicode/CJK vs code/LaTeX |
| 802 | Unicode/Arabic vs markup |
| 1454 | CJK/katakana vs LaTeX citations |
| 1447 | Code syntax vs CJK punctuation |
| 95 | Proper names vs math terms |
| 687 | Java code vs CJK |
| 297 | Code operators vs German/Slavic |

The two dominant output dimensions:
```
k[46] = -3.109*d411 + 1.180*d1454 - 0.848*d95 + 0.750*d802 - 0.750*d687 + ...  (23 terms, 70% energy)
k[49] = -2.453*d411 + 0.617*d1454 + 0.617*d1447 + 0.543*d802 + ...  (12 terms, 67% energy)
```

Full analysis in `DECOMPILED-HEAD.md`.

## What to Do Next (Prioritized)

### 1. Cross-model comparison (do this first)

**The question:** Is the layer-0 outlier and V<Q<K ordering universal, or specific to TinyLlama?

**How:** Download Q4_0 GGUFs for 3-4 models with different architectures:
- Qwen 2.5 0.5B (different tokenizer, different training data)
- Phi-2 2.7B (Microsoft, heavily synthetic data)
- Gemma 2B (Google, different attention variant)
- Llama 2 7B (same family as TinyLlama but 7x larger)

Run `nd intmap` on each. Compare:
- Does layer 0 always have the lowest K entropy?
- Does V<Q<K always hold?
- Do the same embedding dimensions (411, 802, etc.) appear in other models?
- Does head specialization (one head much larger than others) repeat?

If the pattern is universal, it's a structural law worth publishing. If some models break it, the exceptions tell you something about architecture design.

**Effort:** Low — just download models and run the tool. Maybe half a day.

### 2. End-to-end circuit verification

**The question:** Does the sparse decompiled circuit actually produce the same attention patterns as the dense original?

**How:**
1. Load TinyLlama in Python (llama.cpp or transformers)
2. Extract blk.0 K projection for head 2
3. Replace it with the sparse version (23 terms for k[46], 12 for k[49], etc.)
4. Run inference on a test set
5. Measure: attention pattern correlation, perplexity delta, downstream task accuracy

If perplexity moves less than 0.1% — the decompilation is verified. That makes it publishable.

**Effort:** Medium — need an inference harness. Could use llama.cpp Python bindings.

### 3. `nd autopsy` command

**The product:** A single command that does everything we did manually:

```bash
nd autopsy model.gguf --tokenizer <path>
```

Output: a directory of decompiled circuits for every low-entropy head, with:
- Sparse weight decomposition
- Token-level feature labels for each embedding dimension
- Functional interpretation (script classifier, position encoder, etc.)
- HTML visualization

This turns our one-off analysis into a reusable tool anyone can run.

**Effort:** Medium-high — the sparse decomposition is Python (needs tokenizer), the entropy scan is Rust. Either bridge them or rewrite the analysis in Rust.

### 4. MLP circuit extraction (speculative)

FFN layers have most of the parameters. The intmap already shows they're higher entropy (~3.7 bits) than attention, but individual neurons might still be sparse. The approach:
- For each FFN gate neuron, compute its activation on the embedding matrix
- Find which tokens maximally activate each neuron
- Cluster neurons by activation pattern
- Decompile the sparse ones

Risk: MLP structure might be genuinely dense. Don't spend more than a day on this before deciding.

### 5. The paper

Working title: "Neural Decompilation: From Weights to Readable Programs"

Story arc:
1. RNN decompilation: 13/13 tasks, STE training, L1 beats MIPS normalizers
2. Formal verification: Kani proofs for all possible inputs
3. Scaling to real models: block entropy discovers layer-0 outlier
4. First LLM head decompilation: sparse circuit with semantic labels
5. Cross-model universality of V<Q<K ordering (needs step 1 above)
6. End-to-end verification (needs step 2 above)

Venue: ICML or NeurIPS 2026 workshop, or main conference if cross-model results are strong.

## Files That Matter

| Path | What |
|------|------|
| `src/intmap.rs` | Block entropy + scale-aware grid analysis (690 LOC) |
| `src/gguf.rs` | GGUF parser with Q4_0 nibble extraction |
| `src/main.rs` | CLI entry point, 15 commands |
| `DECOMPILED-HEAD.md` | Full writeup of the TinyLlama head decompilation |
| `HANDOVER-KANI.md` | Instructions for Kani formal verification |
| `tests/` | Parity transformer test cases, RNN weight files |
| `~/Projects/research/mips-normalizers/` | Training scripts, STE, batch weights |

## Models on Disk

- `~/models/tinyllama-1.1b-q4_0.gguf` — the model we analyzed
- `~/models/llama-2-7b-q4_0.gguf` — ready for cross-model comparison

## Things That Don't Work / Known Issues

- `nd intmap --html` generates the table view but the per-head entropy heatmap was done in Python (not yet integrated into the Rust HTML renderer)
- Transformer taxonomy not implemented (prints "not yet implemented")
- Transformer diff not implemented (falls through to "use compare instead")
- The Q4_0 dequantization produces slightly different values than llama.cpp due to f16 intermediate precision — this doesn't affect entropy analysis but matters for exact verification
- `nd intmap` with `--min-pct` filtering doesn't work well because the raw integer percentage is misleading for quantized models (everything rounds to 0). Use entropy instead.
- The per-head analysis that produced the best results was done in Python scripts, not yet baked into `nd`. The Rust side does tensor-level entropy; the Python side does head-level decomposition and token feature labeling.

## Vibes

This project works best when you follow curiosity. The normalizer chain was "the right approach" and failed. L1+quantize was the lazy shortcut and it worked. The intmap entropy approach emerged from a failed attempt at scale-aware grid detection. The script classifier discovery came from "let me just look at what tokens these embedding dims respond to."

Don't start with the roadmap. Start with whatever makes you think "oh that would be cool." Run it on real data immediately. The output will tell you what to build next.
