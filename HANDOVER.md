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

**2. K ≤ Q < V on average — but NOT per-layer.**
~~V<Q<K was originally claimed universal — this is WRONG.~~ Per-layer analysis shows K<Q<V is the dominant ordering (K most structured, V least). The cross-layer averages: K=3.317, Q=3.339, V=3.359. Confirmed across both TinyLlama and Llama-2-7B. See `results/cross-model-entropy.md`.

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

**4. Head 15 of layer 0 K is a rank-2 binary gate.**
Entropy 1.824 bits (most structured head). SVD reveals it compresses 2048 embedding dims to just 2 features: (a) syntax/operators vs natural language prose (72% variance), (b) morphologically complex vs analytic languages (23%). Orthogonal to head 21 but reads the same embedding features. Full analysis in `DECOMPILED-HEAD-2.md`.

**5. Full taxonomy of all 32 layer-0 K heads.**
- 11/32 (34%) are rank 1-2 gates (binary routing: newlines, syntax, special tokens)
- 14/32 (44%) are rank 3-4 classifiers (punctuation type, language groups, whitespace)
- 7/32 (22%) are rank 5+ complex (multi-script classification, domain detection)
- Heads 6 & 7 are duplicate rank-1 newline detectors (cosine 0.992) with different output wiring — multi-path routing of the same feature
- Heads 2 & 3 are complementary: H2 detects EOS/BOM, H3 detects CJK (opposite signs)
- Interactive visualization: `results/tinyllama-L0-K-taxonomy.html`

**6. Cross-model comparison with Llama-2 7B.**
Layer 0 Q/K outlier confirmed universal. V is not special. Per-head analysis shows Llama-2 Q head 26 is the most structured (2.381 bits). See `results/cross-model-entropy.md`.

## Future Plan

Three phases. Each phase has a **decision gate** — check the gate before moving on.

### Phase 1: Prove Universality (1-2 sessions)

**Goal:** Determine if the entropy discoveries are a law or a TinyLlama coincidence.

| Step | Task | Command / Method | Done when |
|------|------|-----------------|-----------|
| 1a | Download 3-4 Q4_0 GGUFs | HuggingFace: Qwen2.5-0.5B, Phi-2-2.7B, Gemma-2B, Llama-2-7B | Files in `~/models/` |
| 1b | Run `nd intmap` on each | `nd intmap ~/models/<model>.gguf --deep 5` | Output saved to `results/<model>-intmap.txt` |
| 1c | Compare entropy patterns | Manual or script: extract V/Q/K entropy per layer | Table in `results/cross-model.md` |
| 1d | Check head specialization | Per-head max\|w\| analysis (Python — see Rust-Python gap below) | Per-model head heatmap |

**Decision gate:** Do ≥3/4 models show V<Q<K ordering AND layer-0 K outlier?
- **YES →** This is a structural law. Proceed to Phase 2 (verification) AND Phase 3 (paper) in parallel.
- **PARTIAL →** Document which architectures break it and why. Still publishable as an empirical finding. Proceed to Phase 3.
- **NO →** TinyLlama-specific artifact. Pivot: focus the paper on RNN formal verification (the Kani story). Skip Phase 2.

**Llama 2 7B is already on disk** — start with that. It's the same architecture family, so if even this breaks the pattern, the universality claim is dead early.

### Phase 2: End-to-End Circuit Verification (1 session)

**Goal:** Prove the sparse decompiled circuit is functionally equivalent to the dense original.

| Step | Task | Method | Done when |
|------|------|--------|-----------|
| 2a | Build inference harness | `llama-cpp-python` or `transformers` loading TinyLlama | Can run forward pass, extract attention |
| 2b | Extract dense K head 2 attention | Hook into blk.0 K projection, capture attention patterns on 100 prompts | Baseline attention matrix saved |
| 2c | Build sparse replacement | 23-term k[46] + 12-term k[49] + remaining dims (from DECOMPILED-HEAD.md) | Sparse K projection matrix |
| 2d | Swap and measure | Replace dense K head 2 with sparse, run same 100 prompts | Attention correlation, perplexity delta |

**Decision gate:** Perplexity delta < 0.5% AND attention correlation > 0.95?
- **YES →** Verified decompilation. This is the headline result.
- **NO →** The sparse approximation loses too much. Try: (a) include more terms until it passes, (b) document how many terms are needed for functional equivalence.

### Phase 3: The Paper (2-3 sessions)

**Working title:** "Neural Decompilation: From Weights to Readable Programs"

**Structure:**

| Section | Content | Depends on |
|---------|---------|-----------|
| §1 Introduction | Decompilation framing, contributions list | — |
| §2 Method | L1+quantize, STE 3-phase training, block entropy discovery | — |
| §3 RNN Results | 13/13 tasks, Kani formal proofs, complement discovery | — |
| §4 Scaling | Block entropy scan, V<Q<K ordering, layer-0 outlier | Phase 1 |
| §5 LLM Decompilation | TinyLlama head = multi-script classifier, sparse circuit | — |
| §6 Cross-Model | Universality results (or counterexamples) | Phase 1 |
| §7 Verification | Sparse circuit ≈ dense (perplexity/attention) | Phase 2 |
| §8 Discussion | What this means for interpretability | — |

**Venue decision:**
- Cross-model universality holds + verification passes → **ICML/NeurIPS main conference**
- Partial universality or no verification → **Workshop** (Mechanistic Interpretability @ ICML, or Distill)
- Neither → **arxiv preprint** (still novel: Kani proofs + L1 method + first LLM head decompiled)

### Phase 4: `nd autopsy` — The Product (after paper draft)

**Goal:** Turn manual analysis into a single command.

```bash
nd autopsy model.gguf --tokenizer tokenizer.json
```

| Step | Task | Approach |
|------|------|---------|
| 4a | Add `tokenizers` crate | Rust: `tokenizers = "0.20"` — reads HF tokenizer.json natively |
| 4b | Per-head entropy in Rust | Move Python head decomposition into `intmap.rs` — reshape tensor by n_heads, compute per-head stats |
| 4c | Sparse term extraction | For each low-entropy head: find dims with \|w\| > threshold, emit sparse formula |
| 4d | Token labeling | Decode token IDs through tokenizer, cluster by embedding dim activation sign |
| 4e | HTML report | One page per decompiled head: circuit formula, token clusters, entropy context |

This closes the Rust-Python gap entirely. No Python needed.

### Stretch: MLP Neuron Extraction (speculative, timebox 1 day)

FFN layers are ~3.7 bits entropy (higher than attention) but individual neurons might be sparse.
- Compute each gate neuron's activation on the embedding matrix
- Find max-activating tokens per neuron
- Cluster neurons by activation pattern
- Decompile any that show <2.5 bit entropy

**Kill criterion:** If no FFN neuron has entropy < 3.0 bits after 1 day of analysis, abandon this direction. MLP structure is genuinely dense.

## Source Architecture (8.8K LOC Rust, 20 modules)

| Module | LOC | What |
|--------|-----|------|
| `emit.rs` | 1356 | Code generation: Python/Rust/Rust-Kani/table/circuit for both RNN and transformer |
| `main.rs` | 882 | CLI entry point, 15 subcommands via clap derive |
| `xray.rs` | 841 | Full circuit analysis with HTML reports |
| `gguf.rs` | 732 | GGUF parser, Q4_0/Q8_0/F16/BF16/F32 dequant, nibble extraction |
| `intmap.rs` | 690 | Block entropy, scale-aware grid search, HTML heatmap |
| `trace.rs` | 514 | Hidden state tracing for RNN and transformer |
| `evolve.rs` | 513 | Training evolution visualization across epoch snapshots |
| `diagnose.rs` | 493 | Forensic analysis of quantization failures |
| `slice.rs` | 387 | Dead neuron removal, minimal active subcircuit extraction |
| `transformer.rs` | 350 | Transformer data structures and forward pass |
| `patch.rs` | 347 | Compile decompiled Python back to weight matrices |
| `verify.rs` | 330 | Test case verification for RNN and transformer |
| `compare.rs` | 280 | Complement and shared structure detection between circuits |
| `diff.rs` | 270 | Weight-by-weight semantic delta |
| `quantize.rs` | 265 | Weight quantization (snap to integer grid) |
| `taxonomy.rs` | 235 | Circuit family clustering |
| `visualize.rs` | 157 | HTML rendering for traces |
| `weights.rs` | 121 | Weight loading (JSON → NeuralProgram enum) |
| `fsm.rs` | 38 | FSM state type |

Dependencies: `clap 4`, `ndarray 0.16`, `serde/serde_json`, `anyhow`, `memmap2 0.9`. No Python deps — pure Rust.

## Key Files

| Path | What |
|------|------|
| `DECOMPILED-HEAD.md` | Full writeup of the TinyLlama head decompilation |
| `HANDOVER-KANI.md` | Instructions for Kani formal verification |
| `BREAKTHROUGH_ANALYSIS.md` | Publication gap analysis (scale, relevance, surprise) |
| `autoresearch_scenarios.md` | 6 eval scenarios with rubric (happy path → edge cases) |
| `kani-proofs/` | Formal verification proofs (separate Cargo project) |
| `examples/` | Weight files + test cases for RNN and transformer tasks |
| `scripts/` | Python training scripts (STE, transformer, CNN-transformer) |
| `~/Projects/research/mips-normalizers/` | Training scripts, STE, batch weights |

## Build & Deploy

```bash
cd ~/Projects/neural-decompile
cargo build --release
cp target/release/nd ~/.local/bin/nd
rm -rf target/  # clean up (binary is in ~/.local/bin)
```

## The Rust-Python Gap (Critical)

The **Rust toolchain** handles: entropy scanning, GGUF parsing, tensor extraction, block-level analysis, HTML visualization.

The **Python analysis** (done manually, not baked into `nd`) handles: per-head weight decomposition, sparse term extraction, token-level feature labeling via tokenizer. This is the part that produced the actual "multi-script classifier" interpretation.

To close this gap, `nd autopsy` needs either:
- A Rust tokenizer (tiktoken-rs or tokenizers crate) to label embedding dimensions
- Or a Python sidecar script that `nd` shells out to

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
