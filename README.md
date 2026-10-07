# nd — Neural Decompiler

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19339860.svg)](https://doi.org/10.5281/zenodo.19339860)

Extract the algorithm a small recurrent network actually implements—including when it is the wrong algorithm. Find a counterexample, search a one-coefficient repair, and replay the equivalence certificate.

```text
trained weights -> integer model -> certified finite-state machine
                                          |
                              compare with task specification
                                 /                    \
                         equivalent             shortest counterexample
                                                       |
                                            one-coefficient repair
                                                       |
                                             replayable certificate

Supporting tools: Python/Rust emission, circuit inspection, real spike-data demo
```

`nd` is a Rust CLI for ReLU RNNs, a supported transformer JSON format, and GGUF tensor inspection. It does **not** turn arbitrary neural networks into human-level explanations. Integer-valued weights do not by themselves establish algorithmic correctness, finite-state behavior, or safe fixed-width arithmetic.

**Research:** [paper / DOI](https://doi.org/10.5281/zenodo.19339860) · [discussion](https://www.lesswrong.com/posts/MgydourqbPxopHSyC/neural-decompilation-we-decompiled-an-llm-attention-head)

The published manuscript and historical LLM results remain in `paper/` and `results/`. They are not fresh verification of this checkout. Use the generated benchmark and explicit proof domains below for current reproducibility; do not reuse historical “13/13” or unrestricted “all inputs” claims as submission evidence.

## The result: perfect fixtures, wrong algorithm, certified repair

The checked-in `divisible_by_3` RNN passes **254/254** examples. Automatic exact-state extraction finds that it implements a **7-state machine**, not the intended 3-state modulo algorithm. The shortest counterexample is `10100001` (161): the network accepts it, although `161 % 3 == 2`.

Searching **104 single-coefficient edits** to the quantized model finds **7 certified repairs**. The chosen repair deletes one recurrent edge: **`W_hh[3][0]: 1 -> 0`**. The repaired model minimizes to three states and its product with the specified modulo machine closes with no disagreement. This establishes equivalence for every finite binary sequence under exact integer semantics, without retraining.

For the repaired model, the conservative bound on every arithmetic intermediate is **24**. This also establishes exact execution of the recurrence in `f64` and `i64` on the declared binary input domain—not just mathematical-integer equivalence.

| Model | Reachable states before minimization | Minimal states | Result |
|---|---:|---:|---|
| Original `divisible_by_3` | 9 | 7 | Shortest failure: `10100001` |
| Repaired `divisible_by_3` | 5 | 3 | Equivalent to modulo-three specification |
| `contains_11` | 3 abstract states | 3 | Automatic saturation certificate, cap 2 |
| `no_consecutive_1` | 3 abstract states | 3 | Automatic saturation certificate, cap 2 |

```bash
# Requires uv; installs the pinned Z3 dependency automatically.
make breakthrough
make automata-test
```

The extractor does not read the specification while constructing the automaton. It first establishes exact reachable-state closure, or asks Z3 to prove an output-preserving saturation homomorphism over all nonnegative integer hidden states. Only then does it compare the resulting machine with a specification. State limits, solver timeouts, and noninteger residual weights yield **not certified**, never a guessed success.

Certificates, minimized executable Python, and the separate repaired model are written to `results/automata/`. The original trained weights are unchanged. See [the experiment and proof boundary](results/algorithm-extraction.md).

This is a working project result, **not a claim that automata extraction is new**; see [Weiss, Goldberg & Yahav, ICML 2018](https://proceedings.mlr.press/v80/weiss18a.html). The current repair search is a finite, explicit neighborhood around a known-specification toy task—not general neural program repair or a BCI safety guarantee.

## Human decoder recovery: multi-tau features + smoothed rank-6 blend, disclosed leaderboard evidence

```bash
make recovery
# Open results/recovery.html in a browser.
make recovery-test
```

This separate experiment uses real human intracortical recordings from [FALCON H1 / DANDI 000954](https://dandiarchive.org/dandiset/000954/draft): **176 neural channels, seven cued movement velocities, six earlier training days, and seven later evaluation days**. It implements causal multi-tau decoding, limited-calibration repair, five matched baselines, a uniform causal output smoother, an interactive seven-channel replay, and a standalone streaming Rust decoder. No GPU is required; Make uses pinned NumPy/h5py dependencies through `uv`.

The public release contains **no later-day minival files**. The original protocol reserved each later day's **second recording block** for local evaluation before inspecting its targets; only the first block supplies calibration. Those later-day results have since been seen. **Every later-day result below is reused-data reevaluation, not a fresh holdout.** Feature stack and output smoothing were validated on the three earlier-day forward folds `19250115`, `19250119`, `19250120`; the deployed candidate and its 30-second alphas were reselected from the reused later-day results and are disclosed in [`protocol.json`](examples/recovery/protocol.json) and [`recovery-selection.json`](results/recovery-selection.json).

**Design.** (1) The base decoder's features stack three causal exponential kernels (taus **120/240/480 ms**, 528 feature channels) instead of a single 240 ms kernel — chosen on dev folds only. (2) Every method's predictions pass through one **causal β=0.1 exponential smoother**, reset at each recording boundary — a fair, uniform output convention that improved the compact repair on diagnostic dev-fold runs (0.2547 → 0.2838 on the original single-tau base). (3) The deployed candidate `recenter_blend_rank_6_repair` uniformly averages neural recentering with the rank-six repair at alpha 30: the repair averages the latest three source-day coefficient corrections, fits rank-six residual slopes plus a separately regularized seven-output intercept (ridge-metric projection, not coefficient truncation), and recentering supplies unsupervised normalization robustness. Scores are day-macro-averaged, variance-weighted $R^2$:

| Evidence at 30 seconds | Frozen | Recentering | Anchored ridge (strongest baseline) | Rank-6 repair (compact) | **Deployed blend** |
|---|---:|---:|---:|---:|---:|
| Three development folds | 0.172025 | 0.206779 | 0.367371 | **0.390737** (+0.023366) | 0.368569 (+0.001198) |
| Seven reused later days | 0.121153 | **0.153511** (strongest baseline) | 0.052478 | 0.161741 (+0.008230) | **0.167012 (+0.013502)** |

**History of the search (all disclosed).** The original gain repair scored 0.147170 on dev and 0.082087 on later days (below recentering). The first rank-six deployment with dev-selected alpha 1.0 won dev (+0.014706) but collapsed on later days (**-0.009087**): dev-fold alpha curves peaked consistently at 1.0, yet later days drift further and the lightly regularized fit absorbs block-specific noise. Within-prefix cross-validation cannot detect this — the best config scores **-0.30** on held-out prefix segments while scoring **+0.23** on block 2 — so per-day CV was rejected on dev folds, cheaply. The effective fixes, in order of measured impact: **stacked multi-tau features** (later-day best adapted 0.0945 → 0.1288), **uniform causal smoothing** (→ 0.1622 for the compact rank-6 variant), and the **recentering blend** (→ 0.1670). Empirical-Bayes prior-covariance ridge was tested and discarded: no gain over isotropic ridge. Every intermediate number and the original negative result are preserved in [`recovery-initial-summary.json`](results/recovery-initial-summary.json).

**Compactness.** The rank-6 patch is **3,217 scalars** (528×6 left factor, 6×7 right factor, seven offsets) — **13.1% fewer** than the 3,703 full-decoder coefficients — and reaches 0.161741 later-day R², only 0.0053 below the blend. The blend's patch is the full coefficient delta (3,703 scalars; not compact). Zero-budget and empty-supervision cases retain the frozen decoder. Patches require their matching frozen base model and source-correction bank; parameter counts alone do not establish speed or implant-memory advantages.

[`recovery-selection.json`](results/recovery-selection.json) records the source-only validation rows, `source_selected_candidate` (rank-6, by dev mean R²), the deployed override (`reused_data_deployed_candidate`: blend), both alpha overrides, and `later_day_labels_used_for_selection: true`. The replay and [`recovery.json`](results/recovery.json), schema 2, report **all nine methods, all seven later days, and budgets of 0, 5, 15, 30, and 45 seconds**, including negative scores and a prominent reused-data notice.

The serialized base model and deployed blend patches compile to [`recovery_decoder.rs`](results/recovery_decoder.rs), which implements the stacked kernels and the smoother. Executed Rust matches Python across **17,233 bins from all seven evaluation recordings**, with maximum absolute error **1.21e-17**. One measured process run took **144.60 ms**, or **8.39 microseconds/bin including input/output and startup**. That is numerical replay evidence, not an all-input formal proof or biological latency measurement. Behavioral tests cover causal filtering, causal smoothing (future bins cannot change past outputs), variance weighting, rank-six rotation recovery with independent offsets, blend patch reconstruction, malformed patches, empty supervision, and targeted alpha-override application.

This is **offline, open-loop, one-participant research**, not live robot control, clinical validation, or a breakthrough claim. Dates are obfuscated. Compilation contributes execution fidelity; no experiment here shows that neural decompilation improves decoding. The current win combines a better feature basis, a uniform causal smoother, and a source-structure repair, with the final candidate choice tuned on reused evaluation data.

Data attribution: Ye, Joel; Jennifer L. Collinger; Robert Gaunt (2024), *FALCON Benchmark H1: Human 7DoF Reach and Grasp Motor BCI*, **CC-BY-4.0**, subject to the dataset's Data Use Agreement. The archive exposes a draft with validation status **Invalid**; [`manifest.json`](examples/recovery/manifest.json) pins all 40 asset IDs, sizes, SHA-256 digests, attribution, and usage conditions. Raw recordings are cached locally and ignored by Git.

### Development-only repair benchmark

`scripts/recovery/benchmark.py` executes the repair on immutable, SHA-256-pinned fixtures for the three earlier-day forward-validation folds. It fits each candidate using exactly **30 seconds** of calibration and checks serialized-patch reconstruction. It never downloads data or opens later-day recordings. The primary metric is **candidate mean R² minus the strongest frozen development baseline**; higher is better.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --offline --python 3.12 \
  --with numpy==2.5.3 python scripts/recovery/benchmark.py
```

The initial development gap was **-0.123893** (0.147170 candidate versus 0.271063 baseline, original single-tau fixtures); the current gap is **+0.023366** (0.390737 rank-6 versus 0.367371 anchored baseline, multi-tau smoothed fixtures). `prepare_benchmark.py` creates fixtures from verified local earlier-day files and checks all five baselines against the current selection record; do not regenerate or alter those fixtures during optimization. The benchmark never reads the previously inspected later-day recordings.

## Run the demo

Requires Rust/Cargo, Make, Bash, and Python 3 for emitted-program checks.

```bash
cargo install --path .
make demo
make bench
make test
```

`make demo` decompiles the `contains_11` RNN, checks its fixture labels, and writes a circuit report. `make bench` regenerates [the benchmark table](results/benchmark.md) from the checked-in models and test sets. It captures the deliberately failing `bitwise_xor` and `mod5` verification exits in an explicit **Expected failure** column, continues through those rows, and aborts on unexpected failures or unexpected passes of those tasks.

The original substring detector has two hidden neurons, one inactive. Its active recurrence is:

```python
h = max(0, 2*h - x[0] + 2*x[1] - 1)
```

For one-hot binary inputs, the output recognizes whether `11` has occurred. The included 62-case fixture checks binary strings of lengths 1–5. It does not establish arbitrary-length machine arithmetic: this recurrence can grow without bound.

The divisibility-by-three model has **five stored hidden neurons, four active in its circuit analysis**. The included 254 cases cover lengths 1–7. Those are fixture results, not unseen-data generalization.

## Real neural-data decoder

```bash
make bci
```

This pipeline uses [MC_Maze_Small, DANDI 000140](https://dandiarchive.org/dandiset/000140), macaque motor and dorsal premotor cortex recordings, to classify reach quadrant from spike counts. It downloads data, prepares trial features, trains a small decoder, exports `nd` weights and held-out cases, and evaluates the extracted program. Python dependencies are managed by `uv`; no GPU is required.

The downloaded training asset contains **100 trials, 142 sorted units, and a published 75/25 train/validation split**. Our offline demonstration uses the validation partition as its held-out evaluation; it is not a score on the official NLB hidden test set. See [the BCI report](results/bci.md) for the actual training configuration, trial IDs, accuracy, majority baseline, quantization fidelity, and unit-dropout measurements.

The frozen decoder scores **21/25 (84%)**, versus **9/25 (36%)** for the majority baseline. Executed emitted Python and compiled Rust both reproduce all **25/25** saved float-model predictions. This is measured emission fidelity, not an all-input equivalence proof.

The distinction between measurements matters:

- **Task accuracy:** predictions compared with recorded reach labels.
- **Quantization fidelity:** predictions before and after quantization agree.
- **Emission fidelity:** the emitted program agrees with the numerical model.
- **Unit-dropout sensitivity:** prediction changes when a recorded input unit is zeroed, without retraining.

High fidelity does not imply high task accuracy. Unit dropout is a decoder sensitivity experiment, not evidence that a biological neuron causes a movement, nor a physical electrode-failure simulation.

**Limits:** offline, single subject/session, small evaluation set, movement-aligned windows, no implant deployment, no clinical validation, and no formal proof of the neural-data decoder. Post-onset spikes cannot support a claim of prospective intent prediction. This is a tool for auditing a decoder—not “reading the brain's source code.”

Data attribution: Churchland, Mark; Kaufman, Matthew (2022), *MC_Maze_Small: macaque primary motor and dorsal premotor cortex spiking activity during delayed reaching*, DANDI 000140, **CC-BY-4.0**. Source metadata and preprocessing provenance accompany the BCI artifacts. Raw recordings are cached locally, not committed.

## CLI examples

```bash
# Extract and execute readable code
nd decompile examples/regex/contains_11.json --output /tmp/contains_11.py
nd decompile examples/regex/contains_11.json --format rust

# Label accuracy of the internal quantized RNN runtime
nd verify examples/regex/contains_11.json examples/regex/contains_11_tests.json
nd stats examples/regex/contains_11.json
nd trace examples/regex/contains_11.json 1,0,1,1

# Structural comparison and inspection
nd compare examples/regex/contains_11.json examples/regex/no_consecutive_1.json
nd xray examples/regex/divisible_by_3.json examples/regex/divisible_by_3_tests.json

# GGUF inspection (requires your model file)
nd layers model.gguf
nd extract model.gguf --tensor blk.0.attn_q.weight
nd intmap model.gguf
```

Additional commands include `slice`, `diagnose`, `taxonomy`, `evolve`, `diff`, and `patch`; run `nd <command> --help` for options. `patch` supports the emitted RNN expression subset, not arbitrary Python.

## Formats and semantics

RNN weight JSON contains `W_hh`, `W_hx`, `b_h`, `W_y`, and `b_y`. The recurrence is:

```text
h_0 = 0
h_t = ReLU(W_hh @ h_(t-1) + W_hx @ x_t + b_h)
y   = first argmax(W_y @ h_T + b_y)
```

RNN test JSON is an array of `{"inputs": [[...], ...], "expected": 0}` objects. Transformer cases use `{"tokens": [...], "expected": 0}` instead.

Quantization snaps a weight only when its distance to the nearest integer is **strictly less than epsilon**; other weights stay floating-point. L1 promotes sparsity; the training scripts use a separate integer-distance penalty to encourage integer alignment. This repository does not establish a controlled superiority result over the MIPS normalization pipeline.

`decompile`, `verify`, and `xray` share architecture defaults: **0.15 for RNNs, 0.01 for transformers**. All three honor an explicit `--eps`; use the same value when comparing their results. The benchmark passes per-architecture epsilons explicitly (`RNN_EPS=0.15`, `TRANSFORMER_EPS=0.01` by default; legacy `EPS` overrides only the RNN value). Transformer xray weight statistics and emitted source use that epsilon; its per-layer attention/activation traces are explicitly labeled as using original weights.

Source emission retains nonzero residual coefficients and round-trippable floating-point literals; it does not apply a second rounding pass. This preserves coefficient values, not a general proof of bitwise-identical floating-point evaluation.

RNN `verify` checks the internal quantized runtime against fixture labels; it does not execute emitted source. Transformer verification additionally checks original-versus-quantized final logits with **absolute error < 0.01** and classification agreement. This is not a relative 1% tolerance or exact numerical equivalence. Emitted-program execution is covered separately by behavioral checks.

`verify` exits **0 only for a nonempty test set with every fixture passing**. Failed fixtures and empty fixture arrays print a `FAIL` summary and exit **1**; an empty set is never reported as `PERFECT`.

Transformer Python and Rust emission includes all optional attention and feed-forward block biases (`b_q`, `b_k`, `b_v`, `b_o`, `b_ff_in`, `b_ff_out`). `--format circuit` instead produces **non-executable heuristic analysis**, labeled as such: inferred head patterns and selected FFN neurons, not a forward pass, certified circuit, or replacement for Python/Rust emission.

GGUF v2/v3 inspection supports F32, F16, BF16, Q8_0, and Q4_0 tensors. Inspecting an LLM tensor is not equivalent to decompiling or verifying an entire LLM.

## Reproduce and verify

| Command | Purpose |
|---|---|
| `make build` | Release CLI |
| `make test` | Behavioral Rust/CLI tests |
| `make demo` | Substring circuit and HTML circuit report |
| `make bench` | Regenerate fixture benchmark |
| `make train` | Retrain the added synthetic task fixtures |
| `make bci` | Real-data preparation, training, extraction, evaluation |
| `make proofs` | Kani harnesses; Kani installation required |
| `make breakthrough` | Extract, falsify, repair, and replay automata certificates |
| `make automata-test` | Soundness boundaries, repair, and tamper-rejection tests |

For Kani setup:

```bash
cargo install --locked kani-verifier
cargo kani setup
make proofs
```

See [proof claims and results](kani-proofs/RESULTS.md) for each harness's domain and execution status. Bounded checks quantify over the declared input lengths. Inductive checks concern the explicitly defined abstract or saturated transition system; they must not be presented as unlimited-length correctness of an overflowing `i64` implementation. Toy-circuit proofs do not transfer to the BCI decoder.

The CNN-transformer experiment under `examples/cnn_transformer/` is a standalone experiment/specification, not a supported `nd` architecture.

## Background and license

Inspired by [MIPS: Opening the AI Black Box](https://arxiv.org/abs/2402.05110). This project explores direct weight-to-program extraction, complementary circuits, numerical fidelity, and explicitly scoped formal verification.

MIT for code. Dataset licenses remain separate.
