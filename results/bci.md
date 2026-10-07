# BCI case study: decompiling a sparse reach-decoder RNN

**Scope**: offline, retrospective classification of reach direction from public macaque
motor-cortex spiking data. No human subjects, no clinical application, no implant, and
no causal claim about the brain — this is an interpretability demo for `nd`'s decompiler,
not a BCI control system.

All numbers below were produced by actually running
`scripts/bci/prepare.py` → `scripts/bci/train.py` → `nd` → `scripts/bci/interpret.py`
in this session on 2026-10-06. Nothing here is fabricated or backfilled from a prior run.

## Data provenance

- **Dataset**: DANDI:000140 "MC_Maze_Small" (Churchland & Kaufman, 2022), CC-BY-4.0,
  Neural Latents Benchmark. Single subject (macaque "Jenkins"), single session ("small"
  maze), primary motor + dorsal premotor cortex, delayed-reaching task.
- **Asset used**: `sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb`,
  asset id `7821971e-c6a4-4568-8773-1bfa205c13f8`, 29,207,528 bytes.
  **sha2-256 `dcaf3a524e2b2f65f163ee3b07789b8474bdfc6ca66098bc542ab93dff489884`** —
  verified to match the locally downloaded file byte-for-byte (`sha256sum`), and
  identical (same asset_id/blob) in both the DANDI `draft` and published version
  `0.220113.0408`. Full record in `examples/bci/data/PROVENANCE.json`.
- A second asset in the dandiset, `..._desc-test_ecephys.nwb` (689,312 bytes), carries
  the dataset's held-out *test* split but **no published behavior/target labels**, so
  it cannot be used for supervised training and `prepare.py` never downloads it.
- **This is one subject, one session.** Nothing here is evidence of cross-day or
  cross-subject generalization.

## Task and local split

- Label: quadrant of the active reach target, `2*(x>0) + (y>0)` → 4 classes
  (left-down / left-up / right-down / right-up).
- `prepare.py` cross-checked this target-derived label against the label computed
  independently from measured hand displacement (onset → stop): **100% agreement**
  (`scripts/bci/prepare.py` stderr: `quadrant label from target == from hand displacement: 100%`),
  confirming the label definition is correct, not an artifact of our binning code.
- Trial split is the dataset's **own published** train/val partition
  (`intervals/trials/split` field in the NWB), not something we chose:
  100 trials total → **75 train / 25 held-out (val)**.
  Held-out class counts `[4, 9, 6, 6]` → majority-class baseline on held-out = **36.0%**
  (uniform chance = 25%).
- 142 candidate units (spike-sorted channels) × 100 bins of 10 ms in a ±500 ms window around
  movement onset are in `examples/bci/data/trials.npz`; only a selected subset of 16
  feeds the model (next section).

## No test leakage — what used which split

| Decision | Data used |
|---|---|
| Unit selection (top-16 by ANOVA F across 4 classes) | **train only** (`select_units` on `D.counts[tr]`) |
| Feature window and binning | Fixed at -100 to +300 ms, 50 ms bins (8 steps), not searched |
| L1/int-pull strength, hidden size, k | **train-only 5-fold stratified CV**, scored on the *quantized* model (the thing `nd` actually runs) |
| Final-seed choice among 8 random inits | **train quantized accuracy + loss only**, zero test peeking |
| Held-out (val) trials | Evaluated after every other choice was frozen; repeated reproduction runs do not select or retune the model |
| Unit-dropout sensitivity (this doc's last section) | Post-hoc diagnosis on the frozen, quantized model; **not used to retrain or pick a model** |

CV grid: `k∈{8,12,16} × H∈{4,8} × l1∈{0.02,0.1} × int_pull∈{0.5,2.0}`, 5 train-only
stratified folds, 1500-epoch full-batch Adam fit per cell (`scripts/bci/train.py`,
seed fixed per job). Selected by train-CV quantized accuracy:
**k=16, H=8, l1=0.02, int_pull=0.5** (cv_quant=0.793, cv_float=0.766).
Final seed chosen by train quantized accuracy/loss among 8 seeds: **seed 2**
(train quant acc 1.000, loss 0.001).

## Held-out results (never touched before this point)

From `nd stats`/`nd verify` on `examples/bci/reach_decoder.json` +
`examples/bci/reach_decoder_tests.json` (25 held-out trials), and cross-checked by the
independent float64 NumPy replay in `scripts/bci/interpret.py`:

| Metric | Value |
|---|---|
| Majority-class baseline (always predict class 1) | **36.0%** |
| Uniform chance (4 classes) | 25.0% |
| Float model accuracy (PyTorch, held-out) | **84.0%** (21/25) |
| Quantized model accuracy, `nd verify` (eps=0.15) | **84.0%** (21/25) |
| Float-vs-quantized prediction agreement | **100.0%** (25/25 identical predictions) |
| Executed emitted Python and compiled Rust accuracy | **84.0% each** (21/25) |
| Emitted Python/Rust agreement with saved float predictions | **100.0% each** (25/25) |
| Weights within 0.01 of an integer after epsilon-0.15 snapping (`nd stats`) | **191/236 (80.9%)**; the CLI rounds this to 81% |

**Beats majority baseline by 48.0 points (84.0% vs 36.0%) on 25 held-out trials it never
saw during unit selection, hyperparameter search, or seed choice.**

eps sweep (`nd verify --eps e`, same 25 held-out trials, same frozen float weights):

| eps | 0.01 | 0.05 | 0.10 | **0.15 (trained)** | 0.25 | 0.35 | 0.50 |
|---|---|---|---|---|---|---|---|
| accuracy | 84% | 84% | 84% | **84%** | 72% | 64% | 24% |

Accuracy is flat at 84% from eps=0.01 through the trained eps=0.15, then degrades —
the quantization point was not a lucky single setting; nearby eps values all agree with
the float model.

**Caveat on tie-breaking**: both `nd`'s `fsm::run_fsm` and the Python/NumPy mirror in
`train.py`/`interpret.py` use strict `argmax` ("first index with the max value wins" on
exact ties), and `quantize_val` snaps only when `|w - round(w)| < eps` (strict
inequality; half-integer weights remain unsnapped when eps is at most 0.5). No tie-break mismatch was observed in the
84%/84%/100%-agreement numbers above.

## Unit-dropout sensitivity (diagnostic, not model selection)

Run once, after the model and eps were already frozen, by zeroing one input channel
(one selected spike unit) at a time across every timestep of the same 25 held-out
trials and re-running the frozen *quantized* weights
(`scripts/bci/interpret.py` → `examples/bci/sensitivity.json`). This is a post-hoc
diagnostic only — no retraining, no model/hyperparameter choice was influenced by it.

We say **"unit dropout"**, not "electrode death": `unit_id` is the DANDI units-table row
id for a spike-sorted unit, not a geometry-validated electrode/channel location on a
probe, so no physical-hardware failure claim is implied.

| channel | DANDI unit id | held-out acc with that channel zeroed | Δ from baseline (0.840) |
|---|---|---|---|
| 0 | 1651 | 0.600 | **−0.240** (most sensitive) |
| 1 | 1201 | 0.720 | −0.120 |
| 2 | 2771 | 0.840 | +0.000 |
| 3 | 1131 | 0.680 | −0.160 |
| 4 | 1141 | 0.840 | +0.000 |
| 5 | 1101 | 0.840 | +0.000 |
| 6 | 1581 | 0.840 | +0.000 |
| 7 | 1092 | 0.800 | −0.040 |
| 8 | 1091 | 0.840 | +0.000 |
| 9 | 1031 | 0.840 | +0.000 |
| 10 | 1191 | 0.840 | +0.000 |
| 11 | 1171 | 0.880 | +0.040 |
| 12 | 1251 | 0.800 | −0.040 |
| 13 | 1371 | 0.840 | +0.000 |
| 14 | 1342 | 0.720 | −0.120 |
| 15 | 2021 | 0.920 | +0.080 |

8 of 16 channels leave held-out accuracy unchanged when zeroed individually. Equal accuracy
does not imply unchanged predictions, individual redundancy, or joint redundancy. Zeroing
DANDI unit 1651 lowers measured accuracy from 84% to 60%; this is an input-perturbation
diagnostic, not an additive attribution of the decoder's margin over baseline.

## Reproduce

```
scripts/bci/run.sh                 # full pipeline: download (idempotent) -> train -> nd decompile/verify/stats/slice/xray -> interpret
```
The pipeline pins Python 3.12, NumPy 2.5.3, h5py 3.16.0, and PyTorch 2.14.1.
`interpret.py` executes the emitted Python decoder and fails if any held-out prediction
differs from the quantized numerical model. Compiled Rust emission was also executed
against all 25 cases in a separate smoke run; it is not compiled by `run.sh`.


This Markdown file records the reported run above. Fresh executions regenerate model,
metadata, prediction, sensitivity, and visualization artifacts; compare those artifacts
before reusing these archived numbers.

Artifacts from this run: `examples/bci/reach_decoder.json` (float weights, `nd`
quantizes at load time — the float model is the thing we kept, not a lossy export),
`examples/bci/reach_decoder_tests.json` (25 held-out test cases), `examples/bci/reach_decoder_meta.json`
(full CV grid, selected hyperparameters, seed, per-trial predictions, unit→DANDI-id
mapping), `examples/bci/sensitivity.json` (per-channel dropout numbers), `examples/bci/data/PROVENANCE.json`
(dataset digest/version/license/split provenance), `examples/bci/decompiled/*`
(decompiled Python/Rust/table/sliced/xray outputs), `results/bci_xray.html`.
