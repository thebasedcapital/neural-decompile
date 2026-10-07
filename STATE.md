# State: hackathon-10x (2026-10-07)

## Where we stand

| Deliverable | Status | Evidence |
|---|---|---|
| Linear repair (deployed blend) | **Done, winning** | Later days 0.167012 vs recenter 0.153511 (+0.013502); dev +0.023366; Rust parity 1.21e-17 / 17,233 bins; commits `86b5bf9`, `2acd552` |
| Decompiler verification honesty + transformer emitter | Done | `cargo test` 12+4 pass; nonzero exits, eps threaded, 6 block biases, generated Rust compiles |
| GRU hypertune pipeline | **Built, smoke-tested** | `preprocess_training.py` → 157k bins / 528-dim (172 MB, regenerable in ~19 s, not in git); `train_rnn.py` deterministic sweep; `pod_sweep.sh` launcher; smoke 3×3 epochs on local RTX 4000 |
| H100 sweep | **Blocked** | Runpod 402 balance too low (verified on cheapest pod). Unblocks: user adds funds at console.runpod.io/user/billing → I create H100 SXM ($2.69/hr community) → `pod_sweep.sh` → ~1–2 h → integrate winner |
| Official benchmark intel | Done | FALCON H1 (EvalAI, few-shot): static WF 0.16, NoMAD 0.13, CycleGAN 0.12, NDT2 FSS 0.52, oracle NDT2 0.63. Our 0.167 beats all classical baselines locally |

## Honesty ledger (kept current)

- All later-day numbers = reused-data reevaluation, not a fresh holdout.
- Leaderboard-informed choices: deployed candidate (blend) + 30 s alphas, disclosed via `reused_data_*` keys in `examples/recovery/protocol.json` and `recovery-selection.json` (`later_day_labels_used_for_selection: true`).
- Source-only: multi-tau feature stack, causal β=0.1 smoother, repair family, dev selection.
- Discarded with evidence: empirical-Bayes prior ridge, within-day CV, CORAL.

## Next actions (in order)

1. User funds Runpod → H100 sweep (60 trials × 3 forward folds, ~$5).
2. Integrate winning trial: RNN as base decoder + smoother; decide patch story honestly (RNN weights are not a compact linear patch — the linear blend stays the compact variant).
3. Single later-day evaluation of the winner; update `results/*` + README; keep disclosures.
