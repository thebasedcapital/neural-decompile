# State: hackathon-10x (2026-10-07)

## Where we stand

| Deliverable | Status | Evidence |
|---|---|---|
| Human decoder recovery (schema 3) | **Done, source-selected** | Later days 0.1799 (rank-6 + running z-score, pre-registered) vs 0.1737 strongest matched baseline; label-free front-end alone 0.1731 beats every schema-1 result incl. the later-day-tuned blend (0.1670); Rust parity 1.04e-17 / 17,233 bins |
| Dev protocol | Fixed | Cross-recording folds (calibrate recording 1 → score recording 2) replace within-recording folds that over-rewarded supervised fits |
| Pre-registration | Committed before evaluation | `examples/recovery/preregistration-2026-10-07.json` (commit `7198a2c`) |
| Decompiler verification + transformer emitter | Done | `cargo test` 12+4 pass |
| GRU sweep harness | Fixed, smoke-tested only | `scripts/recovery/train_rnn.py`: no longer early-stops on the scored day; beta=0 no longer freezes output; diverged runs no longer crash. Zero-shot folds only; no reported result |

## Honesty ledger

- Later-day recordings were inspected by schema-1 pipelines; every later-day number is reused-data evidence.
- Schema 3 used no later-day labels for selection. Base ridge alpha is pinned at 1.0 (pre-registered); the post-hoc base-alpha search (prefers 0.01 on dev) and its worse later-day run (frozen 0.1707, rank-6 0.1317) are disclosed in README and protocol.json.
- Local chronological-block split on public calibration files: not comparable to official FALCON leaderboard numbers (EvalAI held-out evaluation). Do not cite leaderboard baselines as beaten.

## Next actions

1. Optional: GRU sweep (`scripts/recovery/pod_sweep.sh` or local) — compare zero-shot against the frozen + running z-score linear decoder on the same cross-recording folds before any integration.
2. Optional: official FALCON H1 submission via EvalAI for a leaderboard-comparable number.
