#!/usr/bin/env python3
"""Interpret the trained reach-decoder: report held-out accuracy (float vs quantized vs
majority baseline), then run a unit-dropout sensitivity diagnostic on the SAME held-out
set the model was already evaluated on.

This is a diagnostic only: it does not retrain, re-tune eps/hyperparameters, or pick
among candidate models using these numbers. All of that already happened in train.py
using train-only selection; this script just asks "how much does the already-frozen
model depend on each individual input channel" by zeroing one channel at a time and
re-running the frozen, already-quantized weights.

We say "unit dropout", not "electrode death": `unit_id` here is the DANDI units-table
row id for the sorted spike unit, not a verified electrode/channel location on a probe
geometry, so we make no claim about physical electrode failure.

Needs: numpy only (no torch — this just replays the exported float64 weights).
"""
import json
import runpy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "examples" / "bci"


def quantize(p, eps):
    return {n: np.where(np.abs(v - np.round(v)) < eps, np.round(v), v) for n, v in p.items()}


def np_predict(p, x):
    """float64 forward pass, same arithmetic as nd's fsm::run_fsm (first-max-wins argmax)."""
    H = p["b_h"].shape[0]
    preds = []
    for seq in x:
        h = np.zeros(H)
        for xt in seq.astype(np.float64):
            h = np.maximum(p["W_hh"] @ h + p["W_hx"] @ xt + p["b_h"], 0.0)
        preds.append(int(np.argmax(p["W_y"] @ h + p["b_y"])))
    return np.array(preds)


def main():
    weights = json.loads((B / "reach_decoder.json").read_text())
    p = {k: np.array(v, dtype=np.float64) for k, v in weights.items()}
    meta = json.loads((B / "reach_decoder_meta.json").read_text())
    tests = json.loads((B / "reach_decoder_tests.json").read_text())
    eps = meta["eps"]
    q = quantize(p, eps)

    x = np.array([t["inputs"] for t in tests], dtype=np.float64)  # [n_test, T, k]
    y = np.array([t["expected"] for t in tests], dtype=np.int64)
    k = x.shape[2]

    f_pred = np_predict(p, x)
    q_pred = np_predict(q, x)
    emitted = runpy.run_path(str(B / "decompiled" / "reach_decoder.py"))["decompiled"]
    emitted_pred = np.array([emitted(test["inputs"]) for test in tests])
    if not np.array_equal(emitted_pred, q_pred):
        raise ValueError("Emitted Python predictions disagree with the quantized model")
    maj = int(np.bincount(y).argmax())
    maj_pred = np.full_like(y, maj)

    print(f"== Held-out evaluation (n={len(y)}, never touched during model/hyperparameter selection) ==")
    print(f"majority-class baseline accuracy: {(maj_pred == y).mean():.3f} (always predicts class {maj})")
    print(f"float model accuracy:             {(f_pred == y).mean():.3f}")
    print(f"quantized (eps={eps}) accuracy:    {(q_pred == y).mean():.3f}  <- this is what nd verify runs")
    print(f"float vs quantized agreement:      {(f_pred == q_pred).mean():.3f}")
    print(f"emitted Python vs model agreement: {(emitted_pred == q_pred).mean():.3f}")

    print("\n== Unit dropout sensitivity (diagnostic only; computed once, not used to pick a model) ==")
    print("Zeroing one input channel (one selected DANDI spike unit) at a time across every timestep")
    print("of every held-out trial, re-running the frozen QUANTIZED weights (the ones nd actually runs).")
    print("'unit dropout', not 'electrode death': unit_id is a DANDI units-table row id, not a validated")
    print("electrode/channel location on a probe geometry.")
    unit_ids = meta["input_index_to_dandi_unit_id"]
    base_acc = float((q_pred == y).mean())
    rows = []
    for i in range(k):
        xi = x.copy()
        xi[:, :, i] = 0.0
        pi = np_predict(q, xi)
        acc = float((pi == y).mean())
        rows.append(dict(channel=i, dandi_unit_id=int(unit_ids[i]), acc_with_dropout=acc,
                          delta_from_baseline=acc - base_acc,
                          changed_predictions=int((pi != q_pred).sum())))
        print(f"  channel {i:2d} (DANDI unit {unit_ids[i]:5d}): acc {acc:.3f}  delta {acc - base_acc:+.3f}")
    ranked = sorted(rows, key=lambda r: r["delta_from_baseline"])
    worst = ranked[0]
    n_unchanged_accuracy = sum(1 for r in rows if abs(r["delta_from_baseline"]) < 1e-9)
    print(f"\nmost sensitive channel: {worst['channel']} (DANDI unit {worst['dandi_unit_id']}), "
          f"dropping it alone changes held-out accuracy by {worst['delta_from_baseline']:+.3f}")
    print(f"{n_unchanged_accuracy}/{k} channels leave accuracy unchanged when individually zeroed. "
          "Equal accuracy does not establish identical predictions or joint redundancy.")

    out = dict(n_test=len(y), majority_baseline_acc=float((maj_pred == y).mean()),
               float_acc=float((f_pred == y).mean()), quant_acc=base_acc,
               float_quant_agreement=float((f_pred == q_pred).mean()),
               emitted_python_agreement=float((emitted_pred == q_pred).mean()),
               unit_dropout=rows)
    (B / "sensitivity.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
