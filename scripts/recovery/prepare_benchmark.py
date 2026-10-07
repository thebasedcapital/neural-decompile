#!/usr/bin/env python3
"""Build fixed, source-only repair fixtures from already verified local recordings.

This preparation command never downloads data and never opens held-out-calib.
Run it before autoresearch, not within an optimization iteration.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from data import load_sessions
from experiment import calibration_prefix, group_days, scored_rows
from model import adapt, fit_base, score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples/recovery/benchmark"


def main():
    protocol_path = ROOT / "examples/recovery/protocol.json"
    selection_path = ROOT / "results/recovery-selection.json"
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads(selection_path.read_text())
    budget = protocol["primary_budget_seconds"]
    if budget != 30:
        raise ValueError("The primary calibration budget must remain 30 seconds")
    sessions = load_sessions(("held-in-calib", "held-in-minival"))
    training = group_days([s for s in sessions if s.split == "held-in-calib"], protocol)
    validation = group_days([s for s in sessions if s.split == "held-in-minival"], protocol)
    days = sorted(training)
    fold_days = days[3:]
    if fold_days != selection["fold_days"]:
        raise ValueError("Development folds differ from the original experiment")
    baseline_rows = []
    for method in protocol["baseline_methods"]:
        rows = [r for r in selection["validation"] if r["method"] == method and r["budget_seconds"] == budget]
        baseline_rows.append(max(rows, key=lambda r: r["mean_r2"]))
    OUT.mkdir(parents=True, exist_ok=True)
    folds = []
    for index, day in enumerate(days[3:], start=3):
        base = fit_base({d: scored_rows(training[d]) for d in days[:index]}, selection["base_alpha"])
        cx, cy, all_x = calibration_prefix(training[day], budget, protocol["bin_ms"])
        vx, vy = scored_rows(validation[day])
        for reference in baseline_rows:
            weights, _, _, _ = adapt(base, cx, cy, all_x, reference["method"], reference["alpha"])
            actual = score(vy, base.predict(vx, weights))[0]
            expected = next(r["r2"] for r in reference["day_scores"] if r["day"] == day)
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Frozen {reference['method']} baseline changed for {day}: {actual} != {expected}")
        path = OUT / f"{day}.npz"
        np.savez_compressed(path, **asdict(base), calibration_x=cx, calibration_y=cy,
                            calibration_all_x=all_x, evaluation_x=vx, evaluation_y=vy)
        folds.append({"day": day, "file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "training_days": days[:index], "calibration_bins": len(all_x),
                      "scored_calibration_bins": len(cx), "evaluation_bins": len(vx),
                      "calibration_files": [s.key for s, _ in training[day]],
                      "evaluation_files": [s.key for s, _ in validation[day]]})
    reference = {
        "schema_version": 1, "budget_seconds": budget,
        "base_alpha": selection["base_alpha"], "alphas": protocol["adaptation_alphas"],
        "baseline_methods": protocol["baseline_methods"], "baselines": baseline_rows,
        "folds": folds, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "original_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "scope": "Development-only optimization on three earlier-day forward folds. Previously inspected later-day recordings are excluded; this is not a new holdout claim.",
    }
    (OUT / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"folds": fold_days, "budget_seconds": budget,
                      "strongest_baseline_r2": max(r["mean_r2"] for r in baseline_rows),
                      "later_day_files_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
