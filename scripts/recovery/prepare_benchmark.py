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
from experiment import (calibration_prefix, front_end, group_days, pooled_statistics, scored_rows,
                        smoothed_scores, zscore_training)
from model import adapt, fit_base

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
    training = group_days(load_sessions(("held-in-calib",)), protocol)
    normalized = zscore_training(training, protocol)
    days = sorted(training)
    fold_days = days[3:]
    if fold_days != selection["fold_days"]:
        raise ValueError("Development folds differ from the original experiment")
    baseline_rows = []
    for method in protocol["baseline_methods"]:
        alpha = selection["adaptation_alphas"][method][str(budget)]
        baseline_rows.append(next(r for r in selection["validation"] if r["method"] == method
                                  and r["budget_seconds"] == budget and r["alpha"] == alpha))
    OUT.mkdir(parents=True, exist_ok=True)
    beta = protocol["output_smoothing"]["beta"]
    folds = []
    for index, day in enumerate(fold_days, start=3):
        base = fit_base({d: scored_rows(normalized[d]) for d in days[:index]}, selection["base_alpha"])
        fallback = pooled_statistics({d: training[d] for d in days[:index]})
        calibration, blocks, _ = front_end(training[day][:1], training[day][1:], budget, fallback, protocol)
        cx, cy, all_x = calibration_prefix(calibration, budget, protocol["bin_ms"])
        for reference in baseline_rows:
            weights, _, _, _ = adapt(base, cx, cy, all_x, reference["method"], reference["alpha"])
            actual = smoothed_scores(base, blocks, weights, beta)[0]
            expected = next(r["r2"] for r in reference["day_scores"] if r["day"] == day)
            if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Frozen {reference['method']} baseline changed for {day}: {actual} != {expected}")
        evaluation_blocks = {}
        for block_index, (s, features) in enumerate(blocks):
            evaluation_blocks[f"evaluation_x_{block_index}"] = features
            evaluation_blocks[f"evaluation_mask_{block_index}"] = s.mask
            evaluation_blocks[f"evaluation_y_{block_index}"] = s.targets
        path = OUT / f"{day}.npz"
        np.savez_compressed(path, **asdict(base), calibration_x=cx, calibration_y=cy,
                            calibration_all_x=all_x, **evaluation_blocks)
        folds.append({"day": day, "file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "training_days": days[:index], "calibration_bins": len(all_x),
                      "scored_calibration_bins": len(cx),
                      "evaluation_bins": sum(int(s.mask.sum()) for s, _ in blocks),
                      "calibration_files": [s.key for s, _ in calibration],
                      "evaluation_files": [s.key for s, _ in blocks]})
    reference = {
        "schema_version": 3, "budget_seconds": budget,
        "base_alpha": selection["base_alpha"], "alphas": protocol["adaptation_alphas"],
        "smoothing_beta": beta, "front_end": protocol["front_end"],
        "baseline_methods": protocol["baseline_methods"], "baselines": baseline_rows,
        "folds": folds, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "original_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "scope": "Development-only optimization on three cross-recording held-in folds (calibrate on recording 1, evaluate on recording 2); features are stored after the label-free running z-score. Later-day recordings are never opened.",
    }
    (OUT / "reference.json").write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"folds": fold_days, "budget_seconds": budget,
                      "strongest_baseline_r2": max(r["mean_r2"] for r in baseline_rows),
                      "later_day_files_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
