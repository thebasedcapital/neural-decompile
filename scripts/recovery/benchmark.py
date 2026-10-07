#!/usr/bin/env python3
"""Deterministic development-only compact-repair benchmark; no downloads or holdout reads."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np

import model

FIXTURES = Path(__file__).resolve().parents[2] / "examples/recovery/benchmark"


def r2(truth, prediction):
    residual = np.sum((truth - prediction) ** 2, axis=0)
    variance = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    varying = variance > 0
    if not varying.any():
        raise ValueError("Development evaluation has no target variance")
    return float(1 - residual[varying].sum() / variance[varying].sum())


def main():
    reference = json.loads((FIXTURES / "reference.json").read_text())
    if reference["schema_version"] != 3 or reference["budget_seconds"] != 30:
        raise ValueError("Unexpected frozen benchmark protocol")
    beta = reference["smoothing_beta"]
    folds = []
    for fold in reference["folds"]:
        path = FIXTURES / fold["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != fold["sha256"]:
            raise ValueError(f"Benchmark fixture changed: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for array in arrays.values():
            array.setflags(write=False)
        folds.append((fold["day"], arrays))
    baseline = max(reference["baselines"], key=lambda row: row["mean_r2"])
    methods = sorted(m["id"] for m in model.METHODS if m["id"] not in reference["baseline_methods"])
    if not methods:
        raise ValueError("No candidate repair methods registered")
    candidates = []
    for method in methods:
        for alpha in reference["alphas"]:
            scores, parameters = [], []
            for day, data in folds:
                decoder = model.Decoder(**{field.name: data[field.name] for field in fields(model.Decoder)})
                np.random.seed(0)
                weights, count, _, patch = model.adapt(
                    decoder, data["calibration_x"], data["calibration_y"],
                    data["calibration_all_x"], method, alpha,
                )
                if weights.shape != data["weights"].shape or not np.isfinite(weights).all():
                    raise ValueError(f"Invalid {method} decoder on {day}")
                if count != len(patch) or count < 0 or not np.isfinite(patch).all():
                    raise ValueError(f"Invalid {method} patch parameter accounting")
                restored = model.apply_patch(decoder, method, patch)
                if not np.allclose(restored, weights, rtol=1e-12, atol=1e-12):
                    raise ValueError(f"Serialized {method} patch does not reproduce fitted weights")
                # Fixed scoring, normalization, and smoothing are outside the
                # mutable candidate code; the smoother resets per block.
                block_truths, block_predictions = [], []
                for key in sorted(k for k in data if k.startswith("evaluation_x_")):
                    index = key.rsplit("_", 1)[1]
                    features = data[f"evaluation_x_{index}"]
                    mask = data[f"evaluation_mask_{index}"].astype(bool)
                    design = np.column_stack(((features - data["mean"]) / data["scale"],
                                              np.ones(len(features))))
                    prediction = (design @ weights) * data["target_scale"] + data["target_mean"]
                    if beta < 1:
                        prediction = model.smooth(prediction, beta)
                    if not np.isfinite(prediction).all():
                        raise ValueError(f"Non-finite {method} predictions")
                    block_truths.append(data[f"evaluation_y_{index}"][mask])
                    block_predictions.append(prediction[mask])
                scores.append({"day": day, "r2": r2(np.concatenate(block_truths),
                                                    np.concatenate(block_predictions))})
                parameters.append(count)
            candidates.append({"method": method, "alpha": alpha,
                               "mean_r2": float(np.mean([s["r2"] for s in scores])),
                               "day_scores": scores, "max_patch_parameters": max(parameters)})
    best = max(candidates, key=lambda candidate: candidate["mean_r2"])
    baseline_days = {s["day"]: s["r2"] for s in baseline["day_scores"]}
    day_gaps = [s["r2"] - baseline_days[s["day"]] for s in best["day_scores"]]
    print(json.dumps({"scope": reference["scope"], "calibration_seconds": 30,
                      "candidate": best, "frozen_strongest_baseline": baseline,
                      "candidate_configurations": len(candidates)}, sort_keys=True, allow_nan=False))
    print(f"METRIC development_advantage_r2={best['mean_r2'] - baseline['mean_r2']:.12f}")
    print(f"METRIC candidate_r2={best['mean_r2']:.12f}")
    print(f"METRIC baseline_r2={baseline['mean_r2']:.12f}")
    print(f"METRIC worst_day_advantage_r2={min(day_gaps):.12f}")
    print(f"METRIC patch_parameters={best['max_patch_parameters']}")
    print(f"ASI selected_method={best['method']}")
    print(f"ASI selected_alpha={best['alpha']}")


if __name__ == "__main__":
    main()
