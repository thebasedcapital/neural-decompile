#!/usr/bin/env python3
"""Select only on earlier days, freeze the selection, then evaluate later days.

Development folds cross a recording boundary exactly like the later-day split:
calibrate on the first recording of a held-in day, evaluate on its second.
Every method sees features through the same label-free causal running z-score.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from data import load_sessions, prepare
from model import (METHODS, adapt, fit_base, prefix_statistics, running_zscore, score,
                   smooth, stacked_features)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "examples/recovery/protocol.json"
PREREGISTRATION = ROOT / "examples/recovery/preregistration-2026-10-07.json"
RESULTS = ROOT / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, payload, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=None if compact else 2,
                               separators=(",", ":") if compact else None,
                               allow_nan=False) + "\n")


def group_days(sessions, protocol):
    groups = defaultdict(list)
    taus = protocol["filter"]["taus_ms"]
    for session in sessions:
        features = stacked_features(session.counts, protocol["bin_ms"], taus)
        groups[session.day].append((session, features))
    return dict(sorted(groups.items()))


def zscore_blocks(blocks, statistics, protocol):
    """Apply the shared front-end to each recording, restarting from `statistics`."""
    tau, bin_ms = protocol["front_end"]["tau_seconds"], protocol["bin_ms"]
    return [(s, running_zscore(features, *statistics, tau, bin_ms)) for s, features in blocks]


def zscore_training(days, protocol):
    """Source recordings start from their own label-free opening statistics."""
    bins = round(protocol["front_end"]["source_init_seconds"] * 1000 / protocol["bin_ms"])
    return {day: [block for s, features in blocks
                  for block in zscore_blocks([(s, features)], prefix_statistics(features[:bins]), protocol)]
            for day, blocks in days.items()}


def pooled_statistics(days):
    """Zero-budget initialization: source-recording statistics, no new-day data."""
    return prefix_statistics(np.concatenate([features for blocks in days.values() for _, features in blocks]))


def front_end(calibration_raw, evaluation_raw, budget, fallback, protocol):
    """Initialize from the budget's calibration prefix (all bins, no labels)."""
    if budget:
        statistics = prefix_statistics(calibration_prefix(calibration_raw, budget, protocol["bin_ms"])[2])
    else:
        statistics = fallback
    return (zscore_blocks(calibration_raw, statistics, protocol),
            zscore_blocks(evaluation_raw, statistics, protocol), statistics)


def smoothed_scores(decoder, blocks, weights, beta):
    """Score with the uniform causal prediction smoother, reset per block."""
    truths, predictions = [], []
    for s, features in blocks:
        p = decoder.predict(features, weights)
        if beta < 1:
            p = smooth(p, beta)
        truths.append(s.targets[s.mask])
        predictions.append(p[s.mask])
    return score(np.concatenate(truths), np.concatenate(predictions))


def scored_rows(blocks):
    return (np.concatenate([features[s.mask] for s, features in blocks]),
            np.concatenate([s.targets[s.mask] for s, _ in blocks]))


def calibration_prefix(blocks, seconds, bin_ms):
    remaining = round(seconds * 1000 / bin_ms)
    xs, ys, all_x = [], [], []
    for s, features in blocks:
        take = min(remaining, len(features))
        mask = s.mask[:take]
        xs.append(features[:take][mask])
        ys.append(s.targets[:take][mask])
        all_x.append(features[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"Requested {seconds}s calibration but {blocks[0][0].day} has too few bins")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(all_x)


def choose(protocol, training):
    """Cross-recording forward folds on held-in days only."""
    days = sorted(training)
    if len(days) < 4:
        raise ValueError("Forward-day selection requires at least four held-in days")
    normalized = zscore_training(training, protocol)
    budgets = protocol["calibration_budgets_seconds"]
    primary = protocol["primary_budget_seconds"]
    folds = []
    for index in range(3, len(days)):
        day = days[index]
        if len(training[day]) < 2:
            raise ValueError(f"Cross-recording fold {day} needs two recordings")
        source = {d: training[d] for d in days[:index]}
        fallback = pooled_statistics(source)
        views = {budget: front_end(training[day][:1], training[day][1:], budget, fallback, protocol)
                 for budget in budgets}
        folds.append((day, {d: scored_rows(normalized[d]) for d in days[:index]}, views))
    beta = protocol["output_smoothing"]["beta"]
    base_results, base_cache = [], {}
    for alpha in protocol["base_ridge_alphas"]:
        scores = []
        for day, train, views in folds:
            decoder = fit_base(train, alpha)
            base_cache[alpha, day] = decoder
            scores.append(smoothed_scores(decoder, views[primary][1], decoder.weights, beta)[0])
        base_results.append({"alpha": alpha, "mean_r2": float(np.mean(scores)), "day_scores": scores})
    # Fixed by the pre-registration, not selected here: the fold table above is
    # reported as a diagnostic (see protocol.json "base_ridge_alpha").
    base_alpha = protocol["base_ridge_alpha"]
    print(f"SOURCE base alpha={base_alpha} (pre-registered); fold diagnostics: "
          + " ".join(f"{r['alpha']:g}:{r['mean_r2']:.4f}" for r in base_results), flush=True)
    choices, validation_rows = {}, []
    for method in [m["id"] for m in METHODS]:
        choices[method] = {}
        for budget in budgets:
            alphas = [0.] if method in ("frozen", "recenter") or budget == 0 else protocol["adaptation_alphas"]
            candidates = []
            for alpha in alphas:
                scores = []
                for day, _, views in folds:
                    decoder = base_cache[base_alpha, day]
                    calibration, evaluation, _ = views[budget]
                    cx, cy, all_x = calibration_prefix(calibration, budget, protocol["bin_ms"])
                    weights, _, _, _ = adapt(decoder, cx, cy, all_x, method, alpha)
                    scores.append({"day": day, "r2": smoothed_scores(decoder, evaluation, weights, beta)[0]})
                candidates.append({"method": method, "budget_seconds": budget, "alpha": alpha,
                                   "mean_r2": float(np.mean([s["r2"] for s in scores])), "day_scores": scores})
            best = max(candidates, key=lambda row: row["mean_r2"])
            choices[method][str(budget)] = best["alpha"]
            validation_rows.extend(candidates)
            print(f"SOURCE {method:14} {budget:2}s alpha={best['alpha']:7g} R2={best['mean_r2']:.4f}", flush=True)
    best_rows = [r for r in validation_rows if r["budget_seconds"] == primary
                 and r["alpha"] == choices[r["method"]][str(primary)]]
    source_best = max((r for r in best_rows if r["method"] in protocol["candidate_methods"]), key=lambda r: r["mean_r2"])
    baseline = max((r for r in best_rows if r["method"] in protocol["baseline_methods"]), key=lambda r: r["mean_r2"])
    return {"base_alpha": base_alpha, "adaptation_alphas": choices,
            "later_day_labels_used_for_selection": False,
            "method": protocol["primary_candidate"], "baseline": baseline["method"],
            "source_best_candidate": source_best["method"],
            "budget_seconds": primary, "base_validation": base_results,
            "validation": validation_rows, "fold_days": [f[0] for f in folds],
            "fold_design": "cross-recording: calibrate on recording 1, evaluate on recording 2 of the same held-in day",
            "protocol_sha256": digest(PROTOCOL),
            "preregistration_sha256": digest(PREREGISTRATION),
            "model_sha256": digest(Path(__file__).with_name("model.py")),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "later_day_evaluation_previously_observed": True}


def display_array(array):
    return [[round(float(v), 6) if np.isfinite(v) else None for v in row] for row in array]


def evaluate(protocol, selection, decoder, calibration, evaluation, fallback):
    sessions, compiled_models, compiled_cases = [], [], []
    budgets = protocol["calibration_budgets_seconds"]
    step = 5
    for day, raw_blocks in evaluation.items():
        day_scores, predictions = [], {}
        views = {budget: front_end(calibration[day], raw_blocks, budget, fallback, protocol) for budget in budgets}
        prefixes = {budget: calibration_prefix(views[budget][0], budget, protocol["bin_ms"]) for budget in budgets}
        beta = protocol["output_smoothing"]["beta"]
        for method in [m["id"] for m in METHODS]:
            predictions[method] = {}
            for budget in budgets:
                blocks = views[budget][1]
                alpha = selection["adaptation_alphas"][method][str(budget)]
                weights, count, milliseconds, patch = adapt(decoder, *prefixes[budget], method, alpha)
                # The smoother is part of the deployed decoder: applied to every
                # bin of each block, reset at recording boundaries, before masking.
                block_predictions = [decoder.predict(features, weights) if beta >= 1 else smooth(decoder.predict(features, weights), beta)
                                     for _, features in blocks]
                vy = np.concatenate([s.targets[s.mask] for s, _ in blocks])
                predicted = np.concatenate([p[s.mask] for p, (s, _) in zip(block_predictions, blocks)])
                r2, per_output = score(vy, predicted)
                day_scores.append({"method": method, "budget_seconds": budget, "r2": r2,
                                   "per_dimension_r2": per_output, "fit_ms": milliseconds,
                                   "adapted_parameters": count, "scored_calibration_bins": len(prefixes[budget][0])})
                predictions[method][str(budget)] = display_array(np.concatenate([p[::step] for p in block_predictions]))
                if method == selection["method"] and budget == selection["budget_seconds"]:
                    slope, bias = decoder.folded(weights)
                    mean, variance = views[budget][2]
                    model_id = len(compiled_models)
                    compiled_models.append({"day": day, "method": method, "budget_seconds": budget,
                                            "slope": slope, "bias": bias, "patch": patch,
                                            "zscore_mean": mean.tolist(), "zscore_variance": variance.tolist()})
                    for block_index, (session, _) in enumerate(raw_blocks):
                        compiled_cases.append((model_id, session.counts, block_predictions[block_index]))
            print(f"EVAL {day} {method:14} " + " ".join(f"{r['budget_seconds']}s:{r['r2']:.4f}" for r in day_scores if r['method'] == method), flush=True)
        truth, masks, breaks, seconds = [], [], [], []
        offset = 0.
        for s, features in raw_blocks:
            indices = np.arange(0, len(features), step)
            truth.extend(display_array(s.targets[indices]))
            masks.extend(s.mask[indices].tolist())
            breaks.extend((indices == 0).tolist())
            seconds.extend((offset + indices * protocol["bin_ms"] / 1000).tolist())
            offset += len(features) * protocol["bin_ms"] / 1000
        sessions.append({"day": day, "calibration_seconds": sum(len(f) for _, f in calibration[day]) * protocol["bin_ms"] / 1000,
                         "n_channels": raw_blocks[0][0].counts.shape[1], "scores": day_scores,
                         "replay": {"dt_seconds": step * protocol["bin_ms"] / 1000, "seconds": seconds,
                                    "mask": masks, "breaks": breaks, "truth": truth, "predictions": predictions}})
    summary = []
    for method in [m["id"] for m in METHODS]:
        for budget in budgets:
            rows = [(s["day"], next(r for r in s["scores"] if r["method"] == method and r["budget_seconds"] == budget)) for s in sessions]
            values = [r["r2"] for _, r in rows]
            summary.append({"method": method, "budget_seconds": budget, "mean_r2": float(np.mean(values)),
                            "median_r2": float(np.median(values)), "fit_ms": float(np.mean([r["fit_ms"] for _, r in rows])),
                            "adapted_parameters": max(r["adapted_parameters"] for _, r in rows),
                            "day_scores": [{"day": d, "r2": r["r2"]} for d, r in rows]})
    return sessions, summary, compiled_models, compiled_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-only", action="store_true", help="Never open later-day held-out-calib labels")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    prepare()
    development = load_sessions(("held-in-calib",))
    training = group_days(development, protocol)
    selection = choose(protocol, training)
    selection_path = RESULTS / "recovery-selection.json"
    save_json(selection_path, selection)
    print(f"FROZEN selection: {selection['method']} versus {selection['baseline']} at {selection['budget_seconds']}s; {digest(selection_path)}", flush=True)
    if args.selection_only:
        return
    decoder = fit_base({day: scored_rows(blocks) for day, blocks in zscore_training(training, protocol).items()},
                       selection["base_alpha"])
    # Current selection is frozen before loading later-day targets. These
    # recordings were evaluated previously; this run is reused-data evidence.
    later_days = group_days(load_sessions(("held-out-calib",)), protocol)
    if any(len(blocks) != 2 for blocks in later_days.values()) or not later_days:
        raise ValueError("Expected exactly two pinned recording blocks per later day")
    calibration = {day: blocks[:1] for day, blocks in later_days.items()}
    evaluation = {day: blocks[1:] for day, blocks in later_days.items()}
    sessions, summary, models, cases = evaluate(protocol, selection, decoder, calibration, evaluation,
                                                pooled_statistics(training))
    from export import compile_and_validate
    compiled = compile_and_validate(decoder, models, cases, protocol, RESULTS / "recovery_decoder.rs")
    payload = {
        "schema_version": 3,
        "evaluation_evidence": "Reused-data reevaluation, not a fresh holdout: earlier pipelines inspected these later-day recordings. This pipeline's method, alphas, and front-end were selected on held-in cross-recording folds only and pre-registered (examples/recovery/preregistration-2026-10-07.json) before any later-day score was computed for them; no later-day labels were used for selection.",
        "dataset": {"name": "FALCON H1", "subject": "One deidentified human participant",
                    "task": "Seven-dimensional cued reach and grasp",
                    "source_url": "https://dandiarchive.org/dandiset/000954/draft", "bin_ms": protocol["bin_ms"],
                    "output_labels": list(development[0].labels),
                    "scope": "Reused-data reevaluation under the original chronological-block split; not a fresh holdout."},
        "protocol": {"calibration_budgets_seconds": protocol["calibration_budgets_seconds"],
                     "selection": protocol["selection"], "evaluation": protocol["scoring"],
                     "limitations": ["Previously inspected second-block recordings from public calibration files; reused-data evidence, not a fresh holdout or official private test.",
                                     "Offline open-loop replay, not live or closed-loop robot control.",
                                     "One participant; obfuscated session dates; no clinical or novelty claim.",
                                     "Draft DANDI release reports Invalid validation status; exact assets and SHA256 are pinned.",
                                     "Display rounded to six decimals and downsampled to10Hz; scores use eligible50Hz bins."]},
        "methods": METHODS,
        "selected": {"method": selection["method"], "budget_seconds": selection["budget_seconds"],
                     "baseline": selection["baseline"], "hyperparameters": {"base_alpha": selection["base_alpha"],
                     "adaptation_alphas": selection["adaptation_alphas"]},
                     "candidate_methods": protocol["candidate_methods"],
                     "baseline_methods": protocol["baseline_methods"]},
        "summary": summary, "sessions": sessions, "compiled_validation": compiled,
        "file_splits": {"calibration": {d: [s.key for s, _ in b] for d, b in calibration.items()},
                        "evaluation": {d: [s.key for s, _ in b] for d, b in evaluation.items()}},
        "provenance": {"manifest_sha256": digest(ROOT / "examples/recovery/manifest.json"),
                       "protocol_sha256": digest(PROTOCOL), "selection_sha256": digest(selection_path),
                       "generated_at": datetime.now(timezone.utc).isoformat(),
                       "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}},
    }
    save_json(RESULTS / "recovery.json", payload, compact=True)
    save_json(RESULTS / "recovery-patches.json", [{k: v for k, v in model.items() if k not in ("slope", "bias")} for model in models])
    budget = selection["budget_seconds"]
    rows = [r for r in summary if r["budget_seconds"] == budget]
    repaired = next(r for r in rows if r["method"] == selection["method"])
    strongest = max((r for r in rows if r["method"] in protocol["baseline_methods"]), key=lambda r: r["mean_r2"])
    frozen = next(r for r in rows if r["method"] == "frozen")
    print(json.dumps({"selected_candidate": selection["method"], "budget_seconds": budget,
                      "frozen_r2": frozen["mean_r2"], "repair_r2": repaired["mean_r2"],
                      "strongest_evaluation_baseline": strongest["method"], "baseline_r2": strongest["mean_r2"],
                      "repair_minus_strongest_baseline": repaired["mean_r2"] - strongest["mean_r2"],
                      "adapted_parameters": repaired["adapted_parameters"], "full_decoder_parameters": decoder.weights.size,
                      "compiled_validation": compiled}, indent=2), flush=True)


if __name__ == "__main__":
    main()
