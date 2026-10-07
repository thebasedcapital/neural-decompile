#!/usr/bin/env python3
"""Deterministic GRU sweep for H1 velocity decoding on precomputed features.

Causal model, forward-chaining fold validation on earlier days only.
Runs identically on a local GPU or a Runpod pod; writes one JSON per trial.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/recovery-training"
OUT = DATA / "sweep"
FEATURE_DIM, OUTPUTS = 528, 7


def load_days():
    """(day, split) -> list of (features, targets, mask); files are <split>_<day>_<key>.npz."""
    days = {}
    for path in sorted(DATA.glob("*.npz")):
        parts = path.name.split("_", 2)
        if len(parts) != 3 or parts[0] not in ("held-in-calib", "held-in-minival"):
            continue
        split, key = parts[0], parts[1]
        with np.load(path) as z:
            days.setdefault((key, split), []).append((z["features"], z["targets"], z["mask"]))
    return days


def score_r2(truth, prediction):
    error = np.sum((truth - prediction) ** 2, axis=0)
    variance = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    varying = variance > 0
    return float(1 - error[varying].sum() / variance[varying].sum())


def smooth(values, beta):
    if beta >= 1:
        return values
    out = np.empty_like(values)
    previous = None
    for index, row in enumerate(values):
        out[index] = row if previous is None else beta * row + (1 - beta) * previous
        previous = out[index]
    return out


class Gru(nn.Module):
    def __init__(self, hidden, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(FEATURE_DIM, hidden, layers, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, OUTPUTS)

    def forward(self, x, state=None):
        y, state = self.gru(x, state)
        return self.head(y), state


class DayData:
    def __init__(self, sessions, x_mean, x_scale, y_mean, y_scale):
        self.windows = []
        for features, targets, mask in sessions:
            x = ((features[mask] - x_mean) / x_scale).astype(np.float32)
            y = ((targets[mask] - y_mean) / y_scale).astype(np.float32)
            self.windows.append((x, y))
        self.total = sum(len(x) for x, _ in self.windows)

    def batches(self, rng, window, batch):
        while True:
            xs, ys = [], []
            for _ in range(batch):
                x, y = self.windows[rng.integers(len(self.windows))]
                start = int(rng.integers(0, max(1, len(x) - window + 1)))
                xs.append(x[start:start + window])
                ys.append(y[start:start + window])
            length = min(window, min(len(a) for a in xs))
            yield (torch.from_numpy(np.stack([a[:length] for a in xs])),
                   torch.from_numpy(np.stack([a[:length] for a in ys])))

    def sequences(self):
        return [(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0))
                for x, y in self.windows]


def stats(sessions):
    x = np.concatenate([f[m] for f, _, m in sessions])
    y = np.concatenate([t[m] for _, t, m in sessions])
    x_scale = x.std(axis=0); x_scale[x_scale < 1e-8] = 1
    y_scale = y.std(axis=0); y_scale[y_scale < 1e-8] = 1
    return x.mean(axis=0), x_scale, y.mean(axis=0), y_scale


def evaluate(model, day_data, device, beta):
    model.eval()
    truths, predictions = [], []
    with torch.no_grad():
        for x, y in day_data.sequences():
            p, _ = model(x.to(device))
            predictions.append(smooth(p.squeeze(0).cpu().numpy(), beta))
            truths.append(y.squeeze(0).numpy())
    return score_r2(np.concatenate(truths), np.concatenate(predictions))


def train_one(config, fold_train, fold_val, seed, epochs, device):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    x_mean, x_scale, y_mean, y_scale = stats(fold_train)
    train = DayData(fold_train, x_mean, x_scale, y_mean, y_scale)
    val = DayData(fold_val, x_mean, x_scale, y_mean, y_scale)
    model = Gru(config["hidden"], config["layers"], config["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    batches = train.batches(rng, config["window"], config["batch"])
    best_val, best_state, patience = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        seen = 0
        while seen < max(train.total, config["batch"] * config["window"]):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x)[0], y)
            if not torch.isfinite(loss):
                patience += config["patience"]  # diverged: force the early stop
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            seen += x.shape[0] * x.shape[1]
        val_r2 = evaluate(model, val, device, config["beta"])
        if val_r2 > best_val:
            best_val, patience = val_r2, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config["patience"]:
                break
    model.load_state_dict(best_state)
    raw = evaluate(model, val, device, 1.0)
    return {"val_r2": best_val, "raw_r2": raw,
            "best_beta": config["beta"] if best_val >= raw else 0.0,
            "epochs_run": epoch + 1}


def sample_config(rng, epoch_cap):
    hidden = int(2 ** rng.uniform(5.5, 9.2) // 8 * 8)
    return {
        "hidden": min(512, max(64, hidden)),
        "layers": int(rng.choice([1, 1, 2])),
        "dropout": float(rng.uniform(0.0, 0.35)),
        "lr": float(10 ** rng.uniform(-4, -2.7)),
        "weight_decay": float(10 ** rng.uniform(-6, -3)),
        "window": int(rng.choice([32, 64, 128])),
        "batch": int(rng.choice([256, 512, 1024])),
        "patience": int(rng.choice([3, 5, 8])),
        "beta": float(rng.choice([0.0, 0.1, 0.2])),
        "epoch_cap": epoch_cap,
    }


def clean(value):
    """JSON-safe: non-finite floats become None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quick", action="store_true", help="one trial, few epochs: smoke test")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    days = load_days()
    all_days = sorted({d for (d, split) in days if split == "held-in-calib"})
    fold_days = all_days[3:]
    OUT.mkdir(parents=True, exist_ok=True)
    trials = [sample_config(np.random.default_rng(args.seed + t), 3 if args.quick else args.epochs)
              for t in range(1 if args.quick else args.trials)]
    start = time.perf_counter()
    for index, config in enumerate(trials):
        fold_scores = []
        for fold_index, fold_day in enumerate(fold_days):
            day_number = 3 + fold_index
            train_sessions = [s for (d, split), ss in days.items() if split == "held-in-calib" and d in all_days[:day_number] for s in ss]
            val_sessions = days.get((fold_day, "held-in-minival"), [])
            if not train_sessions or not val_sessions:
                raise ValueError(f"Empty fold for {fold_day}")
            result = train_one(config, train_sessions, val_sessions, args.seed + index, config["epoch_cap"], device)
            fold_scores.append({"day": fold_day, **result})
        mean = float(np.mean([f["val_r2"] for f in fold_scores if math.isfinite(f["val_r2"])])) \
            if any(math.isfinite(f["val_r2"]) for f in fold_scores) else float("nan")
        record = {"trial": index, "config": config, "fold_days": fold_days,
                  "fold_scores": fold_scores, "mean_r2": mean, "seed": args.seed + index,
                  "elapsed_s": round(time.perf_counter() - start, 1),
                  "device": device,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
        path = OUT / f"trial_{index:03d}.json"
        path.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
        print(json.dumps(clean({"trial": index, "mean_r2": round(mean, 4),
                                "folds": [round(f["val_r2"], 4) for f in fold_scores],
                                "epochs": [f["epochs_run"] for f in fold_scores],
                                "hidden": config["hidden"], "lr": round(config["lr"], 6)})), flush=True)
    scores = []
    for i in range(len(trials)):
        row = json.loads((OUT / f"trial_{i:03d}.json").read_text())
        scores.append(row["mean_r2"] if row["mean_r2"] is not None else float("-inf"))
    best = max(range(len(trials)), key=scores.__getitem__)
    print(json.dumps({"best_trial": best, "best_mean_r2": scores[best], "trials": len(trials), "device": device}))


if __name__ == "__main__":
    main()
