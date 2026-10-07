#!/usr/bin/env python3
"""Precompute causal multi-tau features for every held-in session into one npz.

Output: arrays per session day, split-aware. The pod never needs DANDI access.
"""
import json
from pathlib import Path

import numpy as np

import experiment as ex

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/recovery-training"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    protocol = json.loads(ex.PROTOCOL.read_text())
    taus = protocol["filter"]["taus_ms"]
    ex.prepare()
    sessions = ex.load_sessions(("held-in-calib", "held-in-minival"))
    manifest = {}
    for session in sessions:
        features = ex.stacked_features(session.counts, protocol["bin_ms"], taus)
        key = f"{session.split}_{session.day}_{session.key}"
        np.savez_compressed(OUT / f"{key}.npz",
                            features=features.astype(np.float32),
                            targets=session.targets.astype(np.float32),
                            mask=session.mask)
        manifest[key] = {"split": session.split, "day": session.day, "key": session.key,
                         "bins": int(len(session.counts)),
                         "scored_bins": int(session.mask.sum()),
                         "features_sha256": __import__("hashlib").sha256(
                             (OUT / f"{key}.npz").read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total_bins = sum(m["bins"] for m in manifest.values())
    total_scored = sum(m["scored_bins"] for m in manifest.values())
    size_gb = sum(p.stat().st_size for p in OUT.glob("*.npz")) / 1e9
    print(json.dumps({"files": len(manifest), "total_bins": total_bins,
                      "total_scored_bins": total_scored, "size_gb": round(size_gb, 2),
                      "feature_dim": features.shape[1]}))


if __name__ == "__main__":
    main()
