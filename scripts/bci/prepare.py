#!/usr/bin/env python3
"""Download NLB MC_Maze_Small (DANDI 000140) and cut per-trial spike counts.

Output: examples/bci/data/trials.npz with
  counts   int16 [trial, unit, bin]   spike counts in 10 ms bins, t in [-500, +500) ms around move onset
  label    int64 [trial]              reach quadrant of the active target (see QUADRANTS)
  split    str   [trial]              'train' / 'val' as published in the dataset's trial table
  maze_id  int64 [trial]
  unit_id  int64 [unit]               DANDI units/id of each row of `counts`
  hand_dx, hand_dy float [trial]      hand displacement onset -> end of reach (for sanity check of labels)
Only needs: numpy, h5py (reads the NWB file directly as HDF5).
"""
import hashlib, json, sys, urllib.request
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples" / "bci" / "data"
DANDISET = "000140"
ASSET_PATH = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
API = "https://api.dandiarchive.org/api"
BIN_MS, PRE_MS, POST_MS = 10, 500, 500
# label = 2*(x>0) + (y>0) of the active target position (== sign of hand displacement; checked below)
QUADRANTS = {0: "left-down", 1: "left-up", 2: "right-down", 3: "right-up"}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((DATA / "PROVENANCE.json").read_text())
    asset = provenance["asset_used"]
    dest = DATA / Path(asset["path"]).name
    if not dest.exists():
        print(f"downloading pinned {asset['path']} from DANDI:{DANDISET}", file=sys.stderr)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(f"{API}/assets/{asset['asset_id']}/download/", tmp)
        with tmp.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha2_256"]:
            tmp.unlink()
            raise ValueError("Downloaded NWB digest does not match pinned provenance")
        tmp.rename(dest)
    with dest.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != asset["sha2_256"]:
        raise ValueError("Cached NWB digest does not match pinned provenance")
    return dest


def main():
    nwb = download()
    f = h5py.File(nwb, "r")
    t = f["intervals/trials"]
    onset = t["move_onset_time"][:]
    stop = t["stop_time"][:]
    n_targets = t["num_targets"][:]
    active = t["active_target"][:]
    # target_pos_index is a cumulative END index into target_pos (ragged-array convention)
    tpos_end = t["target_pos_index"][:]
    target_pos = t["target_pos"][:]
    split = np.array([s.decode() if isinstance(s, bytes) else s for s in t["split"][:]])
    maze = t["maze_id"][:]
    assert t["success"][:].all()

    hp = f["processing/behavior/hand_pos/data"][:]
    hts = f["processing/behavior/hand_pos/timestamps"][:]
    tgt = np.array([target_pos[tpos_end[i] - n_targets[i] + active[i]] for i in range(len(onset))])
    p0 = hp[np.searchsorted(hts, onset)]
    p1 = hp[np.searchsorted(hts, stop - 0.05)]
    d = p1 - p0
    label = (tgt[:, 0] > 0).astype(np.int64) * 2 + (tgt[:, 1] > 0)
    label_hand = (d[:, 0] > 0).astype(np.int64) * 2 + (d[:, 1] > 0)
    print(f"quadrant label from target == from hand displacement: {(label == label_hand).mean():.0%}", file=sys.stderr)

    spikes = f["units/spike_times"][:]
    uidx = f["units/spike_times_index"][:]
    unit_id = f["units/id"][:]
    starts = np.r_[0, uidx[:-1]]
    nb = (PRE_MS + POST_MS) // BIN_MS
    counts = np.zeros((len(onset), len(unit_id), nb), dtype=np.int16)
    for u, (a, b) in enumerate(zip(starts, uidx)):
        st = spikes[a:b]
        for i, t0 in enumerate(onset):
            lo = t0 - PRE_MS / 1000
            sel = st[(st >= lo) & (st < lo + nb * BIN_MS / 1000)]
            counts[i, u] = np.bincount(((sel - lo) * 1000 // BIN_MS).astype(int), minlength=nb)
    out = DATA / "trials.npz"
    np.savez_compressed(out, counts=counts, label=label, split=split, maze_id=maze, unit_id=unit_id,
                        hand_dx=d[:, 0], hand_dy=d[:, 1], bin_ms=BIN_MS, pre_ms=PRE_MS)
    print(f"wrote {out}: {counts.shape[0]} trials x {counts.shape[1]} units x {nb} bins; "
          f"label counts {np.bincount(label).tolist()}; split {dict(zip(*np.unique(split, return_counts=True)))}")


if __name__ == "__main__":
    main()
