#!/usr/bin/env python3
"""Verified download and h5py loading for the public FALCON H1 dataset."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.request import Request, urlopen

import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _ROOT / "examples" / "recovery" / "manifest.json"
_DEFAULT_DATA_DIR = _ROOT / "examples" / "recovery" / "data"
_BIN_SIZE_SECONDS = 0.02
_ALLOWED_SPLITS = frozenset(
    (
        "held-in-calib",
        "held-in-minival",
        "held-out-calib",
        "held-out-minival",
    )
)
_DAY_RE = re.compile(r"_ses-(\d{8})(?:T\d{6})?\.nwb$")
_DOWNLOAD_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Session:
    key: str
    day: str
    split: str
    counts: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    timestamps: np.ndarray
    trial_ids: np.ndarray
    labels: list[str]


def _manifest() -> dict[str, Any]:
    with _MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assets = manifest.get("assets")
    if manifest.get("schema_version") != 1 or not isinstance(assets, list):
        raise ValueError(f"Unsupported recovery manifest: {_MANIFEST_PATH}")
    if len(assets) != 40:
        raise ValueError(f"Expected 40 pinned DANDI assets, found {len(assets)}")
    paths: set[str] = set()
    asset_ids: set[str] = set()
    for asset in assets:
        required = {"asset_id", "path", "split", "size", "sha256", "download_url"}
        if not required.issubset(asset):
            raise ValueError(f"Incomplete asset entry in {_MANIFEST_PATH}")
        if asset["split"] not in _ALLOWED_SPLITS:
            raise ValueError(f"Unknown manifest split: {asset['split']!r}")
        if asset["path"] in paths or asset["asset_id"] in asset_ids:
            raise ValueError("Manifest contains a duplicate path or asset UUID")
        if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {asset['path']}")
        paths.add(asset["path"])
        asset_ids.add(asset["asset_id"])
    return manifest


def _cache_root(data_dir: Path | None) -> Path:
    return (Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR).resolve()


def _asset_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest asset escapes data directory: {relative_path!r}")
    return path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DOWNLOAD_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(path: Path, asset: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == asset["size"]
        and _file_sha256(path) == asset["sha256"]
    )


def _download(asset: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        asset["download_url"],
        headers={"User-Agent": "neural-decompile-recovery/1"},
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".part",
            dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            digest = sha256()
            size = 0
            with urlopen(request) as response:
                while block := response.read(_DOWNLOAD_BLOCK_BYTES):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if size != asset["size"]:
            raise OSError(
                f"Downloaded size mismatch for {asset['path']}: "
                f"expected {asset['size']}, received {size}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != asset["sha256"]:
            raise OSError(
                f"SHA-256 mismatch for {asset['path']}: "
                f"expected {asset['sha256']}, received {actual_digest}"
            )
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(data_dir: Path | None = None) -> None:
    """Download every pinned public asset and verify its size and SHA-256."""
    root = _cache_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for asset in _manifest()["assets"]:
        destination = _asset_path(root, asset["path"])
        if not _verified(destination, asset):
            _download(asset, destination)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _labels(group: Any) -> list[str]:
    description = group.attrs.get("description")
    if description is None:
        raise ValueError(f"Missing output description on {group.name}")
    labels = [label.strip() for label in _text(description).split(",")]
    if len(labels) != 7 or any(not label for label in labels):
        raise ValueError(f"Expected seven output labels on {group.name}, got {labels!r}")
    return labels


def _canonical_timestamps(kinematics: Any, length: int) -> np.ndarray:
    data = kinematics["data"]
    starting_time = kinematics.get("starting_time")
    if starting_time is None or "rate" not in starting_time.attrs:
        raise ValueError(f"Missing regular timestamp metadata on {kinematics.name}")
    # This intentionally follows the public FALCON H1 loader: its timestamp
    # origin is the TimeSeries data offset and its stored rate is the 20 ms step.
    offset = float(data.attrs.get("offset", 0.0))
    step = float(starting_time.attrs["rate"])
    if not np.isclose(step, _BIN_SIZE_SECONDS):
        raise ValueError(f"Expected 20 ms samples on {kinematics.name}, got {step!r}")
    return offset + np.arange(length, dtype=np.float64) * step


def _bin_spikes(units: Any, timestamps: np.ndarray) -> np.ndarray:
    """Match falcon_challenge.dataloaders.bin_units for timestamp-ending bins."""
    unit_ids = units["id"]
    spike_times = np.asarray(units["spike_times"], dtype=np.float64)
    spike_ends = np.asarray(units["spike_times_index"], dtype=np.int64)
    if spike_ends.shape != unit_ids.shape:
        raise ValueError("NWB units id and spike_times_index lengths differ")
    if spike_ends.size and (
        np.any(np.diff(spike_ends) < 0) or spike_ends[-1] != spike_times.size
    ):
        raise ValueError("Invalid ragged spike_times index")

    bin_ends = timestamps
    if bin_ends.size == 0:
        return np.empty((0, spike_ends.size), dtype=np.float64)
    gaps = np.diff(bin_ends)
    if np.any(gaps <= 0):
        raise ValueError("Bin timestamps must be monotonically increasing")
    if gaps.size == 0 or np.allclose(gaps, _BIN_SIZE_SECONDS):
        expanded_ends = bin_ends
        keep = np.ones(bin_ends.size, dtype=bool)
    else:
        expanded = [float(bin_ends[0])]
        keep_values = [True]
        for index, gap in enumerate(gaps):
            next_end = float(bin_ends[index + 1])
            if not np.isclose(gap, _BIN_SIZE_SECONDS) and gap > _BIN_SIZE_SECONDS:
                expanded.extend((next_end - _BIN_SIZE_SECONDS, next_end))
                keep_values.extend((False, True))
            else:
                expanded.append(next_end)
                keep_values.append(True)
        expanded_ends = np.asarray(expanded, dtype=np.float64)
        keep = np.asarray(keep_values, dtype=bool)

    edges = np.concatenate(
        (np.asarray([expanded_ends[0] - _BIN_SIZE_SECONDS]), expanded_ends)
    )
    counts = np.empty((timestamps.size, spike_ends.size), dtype=np.float64)
    start = 0
    for column, end in enumerate(spike_ends):
        histogram = np.histogram(spike_times[start:end], bins=edges)[0]
        counts[:, column] = histogram[keep]
        start = int(end)
    return counts


def _read_session(path: Path, split: str) -> Session:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("load_sessions requires h5py") from error

    match = _DAY_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract deidentified session day from {path.name!r}")
    with h5py.File(path, "r") as nwb:
        acquisition = nwb["acquisition"]
        kinematics = acquisition["OpenLoopKinematics"]
        velocity = acquisition["OpenLoopKinematicsVelocity"]
        targets = np.asarray(velocity["data"], dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 7:
            raise ValueError(f"Expected [T, 7] velocity targets in {path}")
        labels = _labels(kinematics)
        velocity_labels = _labels(velocity)
        if velocity_labels != labels:
            raise ValueError(f"Position and velocity labels differ in {path}")
        timestamps = _canonical_timestamps(kinematics, targets.shape[0])
        counts = _bin_spikes(nwb["units"], timestamps)
        trial_ids = np.asarray(acquisition["TrialNum"]["data"])
        eval_mask = np.asarray(acquisition["eval_mask"]["data"], dtype=bool)

    length = targets.shape[0]
    for name, array in (
        ("counts", counts),
        ("timestamps", timestamps),
        ("trial_ids", trial_ids),
        ("eval_mask", eval_mask),
    ):
        if array.shape[0] != length:
            raise ValueError(
                f"{name} length {array.shape[0]} does not match targets {length} in {path}"
            )
    mask = eval_mask & np.isfinite(targets).all(axis=1)
    return Session(
        key=path.stem,
        day=match.group(1),
        split=split,
        counts=counts,
        targets=targets,
        mask=mask,
        timestamps=timestamps,
        trial_ids=trial_ids,
        labels=labels,
    )


def load_sessions(
    splits: tuple[str, ...], data_dir: Path | None = None
) -> list[Session]:
    """Load only the requested public splits, sorted by unique filename stem."""
    unknown = set(splits) - _ALLOWED_SPLITS
    if unknown:
        raise ValueError(f"Unknown recovery split(s): {sorted(unknown)!r}")
    requested = set(splits)
    root = _cache_root(data_dir)
    selected = [
        asset for asset in _manifest()["assets"] if asset["split"] in requested
    ]
    sessions: list[Session] = []
    expected_labels: list[str] | None = None
    keys: set[str] = set()
    for asset in selected:
        path = _asset_path(root, asset["path"])
        if not _verified(path, asset):
            raise FileNotFoundError(
                f"Missing or unverified recovery asset {path}; run {__file__} first"
            )
        session = _read_session(path, asset["split"])
        if session.key in keys:
            raise ValueError(f"Duplicate recovery session key: {session.key}")
        keys.add(session.key)
        if expected_labels is None:
            expected_labels = session.labels
        elif session.labels != expected_labels:
            raise ValueError(
                f"Output labels differ in {path}: {session.labels!r} != {expected_labels!r}"
            )
        sessions.append(session)
    return sorted(sessions, key=lambda session: session.key)


def _main() -> None:
    prepare()
    manifest = _manifest()
    split_counts = {
        split: sum(asset["split"] == split for asset in manifest["assets"])
        for split in sorted(_ALLOWED_SPLITS)
    }
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"]["name"],
                "dandiset": manifest["dataset"]["identifier"],
                "version": manifest["dataset"]["version"],
                "status": manifest["dataset"]["status"],
                "license": manifest["dataset"]["license"],
                "assets": len(manifest["assets"]),
                "bytes": sum(asset["size"] for asset in manifest["assets"]),
                "splits": split_counts,
                "cache": str(_DEFAULT_DATA_DIR),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _main()
