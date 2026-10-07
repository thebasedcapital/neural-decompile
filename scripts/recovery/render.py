#!/usr/bin/env python3
"""Render a self-contained offline replay of measured human decoder recovery results."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlparse

SCHEMA_VERSION = 3
DAY_RE = re.compile(r"^[0-9]{8}$")
METHOD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ContractError(ValueError):
    """Raised when recovery JSON does not match the current measured-result schema."""


def fail(path: str, message: str) -> NoReturn:
    raise ContractError(f"{path}: {message}")


def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def obj(value: Any, path: str, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(path, "expected object")
    if keys is not None:
        actual = set(value)
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        if missing:
            fail(path, f"missing keys: {', '.join(missing)}")
        if extra:
            fail(path, f"unexpected keys: {', '.join(extra)}")
    return value


def array(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        fail(path, "expected array")
    if nonempty and not value:
        fail(path, "must not be empty")
    return value


def string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        fail(path, "expected string")
    if nonempty and not value.strip():
        fail(path, "must not be empty")
    return value


def number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, "expected finite number")
    result = float(value)
    if not math.isfinite(result):
        fail(path, "expected finite number")
    if minimum is not None and result < minimum:
        fail(path, f"must be at least {minimum:g}")
    return result


def integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, "expected integer")
    if minimum is not None and value < minimum:
        fail(path, f"must be at least {minimum}")
    return value


def boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, "expected boolean")
    return value


def strings(value: Any, path: str, *, length: int | None = None) -> list[str]:
    values = array(value, path, nonempty=True)
    if length is not None and len(values) != length:
        fail(path, f"expected exactly {length} strings")
    result = [string(item, f"{path}[{index}]") for index, item in enumerate(values)]
    return result


def numeric_vector(
    value: Any, path: str, length: int, *, nullable: bool = False
) -> list[Any]:
    values = array(value, path)
    if len(values) != length:
        fail(path, f"expected exactly {length} values")
    for index, item in enumerate(values):
        if item is None and nullable:
            continue
        number(item, f"{path}[{index}]")
    return values


def matrix7(
    value: Any, path: str, rows: int, *, nullable_rows: list[bool] | None = None
) -> list[Any]:
    values = array(value, path)
    if len(values) != rows:
        fail(path, f"expected {rows} rows, found {len(values)}")
    for index, row in enumerate(values):
        numeric_vector(
            row,
            f"{path}[{index}]",
            7,
            nullable=nullable_rows is not None and nullable_rows[index],
        )
    return values


def same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def budget_key(value: Any, path: str) -> float:
    if not isinstance(value, str) or not value.strip():
        fail(path, "prediction budget key must be a numeric string")
    try:
        parsed = float(value)
    except ValueError:
        fail(path, "prediction budget key must be a numeric string")
    if not math.isfinite(parsed) or parsed < 0:
        fail(path, "prediction budget key must be finite and non-negative")
    return parsed


def validate(data: Any) -> dict[str, Any]:
    root = obj(data, "$")
    required_root = {
        "schema_version",
        "evaluation_evidence",
        "dataset",
        "protocol",
        "methods",
        "selected",
        "summary",
        "sessions",
        "compiled_validation",
        "file_splits",
        "provenance",
    }
    missing_root = sorted(required_root - set(root))
    if missing_root:
        fail("$", f"missing keys: {', '.join(missing_root)}")
    if integer(root["schema_version"], "$.schema_version") != SCHEMA_VERSION:
        fail("$.schema_version", f"expected {SCHEMA_VERSION}")
    string(root["evaluation_evidence"], "$.evaluation_evidence")

    dataset = obj(
        root["dataset"],
        "$.dataset",
        {"name", "subject", "task", "source_url", "bin_ms", "output_labels", "scope"},
    )
    for key in ("name", "subject", "task", "scope"):
        string(dataset[key], f"$.dataset.{key}")
    source_url = string(dataset["source_url"], "$.dataset.source_url")
    if urlparse(source_url).scheme not in {"http", "https"}:
        fail("$.dataset.source_url", "must be an http(s) URL")
    number(dataset["bin_ms"], "$.dataset.bin_ms", minimum=0.000001)
    labels = strings(dataset["output_labels"], "$.dataset.output_labels", length=7)
    if len(set(labels)) != 7:
        fail("$.dataset.output_labels", "labels must be unique")

    protocol = obj(
        root["protocol"],
        "$.protocol",
        {"calibration_budgets_seconds", "selection", "evaluation", "limitations"},
    )
    raw_budgets = array(
        protocol["calibration_budgets_seconds"],
        "$.protocol.calibration_budgets_seconds",
        nonempty=True,
    )
    budgets = [
        number(value, f"$.protocol.calibration_budgets_seconds[{index}]", minimum=0)
        for index, value in enumerate(raw_budgets)
    ]
    if len(set(budgets)) != len(budgets):
        fail("$.protocol.calibration_budgets_seconds", "budgets must be unique")
    if budgets != sorted(budgets):
        fail("$.protocol.calibration_budgets_seconds", "budgets must be increasing")
    string(protocol["selection"], "$.protocol.selection")
    string(protocol["evaluation"], "$.protocol.evaluation")
    limitations = array(protocol["limitations"], "$.protocol.limitations", nonempty=True)
    for index, limitation in enumerate(limitations):
        string(limitation, f"$.protocol.limitations[{index}]")

    raw_methods = array(root["methods"], "$.methods", nonempty=True)
    method_ids: list[str] = []
    for index, raw_method in enumerate(raw_methods):
        path = f"$.methods[{index}]"
        method = obj(raw_method, path, {"id", "label", "description"})
        method_id = string(method["id"], f"{path}.id")
        if not METHOD_RE.fullmatch(method_id):
            fail(f"{path}.id", "must contain only letters, digits, dot, underscore, or hyphen")
        string(method["label"], f"{path}.label")
        string(method["description"], f"{path}.description")
        method_ids.append(method_id)
    if len(set(method_ids)) != len(method_ids):
        fail("$.methods", "method ids must be unique")
    if "frozen" not in method_ids:
        fail("$.methods", "must include the measured baseline method id 'frozen'")
    adapted_methods = [method_id for method_id in method_ids if method_id != "frozen"]
    if not adapted_methods:
        fail("$.methods", "must include at least one adapted method in addition to 'frozen'")

    selected = obj(
        root["selected"],
        "$.selected",
        {"method", "budget_seconds", "baseline", "hyperparameters", "candidate_methods", "baseline_methods"},
    )
    selected_method = string(selected["method"], "$.selected.method")
    if selected_method not in adapted_methods:
        fail("$.selected.method", "must identify a non-frozen method from $.methods")
    selected_baseline = string(selected["baseline"], "$.selected.baseline")
    if selected_baseline not in method_ids or selected_baseline == selected_method:
        fail("$.selected.baseline", "must identify a different method from $.methods")
    selected_budget = number(selected["budget_seconds"], "$.selected.budget_seconds", minimum=0)
    if selected_budget not in budgets:
        fail("$.selected.budget_seconds", "must occur in protocol calibration budgets")
    obj(selected["hyperparameters"], "$.selected.hyperparameters")

    expected_combos = {
        (method_id, budget) for method_id in method_ids for budget in budgets
    }
    raw_summary = array(root["summary"], "$.summary", nonempty=True)
    summary_by_combo: dict[tuple[str, float], dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_summary):
        path = f"$.summary[{index}]"
        row = obj(
            raw_row,
            path,
            {
                "method",
                "budget_seconds",
                "mean_r2",
                "median_r2",
                "fit_ms",
                "adapted_parameters",
                "day_scores",
            },
        )
        method_id = string(row["method"], f"{path}.method")
        if method_id not in method_ids:
            fail(f"{path}.method", "unknown method id")
        budget = number(row["budget_seconds"], f"{path}.budget_seconds", minimum=0)
        combo = (method_id, budget)
        if combo in summary_by_combo:
            fail(path, "duplicate method and budget")
        summary_by_combo[combo] = row
        number(row["mean_r2"], f"{path}.mean_r2")
        number(row["median_r2"], f"{path}.median_r2")
        number(row["fit_ms"], f"{path}.fit_ms", minimum=0)
        integer(row["adapted_parameters"], f"{path}.adapted_parameters", minimum=0)
        raw_days = array(row["day_scores"], f"{path}.day_scores", nonempty=True)
        seen_days: set[str] = set()
        for day_index, raw_day in enumerate(raw_days):
            day_path = f"{path}.day_scores[{day_index}]"
            day_score = obj(raw_day, day_path, {"day", "r2"})
            day = string(day_score["day"], f"{day_path}.day")
            if not DAY_RE.fullmatch(day):
                fail(f"{day_path}.day", "expected an 8-digit deidentified day")
            if day in seen_days:
                fail(day_path, "duplicate day")
            seen_days.add(day)
            number(day_score["r2"], f"{day_path}.r2")
    actual_combos = set(summary_by_combo)
    if actual_combos != expected_combos:
        missing = sorted(expected_combos - actual_combos)
        extra = sorted(actual_combos - expected_combos)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        fail("$.summary", "; ".join(details))

    raw_sessions = array(root["sessions"], "$.sessions", nonempty=True)
    sessions_by_day: dict[str, dict[str, Any]] = {}
    scores_by_day: dict[str, dict[tuple[str, float], float]] = {}
    for index, raw_session in enumerate(raw_sessions):
        path = f"$.sessions[{index}]"
        session = obj(
            raw_session,
            path,
            {"day", "calibration_seconds", "n_channels", "scores", "replay"},
        )
        day = string(session["day"], f"{path}.day")
        if not DAY_RE.fullmatch(day):
            fail(f"{path}.day", "expected an 8-digit deidentified day")
        if day in sessions_by_day:
            fail(path, "duplicate day")
        sessions_by_day[day] = session
        number(session["calibration_seconds"], f"{path}.calibration_seconds", minimum=0)
        integer(session["n_channels"], f"{path}.n_channels", minimum=1)

        raw_scores = array(session["scores"], f"{path}.scores", nonempty=True)
        session_scores: dict[tuple[str, float], float] = {}
        for score_index, raw_score in enumerate(raw_scores):
            score_path = f"{path}.scores[{score_index}]"
            score = obj(raw_score, score_path)
            required_score = {
                "method", "budget_seconds", "r2", "per_dimension_r2", "fit_ms",
                "adapted_parameters", "scored_calibration_bins",
            }
            missing_score = sorted(required_score - set(score))
            if missing_score:
                fail(score_path, f"missing keys: {', '.join(missing_score)}")
            method_id = string(score["method"], f"{score_path}.method")
            if method_id not in method_ids:
                fail(f"{score_path}.method", "unknown method id")
            budget = number(score["budget_seconds"], f"{score_path}.budget_seconds", minimum=0)
            combo = (method_id, budget)
            if combo in session_scores:
                fail(score_path, "duplicate method and budget")
            score_value = number(score["r2"], f"{score_path}.r2")
            session_scores[combo] = score_value
            numeric_vector(score["per_dimension_r2"], f"{score_path}.per_dimension_r2", 7)
            number(score["fit_ms"], f"{score_path}.fit_ms", minimum=0)
            integer(score["adapted_parameters"], f"{score_path}.adapted_parameters", minimum=0)
            integer(score["scored_calibration_bins"], f"{score_path}.scored_calibration_bins", minimum=0)
        if set(session_scores) != expected_combos:
            fail(f"{path}.scores", "must contain every method and calibration-budget combination")
        scores_by_day[day] = session_scores

        replay = obj(
            session["replay"],
            f"{path}.replay",
            {"dt_seconds", "seconds", "mask", "breaks", "truth", "predictions"},
        )
        number(replay["dt_seconds"], f"{path}.replay.dt_seconds", minimum=0.000001)
        raw_seconds = array(replay["seconds"], f"{path}.replay.seconds", nonempty=True)
        if len(raw_seconds) < 2:
            fail(f"{path}.replay.seconds", "must contain at least two measured samples")
        seconds = [
            number(value, f"{path}.replay.seconds[{sample_index}]")
            for sample_index, value in enumerate(raw_seconds)
        ]
        if any(right <= left for left, right in zip(seconds, seconds[1:])):
            fail(f"{path}.replay.seconds", "timestamps must be strictly increasing")
        sample_count = len(seconds)
        for vector_name in ("mask", "breaks"):
            values = array(replay[vector_name], f"{path}.replay.{vector_name}")
            if len(values) != sample_count:
                fail(f"{path}.replay.{vector_name}", f"expected {sample_count} values")
            for sample_index, value in enumerate(values):
                boolean(value, f"{path}.replay.{vector_name}[{sample_index}]")
        if not any(replay["mask"]):
            fail(f"{path}.replay.mask", "must include at least one scored sample")
        nullable_rows = [not value for value in replay["mask"]]
        truth = matrix7(
            replay["truth"], f"{path}.replay.truth", sample_count,
            nullable_rows=nullable_rows,
        )
        for sample_index, row in enumerate(truth):
            if replay["mask"][sample_index] and any(value is None for value in row):
                fail(
                    f"{path}.replay.truth[{sample_index}]",
                    "scored rows must contain seven finite targets",
                )

        predictions = obj(replay["predictions"], f"{path}.replay.predictions")
        if set(predictions) != set(method_ids):
            fail(f"{path}.replay.predictions", "must contain every method id and no unknown ids")
        for method_id in method_ids:
            method_path = f"{path}.replay.predictions.{method_id}"
            raw_method_predictions = obj(predictions[method_id], method_path)
            parsed_budgets: dict[float, str] = {}
            for raw_budget, prediction in raw_method_predictions.items():
                parsed_budget = budget_key(raw_budget, f"{method_path}.{raw_budget}")
                if parsed_budget in parsed_budgets:
                    fail(method_path, "duplicate numerically equivalent budget keys")
                parsed_budgets[parsed_budget] = raw_budget
                matrix7(prediction, f"{method_path}.{raw_budget}", sample_count)
            expected_budgets = set(budgets)
            if set(parsed_budgets) != expected_budgets:
                fail(method_path, f"expected prediction budgets {sorted(expected_budgets)}")

    session_days = set(sessions_by_day)
    for combo, row in summary_by_combo.items():
        row_days = {entry["day"] for entry in row["day_scores"]}
        if row_days != session_days:
            fail("$.summary", f"{combo} day_scores must cover every session day")
        for entry in row["day_scores"]:
            measured = scores_by_day[entry["day"]][combo]
            if not same_number(float(entry["r2"]), measured):
                fail(
                    "$.summary",
                    f"{combo} day score for {entry['day']} disagrees with session score",
                )

    compiled = obj(root["compiled_validation"], "$.compiled_validation")
    required_compiled = {
        "bin_count", "recording_count", "max_absolute_error", "tolerance",
        "total_wall_ms", "wall_us_per_bin_including_io_and_startup",
        "source_sha256", "base_sha256", "rustc", "scope",
    }
    missing_compiled = sorted(required_compiled - set(compiled))
    if missing_compiled:
        fail("$.compiled_validation", f"missing keys: {', '.join(missing_compiled)}")
    integer(compiled["bin_count"], "$.compiled_validation.bin_count", minimum=1)
    integer(compiled["recording_count"], "$.compiled_validation.recording_count", minimum=1)
    number(compiled["max_absolute_error"], "$.compiled_validation.max_absolute_error", minimum=0)
    number(compiled["total_wall_ms"], "$.compiled_validation.total_wall_ms", minimum=0)
    number(compiled["wall_us_per_bin_including_io_and_startup"], "$.compiled_validation.wall_us_per_bin_including_io_and_startup", minimum=0)
    tolerance = obj(compiled["tolerance"], "$.compiled_validation.tolerance", {"rtol", "atol"})
    number(tolerance["rtol"], "$.compiled_validation.tolerance.rtol", minimum=0)
    number(tolerance["atol"], "$.compiled_validation.tolerance.atol", minimum=0)
    for key in ("source_sha256", "base_sha256"):
        digest = string(compiled[key], f"$.compiled_validation.{key}")
        if not HASH_RE.fullmatch(digest):
            fail(f"$.compiled_validation.{key}", "expected a SHA-256 hex digest")
    string(compiled["rustc"], "$.compiled_validation.rustc")
    string(compiled["scope"], "$.compiled_validation.scope")
    file_splits = obj(root["file_splits"], "$.file_splits", {"calibration", "evaluation"})
    for split_name in ("calibration", "evaluation"):
        split = obj(file_splits[split_name], f"$.file_splits.{split_name}")
        if set(split) != session_days:
            fail(f"$.file_splits.{split_name}", "must list every evaluated day")
        for day, filenames in split.items():
            if not DAY_RE.fullmatch(day):
                fail(f"$.file_splits.{split_name}", f"invalid day key {day!r}")
            strings(filenames, f"$.file_splits.{split_name}.{day}")

    provenance = obj(root["provenance"], "$.provenance")
    required_provenance = {
        "manifest_sha256", "protocol_sha256", "selection_sha256",
        "generated_at", "code_sha256",
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    if missing_provenance:
        fail("$.provenance", f"missing keys: {', '.join(missing_provenance)}")
    for key in ("manifest_sha256", "protocol_sha256", "selection_sha256"):
        value = string(provenance[key], f"$.provenance.{key}")
        if not HASH_RE.fullmatch(value):
            fail(f"$.provenance.{key}", "expected a 64-character SHA-256 hex digest")
    code_hashes = obj(provenance["code_sha256"], "$.provenance.code_sha256")
    if not code_hashes:
        fail("$.provenance.code_sha256", "must not be empty")
    for filename, value in code_hashes.items():
        string(filename, "$.provenance.code_sha256 key")
        digest = string(value, f"$.provenance.code_sha256.{filename}")
        if not HASH_RE.fullmatch(digest):
            fail(f"$.provenance.code_sha256.{filename}", "expected a SHA-256 hex digest")
    string(provenance["generated_at"], "$.provenance.generated_at")
    return root


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"input does not exist or is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(
                handle,
                object_pairs_hook=object_pairs,
                parse_constant=lambda value: fail("JSON", f"invalid numeric constant {value}"),
            )
    except UnicodeDecodeError as exc:
        raise ContractError(f"input is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    return validate(data)


def render(data: dict[str, Any], template_path: Path) -> str:
    if not template_path.is_file():
        raise ContractError(f"template does not exist or is not a file: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    marker = "__RECOVERY_JSON__"
    if template.count(marker) != 1:
        raise ContractError(f"template must contain exactly one {marker} marker")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return template.replace(marker, encoded)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render measured recovery JSON as a self-contained offline replay."
    )
    parser.add_argument("input", type=Path, help="schema-v1 recovery JSON")
    parser.add_argument("output", type=Path, help="destination self-contained HTML")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("replay.html"),
        help="HTML template containing __RECOVERY_JSON__ (default: replay.html beside this script)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ContractError("input and output paths must differ")
        data = load_json(args.input)
        page = render(data, args.template)
        atomic_write(args.output, page)
    except (ContractError, OSError) as exc:
        print(f"render.py: error: {exc}", file=sys.stderr)
        return 2
    print(f"Rendered {len(data['sessions'])} measured day(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
