#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["z3-solver==5.1.0.0"]
# ///
"""Extract an exact Moore machine from an integer-quantized binary-input ReLU RNN.

Two certification routes, neither based on test-set agreement:
  * Enumerate all reachable integer states until the set closes.
  * Prove a coordinate saturation map is an output-preserving transition
    homomorphism over ALL nonnegative integer states, using Z3, then enumerate
    its finite reachable quotient.

Guarantees concern the snapped mathematical-integer network, not original float
weights or unbounded floating/fixed-width execution. Unknown/timeouts/limits fail
closed. Built-in specifications are optional: the extraction never reads them.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

KEYS = ("W_hh", "W_hx", "b_h", "W_y", "b_y")
DOMAIN = "nonnegative mathematical-integer hidden states; one-hot binary inputs; first-argmax output"


class Unresolved(ValueError):
    pass


def load_model(path: Path, eps: float) -> tuple[dict, str]:
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    raw = path.read_bytes()
    weights = json.loads(raw)
    h, o = len(weights["b_h"]), len(weights["b_y"])
    if h == 0 or o == 0:
        raise ValueError("hidden and output dimensions must be nonzero")
    for key, rows, cols in (("W_hh", h, h), ("W_hx", h, 2), ("W_y", o, h)):
        if len(weights[key]) != rows or any(len(row) != cols for row in weights[key]):
            raise ValueError(f"{key}: expected {rows} by {cols}; only two-channel one-hot binary inputs are supported")

    def snap(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("weights must be finite numbers")
        value = float(value)  # Match nd's JSON-to-f64 conversion before snapping.
        if not math.isfinite(value):
            raise ValueError("weights must be finite numbers")
        # Half-away-from-zero without adding 0.5 (which misrounds large floats).
        fraction, whole = math.modf(value)
        nearest = int(whole) + (int(math.copysign(1, value)) if abs(fraction) >= 0.5 else 0)
        snapped = nearest if abs(value - nearest) < eps else value
        if snapped != int(snapped):
            raise Unresolved("noninteger weights remain after snapping; exact integer certification refused")
        return int(snapped)

    model = {key: [[snap(v) for v in row] for row in weights[key]] if key.startswith("W_")
             else [snap(v) for v in weights[key]] for key in KEYS}
    return model, hashlib.sha256(raw).hexdigest()


def step(model: dict, h: tuple[int, ...], bit: int) -> tuple[int, ...]:
    return tuple(max(0, sum(a * b for a, b in zip(row, h)) + model["W_hx"][i][bit] + model["b_h"][i])
                 for i, row in enumerate(model["W_hh"]))


def output(model: dict, h: tuple[int, ...]) -> int:
    scores = [sum(a * b for a, b in zip(row, h)) + bias for row, bias in zip(model["W_y"], model["b_y"])]
    return max(range(len(scores)), key=scores.__getitem__)


def enumerate_states(model: dict, limit: int, cap: int | None = None) -> dict:
    initial = (0,) * len(model["b_h"])
    states, ids, transitions = [initial], {initial: 0}, []
    for h in states:
        row = []
        for bit in (0, 1):
            nxt = step(model, h, bit)
            if cap is not None:
                nxt = tuple(min(v, cap) for v in nxt)
            if nxt not in ids:
                if len(states) >= limit:
                    raise Unresolved(f"reachable-state limit {limit} reached before closure")
                ids[nxt] = len(states)
                states.append(nxt)
            row.append(ids[nxt])
        transitions.append(row)
    return {"states": [list(h) for h in states], "transitions": transitions,
            "outputs": [output(model, h) for h in states]}


def certify_cap(model: dict, cap: int, timeout_ms: int) -> tuple[str, dict | None]:
    import z3
    h = [z3.Int(f"h{i}") for i in range(len(model["b_h"]))]
    representative = [z3.If(v > cap, cap, v) for v in h]

    def symbolic_step(xs, bit):
        values = [z3.Sum([a * x for a, x in zip(row, xs)]) + model["W_hx"][i][bit] + model["b_h"][i]
                  for i, row in enumerate(model["W_hh"])]
        return [z3.If(v > 0, v, 0) for v in values]

    def symbolic_output(xs):
        scores = [z3.Sum([a * x for a, x in zip(row, xs)]) + bias
                  for row, bias in zip(model["W_y"], model["b_y"])]
        best, value = z3.IntVal(0), scores[0]
        for i, score in enumerate(scores[1:], 1):
            best = z3.If(score > value, i, best)
            value = z3.If(score > value, score, value)
        return best

    mismatch = [symbolic_output(h) != symbolic_output(representative)]
    for bit in (0, 1):
        raw = symbolic_step(h, bit)
        abstract = symbolic_step(representative, bit)
        mismatch.extend(z3.If(a > cap, cap, a) != z3.If(b > cap, cap, b) for a, b in zip(raw, abstract))
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*(v >= 0 for v in h), z3.Or(mismatch))
    result = solver.check()
    if result == z3.unsat:
        return "unsat", None
    if result == z3.sat:
        witness = solver.model()
        return "sat", {str(v): witness.eval(v, model_completion=True).as_long() for v in h}
    return "unknown", {"reason": solver.reason_unknown()}


def minimize(machine: dict) -> dict:
    """Moore partition refinement, then BFS-renumber the reachable quotient."""
    transitions, outputs = machine["transitions"], machine["outputs"]
    colors = list(outputs)
    while True:
        signatures, refined = {}, []
        for i, value in enumerate(outputs):
            sig = (value, tuple(colors[n] for n in transitions[i]))
            if sig not in signatures:
                signatures[sig] = len(signatures)
            refined.append(signatures[sig])
        if refined == colors:
            break
        colors = refined
    reps = {}
    for i, color in enumerate(colors):
        reps.setdefault(color, i)
    order, renamed = [colors[0]], {colors[0]: 0}
    table, labels = [], []
    for color in order:
        rep = reps[color]
        row = []
        for nxt in transitions[rep]:
            group = colors[nxt]
            if group not in renamed:
                renamed[group] = len(order)
                order.append(group)
            row.append(renamed[group])
        table.append(row)
        labels.append(outputs[rep])
    return {"initial": 0, "transitions": table, "outputs": labels,
            "state_partition": [renamed[c] for c in colors]}


SPECS = {
    "divisible_by_3": {"transitions": [[0, 1], [2, 0], [1, 2]], "outputs": [1, 0, 0]},
    "contains_11": {"transitions": [[0, 1], [0, 2], [2, 2]], "outputs": [0, 0, 1]},
    "no_consecutive_1": {"transitions": [[0, 1], [0, 2], [2, 2]], "outputs": [1, 1, 0]},
    "parity": {"transitions": [[0, 1], [1, 0]], "outputs": [0, 1]},
}


def compare_spec(machine: dict, name: str) -> dict:
    """Complete product search; the first mismatch is a shortest witness."""
    spec = SPECS[name]
    initial = (0, 0)
    queue, parents = deque([initial]), {initial: None}
    while queue:
        state = queue.popleft()
        left, right = state
        actual, expected = machine["outputs"][left], spec["outputs"][right]
        if actual != expected:
            bits = []
            cursor = state
            while parents[cursor] is not None:
                cursor, bit = parents[cursor]
                bits.append(str(bit))
            return {"name": name, "equivalent": False, "shortest_counterexample": "".join(reversed(bits)),
                    "actual": actual, "expected": expected, "product_states_visited": len(parents)}
        for bit in (0, 1):
            nxt = (machine["transitions"][left][bit], spec["transitions"][right][bit])
            if nxt not in parents:
                parents[nxt] = (state, bit)
                queue.append(nxt)
    return {"name": name, "equivalent": True, "product_states_visited": len(parents)}


def arithmetic_bridge(model: dict, machine: dict, proof: dict) -> dict:
    """Sufficient, order-independent bound for exact f64 and i64 evaluation."""
    if proof["method"] != "exact_reachable_closure":
        return {"status": "not_established", "reason": "raw hidden states are not bounded by the saturation quotient"}
    values = [v for key in KEYS for row in model[key]
              for v in (row if isinstance(row, list) else [row])]
    bound = max(abs(v) for v in values)
    for h in machine["states"]:
        bound = max(bound, *(abs(v) for v in h))
        for i, row in enumerate(model["W_hh"]):
            # Sum of absolute terms bounds every product and every partial sum,
            # independent of the runtime's accumulation order.
            for bit in (0, 1):
                bound = max(bound, sum(abs(a * b) for a, b in zip(row, h))
                            + abs(model["W_hx"][i][bit]) + abs(model["b_h"][i]))
        for row, bias in zip(model["W_y"], model["b_y"]):
            bound = max(bound, sum(abs(a * b) for a, b in zip(row, h)) + abs(bias))
    return {"status": "exact_f64_and_i64" if bound <= 2**53 else "not_established",
            "absolute_operation_bound": bound, "sufficient_exact_integer_bound": 2**53}


def extract(path: Path, eps: float = 0.15, limit: int = 256, max_cap: int = 8,
            timeout_ms: int = 5000, spec: str | None = None) -> dict:
    if limit < 1 or max_cap < 0 or timeout_ms < 1:
        raise ValueError("state limit and timeout must be positive; max cap nonnegative")
    model, digest = load_model(path, eps)
    attempts = []
    try:
        machine = enumerate_states(model, limit)
        proof = {"method": "exact_reachable_closure"}
    except Unresolved:
        for cap in range(max_cap + 1):
            status, witness = certify_cap(model, cap, timeout_ms)
            attempts.append({"cap": cap, "solver_result": status, "witness": witness})
            if status == "unsat":
                machine = enumerate_states(model, limit, cap)
                proof = {"method": "saturation_homomorphism", "cap": cap, "solver_result": status}
                break
        else:
            raise Unresolved("exact exploration did not close and no saturation homomorphism was proved; no certificate emitted")
    minimal = minimize(machine)
    result = {"schema": 1, "model_sha256": digest, "epsilon": eps, "domain": DOMAIN,
              "proof": proof, "abstraction_attempts": attempts, "reachable": machine, "minimal": minimal}
    result["arithmetic_bridge"] = arithmetic_bridge(model, machine, proof)
    if spec is not None:
        result["specification"] = compare_spec(minimal, spec)
    return result


def check_certificate(path: Path, certificate: dict, timeout_ms: int = 5000,
                      original: Path | None = None) -> None:
    if certificate["schema"] != 1 or certificate["domain"] != DOMAIN:
        raise ValueError("unsupported certificate schema or domain")
    model, digest = load_model(path, certificate["epsilon"])
    if digest != certificate["model_sha256"]:
        raise ValueError("certificate belongs to different model bytes")
    proof = certificate["proof"]
    cap = None
    if proof["method"] == "saturation_homomorphism":
        cap = proof["cap"]
        if not isinstance(cap, int) or cap < 0:
            raise ValueError("invalid saturation cap")
        status, _ = certify_cap(model, cap, timeout_ms)
        if status != "unsat":
            raise Unresolved(f"saturation proof replay returned {status}")
    elif proof["method"] != "exact_reachable_closure":
        raise ValueError("unknown proof method")
    # Recompute closure and minimization from weights, not from claimed edges.
    fresh = enumerate_states(model, len(certificate["reachable"]["states"]), cap)
    if fresh != certificate["reachable"] or minimize(fresh) != certificate["minimal"]:
        raise ValueError("certificate transition/output/partition mismatch")
    if arithmetic_bridge(model, fresh, proof) != certificate["arithmetic_bridge"]:
        raise ValueError("arithmetic bound mismatch")
    if "specification" in certificate:
        spec = certificate["specification"]
        if compare_spec(certificate["minimal"], spec["name"]) != spec:
            raise ValueError("specification comparison mismatch")
    if "repair_search" in certificate:
        if original is None:
            raise ValueError("repair certificate replay requires --original source weights")
        source, source_digest = load_model(original, certificate["epsilon"])
        recorded = certificate["repair_search"]
        if source_digest != recorded["original_sha256"]:
            raise ValueError("repair source model digest mismatch")
        repaired, evidence = repair_one(source, certificate["specification"]["name"],
                                        recorded["radius"], recorded["state_limit"])
        evidence["original_sha256"] = source_digest
        if repaired != model or evidence != recorded:
            raise ValueError("repair search replay mismatch")


def emit_python(machine: dict) -> str:
    return ("# Exact finite-state implementation of the certified integer model.\n"
            "# Input: binary symbols (0 or 1), not one-hot vectors.\n"
            f"TRANSITIONS = {machine['transitions']!r}\n"
            f"OUTPUTS = {machine['outputs']!r}\n\n"
            "def decompiled(bits):\n"
            "    state = 0\n"
            "    for bit in bits:\n"
            "        if bit not in (0, 1):\n"
            "            raise ValueError('expected binary symbols')\n"
            "        state = TRANSITIONS[state][bit]\n"
            "    return OUTPUTS[state]\n")


def product_check(model: dict, spec: dict, limit: int) -> tuple[str, int]:
    """Search network x specification until mismatch, exact closure, or limit."""
    initial = ((0,) * len(model["b_h"]), 0)
    pending, seen = deque([initial]), {initial}
    while pending:
        h, state = pending.popleft()
        if output(model, h) != spec["outputs"][state]:
            return "mismatch", len(seen)
        for bit in (0, 1):
            nxt = (step(model, h, bit), spec["transitions"][state][bit])
            if nxt not in seen:
                if len(seen) >= limit:
                    return "unresolved", len(seen)
                seen.add(nxt)
                pending.append(nxt)
    return "equivalent", len(seen)


def repair_one(model: dict, name: str, radius: int, limit: int) -> tuple[dict, dict]:
    """Exhaustively search a declared one-coefficient integer edit neighborhood."""
    if radius < 1 or limit < 1:
        raise ValueError("repair radius and state limit must be positive")
    baseline, _ = product_check(model, SPECS[name], limit)
    if baseline == "equivalent":
        raise ValueError("model already matches specification; no repair needed")
    counts = {"mismatch": 0, "unresolved": 0, "equivalent": 0}
    best = None
    for key in KEYS:
        coords = ([(i, j) for i, row in enumerate(model[key]) for j in range(len(row))]
                  if key.startswith("W_") else [(i,) for i in range(len(model[key]))])
        for coord in coords:
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                candidate = copy.deepcopy(model)
                leaf = candidate[key]
                for index in coord[:-1]:
                    leaf = leaf[index]
                before = leaf[coord[-1]]
                leaf[coord[-1]] += delta
                verdict, product_states = product_check(candidate, SPECS[name], limit)
                counts[verdict] += 1
                if verdict != "equivalent":
                    continue
                # Prefer a smaller edit, then deleting an edge over adding one,
                # then a smaller certified state space. Ties follow file order.
                score = (abs(delta), int(before + delta != 0) - int(before != 0), product_states)
                if best is None or score < best[0]:
                    best = (score, candidate, {"parameter": key, "index": list(coord),
                            "before": before, "after": before + delta,
                            "product_states": product_states})
    if best is None:
        raise Unresolved(f"no certified one-coefficient repair in radius {radius}: {counts}")
    return best[1], {"edit": best[2], "radius": radius, "state_limit": limit, "candidate_counts": counts,
                     "scope": "one edit to the snapped integer model; exact product-closure certification"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--max-states", type=int, default=256)
    parser.add_argument("--max-cap", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--spec", choices=SPECS)
    parser.add_argument("--certificate", type=Path, help="write replayable evidence JSON")
    parser.add_argument("--emit", type=Path, help="write the minimized executable Python machine")
    parser.add_argument("--check", type=Path, help="replay an existing certificate against these model bytes")
    parser.add_argument("--repair-one", type=Path, help="search a certified one-coefficient repair and write NEW weight JSON")
    parser.add_argument("--radius", type=int, default=1, help="integer edit radius for --repair-one")
    parser.add_argument("--original", type=Path, help="original model for replaying a repair certificate")
    args = parser.parse_args()
    try:
        if args.check:
            check_certificate(args.model, json.loads(args.check.read_text()), args.timeout_ms, args.original)
            print("Certificate replay: verified")
            return 0
        repair = None
        path = args.model
        if args.repair_one:
            if args.spec is None:
                raise ValueError("--repair-one requires --spec")
            if args.repair_one.resolve() == args.model.resolve():
                raise ValueError("repair output must not overwrite the original model")
            model, original_digest = load_model(args.model, args.eps)
            repaired, repair = repair_one(model, args.spec, args.radius, args.max_states)
            repair["original_sha256"] = original_digest
            args.repair_one.parent.mkdir(parents=True, exist_ok=True)
            args.repair_one.write_text(json.dumps(repaired, indent=2) + "\n")
            path = args.repair_one
        result = extract(path, args.eps, args.max_states, args.max_cap, args.timeout_ms, args.spec)
        if repair is not None:
            result["repair_search"] = repair
        if args.certificate:
            args.certificate.parent.mkdir(parents=True, exist_ok=True)
            args.certificate.write_text(json.dumps(result, indent=2) + "\n")
        if args.emit:
            args.emit.parent.mkdir(parents=True, exist_ok=True)
            args.emit.write_text(emit_python(result["minimal"]))
        print(json.dumps({"proof": result["proof"], "reachable_states": len(result["reachable"]["states"]),
                          "minimal_states": len(result["minimal"]["outputs"]),
                          "specification": result.get("specification"), "repair_search": repair,
                          "arithmetic_bridge": result["arithmetic_bridge"],
                          "domain": DOMAIN}, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(f"Not certified: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
