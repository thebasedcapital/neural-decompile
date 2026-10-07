#!/usr/bin/env python3
"""Train small ReLU RNNs on algorithmic tasks and export them in `nd`'s JSON format.

Run (deterministic; CPU is enough):

    uv run --with torch --with numpy scripts/train_rnn_tasks.py            # all tasks
    uv run --with torch --with numpy scripts/train_rnn_tasks.py max4 xor   # subset

Model (exactly what `nd` executes, see src/fsm.rs):

    h_0 = 0
    h_t = ReLU(W_hh h_{t-1} + W_hx x_t + b_h)
    y   = argmax(W_y h_T + b_y)           # only the last step is read out

Training = cross-entropy on the final output + an **integer-pulling L1 penalty**
    lambda(t) * sum_w |w - round(w)|
whose weight is ramped up over training (see README "How It Works"). The weights
exported to JSON are the raw trained floats: nothing is snapped or rounded here.
Quantization is left to `nd` (default eps = 0.15).

Tasks (one-hot inputs; every task is a *finite* function, so the exhaustive test
set is also the training set -- this is decompilation of a trained circuit, not a
generalisation benchmark):

  parity5         seq of 5 bits, in-dim 2. label = (#ones) mod 2.            all 2^5 = 32 strings.
  evens_detector  bit strings of length 1..5, in-dim 2, MSB first.           all 62 strings.
                  label = 1 iff the binary number they spell is even (last bit is 0).
  max4            two digits in {0..3}, in-dim 4, out-dim 4. label = max(a,b).   all 16 pairs.
  max5            two digits in {0..4}, in-dim 5, out-dim 5. label = max(a,b).   all 25 pairs.
  bitwise_xor     two 2-bit numbers a,b in {0..3} (one-hot over 4 values per
                  step), out-dim 4. label = a XOR b.                          all 16 pairs.
  mod5_add        two digits in {0..4}, in-dim 5, out-dim 5. label = (a+b) mod 5. all 25 pairs.

README claims hidden dims 2/2/2/2/3 for the first five; each task below lists
candidate hidden sizes, tried in order, and the first size for which at least one
seed reaches 100% float accuracy is exported (so a larger size is used, and
logged, if the README's size does not train here).

Seed selection uses only training-side quantities (float accuracy, then fraction
of weights within 0.15 of an integer); `nd verify` output is never consulted.
"""

import itertools
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
EPS = 0.15  # nd's default quantization epsilon (used only for seed ranking)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Task definitions: each returns (list[(tuple[int,...] digits, int label)], in_dim, out_dim)
# ---------------------------------------------------------------------------
def task_parity5():
    data = [(s, sum(s) % 2) for s in itertools.product(range(2), repeat=5)]
    return data, 2, 2


def task_evens_detector():
    data = []
    for n in range(1, 6):
        for s in itertools.product(range(2), repeat=n):
            data.append((s, 1 if s[-1] == 0 else 0))
    return data, 2, 2


def _pairs(k, f):
    return [((a, b), f(a, b)) for a in range(k) for b in range(k)], k, None


def task_max4():
    d, i, _ = _pairs(4, max)
    return d, i, 4


def task_max5():
    d, i, _ = _pairs(5, max)
    return d, i, 5


def task_bitwise_xor():
    d, i, _ = _pairs(4, lambda a, b: a ^ b)
    return d, i, 4


def task_mod5_add():
    d, i, _ = _pairs(5, lambda a, b: (a + b) % 5)
    return d, i, 5


# name -> (builder, candidate hidden sizes, number of seeds, steps)
TASKS = {
    "parity5": (task_parity5, [2, 3, 4], 24),
    "evens_detector": (task_evens_detector, [2, 3], 24),
    "max4": (task_max4, [2, 3, 4], 24),
    "max5": (task_max5, [2, 3, 4], 24),
    "bitwise_xor": (task_bitwise_xor, [3, 4, 5, 6], 24),
    "mod5_add": (task_mod5_add, [3, 4, 5, 6], 24),
}


# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
def one_hot_batches(data, in_dim):
    """Group sequences by length -> list of (X[N,L,in], y[N])."""
    groups = {}
    for seq, y in data:
        groups.setdefault(len(seq), []).append((seq, y))
    out = []
    for L, items in sorted(groups.items()):
        X = torch.zeros(len(items), L, in_dim)
        Y = torch.zeros(len(items), dtype=torch.long)
        for n, (seq, y) in enumerate(items):
            for t, d in enumerate(seq):
                X[n, t, d] = 1.0
            Y[n] = y
        out.append((X, Y))
    return out


def forward(P, X):
    W_hh, W_hx, b_h, W_y, b_y = P
    h = torch.zeros(X.shape[0], W_hh.shape[0])
    for t in range(X.shape[1]):
        h = torch.relu(h @ W_hh.T + X[:, t] @ W_hx.T + b_h)
    return h @ W_y.T + b_y


def int_penalty(P):
    return sum((w - torch.round(w).detach()).abs().sum() for w in P)


def accuracy(P, batches):
    ok = tot = 0
    with torch.no_grad():
        for X, Y in batches:
            ok += (forward(P, X).argmax(1) == Y).sum().item()
            tot += len(Y)
    return ok / tot


def frac_near_int(P, eps=EPS):
    flat = torch.cat([w.detach().flatten() for w in P])
    return ((flat - flat.round()).abs() < eps).double().mean().item()


def train_one(batches, in_dim, out_dim, hidden, seed, steps=6000):
    g = torch.Generator().manual_seed(seed)
    s = 0.5
    P = [
        (torch.randn(hidden, hidden, generator=g) * s).requires_grad_(),
        (torch.randn(hidden, in_dim, generator=g) * s).requires_grad_(),
        (torch.randn(hidden, generator=g) * 0.1).requires_grad_(),
        (torch.randn(out_dim, hidden, generator=g) * s).requires_grad_(),
        torch.zeros(out_dim).requires_grad_(),
    ]
    opt = torch.optim.Adam(P, lr=0.02)
    warm = steps // 4  # pure CE (+ tiny L1 for sparsity)
    lam_max = 0.02
    for step in range(steps):
        opt.zero_grad()
        ce = 0.0
        n = 0
        for X, Y in batches:
            ce = ce + torch.nn.functional.cross_entropy(forward(P, X), Y, reduction="sum")
            n += len(Y)
        ce = ce / n
        if step < warm:
            lam, l1 = 0.0, 1e-3 * sum(w.abs().sum() for w in P)
        else:
            frac = (step - warm) / max(1, steps - warm)
            lam, l1 = lam_max * frac, 0.0
        loss = ce + l1 + (lam * int_penalty(P) if lam > 0 else 0.0)
        loss.backward()
        opt.step()
        if step == int(steps * 0.85):
            for gr in opt.param_groups:
                gr["lr"] = 0.005
    return P


def export(P, tests_data, in_dim, name):
    W_hh, W_hx, b_h, W_y, b_y = [w.detach().numpy() for w in P]
    weights = {
        "W_hh": W_hh.tolist(),
        "W_hx": W_hx.tolist(),
        "b_h": b_h.tolist(),
        "W_y": W_y.tolist(),
        "b_y": b_y.tolist(),
    }
    tests = []
    for seq, y in tests_data:
        inputs = []
        for d in seq:
            v = [0.0] * in_dim
            v[d] = 1.0
            inputs.append(v)
        tests.append({"inputs": inputs, "expected": int(y)})
    (EXAMPLES / f"{name}.json").write_text(json.dumps(weights, indent=2) + "\n")
    (EXAMPLES / f"{name}_tests.json").write_text(json.dumps(tests, indent=2) + "\n")


def _seed_job(args):
    name, hidden, seed = args
    builder, _, _ = TASKS[name]
    data, in_dim, out_dim = builder()
    batches = one_hot_batches(data, in_dim)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    P = train_one(batches, in_dim, out_dim, hidden, seed)
    return seed, accuracy(P, batches), frac_near_int(P), [w.detach().numpy() for w in P]


def run_task(name, log, pool):
    builder, hiddens, n_seeds = TASKS[name]
    data, in_dim, out_dim = builder()
    log(f"== {name}: {len(data)} exhaustive cases, in={in_dim} out={out_dim}")
    for hidden in hiddens:
        t0 = time.time()
        results = pool.map(_seed_job, [(name, hidden, s) for s in range(n_seeds)])
        for seed, acc, fni, _ in results:
            log(f"   hidden={hidden} seed={seed:2d} float_acc={acc:.3f} near_int={fni:.2f}")
        perfect = [r for r in results if r[1] == 1.0]
        log(f"   hidden={hidden}: {len(perfect)}/{n_seeds} seeds reach 100% float accuracy "
            f"({time.time() - t0:.0f}s)")
        if perfect:
            seed, acc, fni, P = max(perfect, key=lambda r: (r[2], -r[0]))
            log(f"   -> exporting hidden={hidden} seed={seed} (near_int={fni:.2f})")
            export([torch.from_numpy(w) for w in P], data, in_dim, name)
            return True
    log(f"   !! no hidden size in {hiddens} reached 100% float accuracy; nothing exported for {name}")
    return False


def main():
    names = sys.argv[1:] or list(TASKS)
    alias = {"xor": "bitwise_xor"}
    names = [alias.get(n, n) for n in names]
    bad = [n for n in names if n not in TASKS]
    if bad:
        sys.exit(f"unknown task(s) {bad}; choose from {list(TASKS)}")
    log_dir = EXAMPLES / "train_logs"
    log_dir.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(min(8, mp.cpu_count())) as pool:
        for n in names:
            lines = []

            def log(msg):
                print(msg, flush=True)
                lines.append(msg)

            run_task(n, log, pool)
            (log_dir / f"{n}.log").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
