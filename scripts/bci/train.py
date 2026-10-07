#!/usr/bin/env python3
"""Train a tiny sparse ReLU RNN reach-direction decoder on MC_Maze_Small and export it for `nd`.

Pipeline (all selection decisions use TRAIN trials only; the held-out 'val' trials are touched once at the end):
  1. features: spike counts in BIN_MS bins over [WIN_LO, WIN_HI) ms around movement onset, for the top-k units
     ranked by ANOVA F (4 reach quadrants) on the train trials.
  2. model:   h_t = ReLU(W_hh h_{t-1} + W_hx x_t + b_h), y = argmax(W_y h_T + b_y)   (exactly nd's format)
  3. loss:    cross-entropy + L1 on all weights + a ramped "integer pull" |w - round(w)| (README recipe).
             Training augmentation: the 50 ms bin grid is shifted by 0..40 ms (train trials only).
  4. hyper-parameter / seed choice: stratified 5-fold CV over the train trials, scored on the *quantized*
             (eps-snapped) model, i.e. the thing nd will actually run.
  5. final fit on all train trials, export JSON weights + test cases from the held-out trials.
Quantization here mirrors nd (`quantize_val`): a weight within eps of an integer is snapped, others stay float.
"""
import argparse, copy, itertools, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples" / "bci"
torch.set_num_threads(1)
_G = {}  # (Data, args) shared with forked workers


def _job(a):
    tr, te, k, H, l1, ip, seed = a
    r = train_eval(_G['D'], tr, te, _G['args'], k, H, l1, ip, seed)
    return r


def pmap(jobs):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(_G['args'].workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


# ---------------------------------------------------------------- data
def make_features(counts, units, lo_ms, hi_ms, bin_ms, pre_ms, fine_ms, shift_bins=0):
    """counts [trial, unit, fine_bin] -> [trial, T, len(units)] summed into bin_ms bins."""
    f = bin_ms // fine_ms
    lo = (pre_ms + lo_ms) // fine_ms + shift_bins
    T = (hi_ms - lo_ms) // bin_ms
    x = counts[:, units, lo:lo + T * f].astype(np.float32)
    x = x.reshape(x.shape[0], len(units), T, f).sum(3)
    return np.ascontiguousarray(x.transpose(0, 2, 1))


def select_units(counts_tr, y_tr, k, lo_ms, hi_ms, pre_ms, fine_ms):
    lo, hi = (pre_ms + lo_ms) // fine_ms, (pre_ms + hi_ms) // fine_ms
    tot = np.sqrt(counts_tr[:, :, lo:hi].sum(2).astype(np.float64))  # [trial, unit]
    gm = tot.mean(0)
    ss_b = sum((y_tr == c).sum() * (tot[y_tr == c].mean(0) - gm) ** 2 for c in np.unique(y_tr))
    ss_w = sum(((tot[y_tr == c] - tot[y_tr == c].mean(0)) ** 2).sum(0) for c in np.unique(y_tr))
    F = (ss_b / (len(np.unique(y_tr)) - 1)) / (ss_w / (len(y_tr) - len(np.unique(y_tr))) + 1e-9)
    return np.argsort(-F)[:k], F


# ---------------------------------------------------------------- model
class Rnn(torch.nn.Module):
    def __init__(self, k, H, n_out, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_hh = torch.nn.Parameter(torch.randn(H, H, generator=g) * 0.1)
        self.W_hx = torch.nn.Parameter(torch.randn(H, k, generator=g) * 0.3)
        self.b_h = torch.nn.Parameter(torch.zeros(H))
        self.W_y = torch.nn.Parameter(torch.randn(n_out, H, generator=g) * 0.3)
        self.b_y = torch.nn.Parameter(torch.zeros(n_out))

    def params(self):
        return [self.W_hh, self.W_hx, self.b_h, self.W_y, self.b_y]

    def forward(self, x):  # x [B, T, k]
        h = x.new_zeros(x.shape[0], self.W_hh.shape[0])
        for t in range(x.shape[1]):
            h = torch.relu(h @ self.W_hh.T + x[:, t] @ self.W_hx.T + self.b_h)
        return h @ self.W_y.T + self.b_y


def quantize(p, eps):
    return {n: np.where(np.abs(v - np.round(v)) < eps, np.round(v), v) for n, v in p.items()}


def np_predict(p, x):
    """float64 forward pass, same arithmetic as nd's fsm::run_fsm."""
    H = p["b_h"].shape[0]
    preds = []
    for seq in x:
        h = np.zeros(H)
        for xt in seq.astype(np.float64):
            h = np.maximum(p["W_hh"] @ h + p["W_hx"] @ xt + p["b_h"], 0.0)
        preds.append(int(np.argmax(p["W_y"] @ h + p["b_y"])))  # first max wins, like nd
    return np.array(preds)


def get_params(model):
    return {n: getattr(model, n).detach().double().numpy().copy() for n in ["W_hh", "W_hx", "b_h", "W_y", "b_y"]}


def fit(x_aug, y_aug, k, H, n_out, seed, l1, int_pull, epochs, lr=0.02, eps=0.15):
    """Full-batch Adam. L1 constant, integer pull ramps up over the last 2/3 of training.
    Final phase: snap everything within eps and fine-tune only the remaining non-integer weights."""
    torch.manual_seed(seed)
    m = Rnn(k, H, n_out, seed)
    X = torch.tensor(x_aug)
    Y = torch.tensor(y_aug)
    opt = torch.optim.Adam(m.params(), lr=lr)
    n_w = sum(p.numel() for p in m.params())
    for ep in range(epochs):
        ramp = min(1.0, max(0.0, (ep / epochs - 0.33) / 0.5))
        loss = torch.nn.functional.cross_entropy(m(X), Y)
        reg_l1 = sum(p.abs().sum() for p in m.params()) / n_w
        reg_int = sum((p - p.detach().round()).abs().sum() for p in m.params()) / n_w
        total = loss + l1 * reg_l1 + int_pull * ramp * reg_int
        opt.zero_grad()
        total.backward()
        opt.step()
    pre = copy.deepcopy(m)  # the plain float model (L1 + integer pull only), before any snapping
    # snap-and-freeze fine-tune: repeat a few rounds
    for _ in range(3):
        with torch.no_grad():
            masks = []
            for p in m.params():
                near = (p - p.round()).abs() < eps
                p[near] = p[near].round()
                masks.append(~near)
        opt = torch.optim.Adam(m.params(), lr=lr / 4)
        for _ in range(150):
            loss = torch.nn.functional.cross_entropy(m(X), Y)
            opt.zero_grad()
            loss.backward()
            for p, mk in zip(m.params(), masks):
                p.grad *= mk
            opt.step()
    with torch.no_grad():
        final_loss = torch.nn.functional.cross_entropy(m(X), Y).item()
    return m, final_loss, pre


# ---------------------------------------------------------------- experiment helpers
class Data:
    def __init__(self, path, args):
        d = np.load(path)
        self.counts, self.label, self.split = d["counts"], d["label"], d["split"]
        self.unit_id, self.maze = d["unit_id"], d["maze_id"]
        self.pre_ms, self.fine_ms = int(d["pre_ms"]), int(d["bin_ms"])
        self.args = args

    def feats(self, idx, units, shift=0):
        a = self.args
        return make_features(self.counts[idx], units, a.win_lo, a.win_hi, a.bin_ms, self.pre_ms, self.fine_ms, shift)

    def aug(self, idx, units):
        shifts = range(0, self.args.bin_ms // self.fine_ms)
        xs = [self.feats(idx, units, s) for s in shifts]
        return np.concatenate(xs), np.concatenate([self.label[idx]] * len(xs))


def train_eval(D, tr, te, args, k, H, l1, int_pull, seed, quiet=True):
    units, _ = select_units(D.counts[tr], D.label[tr], k, args.win_lo, args.win_hi, D.pre_ms, D.fine_ms)
    xa, ya = D.aug(tr, units)
    m, loss, pre = fit(xa, ya, k, H, 4, seed, l1, int_pull, args.epochs)
    p = get_params(m)
    q = quantize(p, args.eps)
    x_tr, x_te = D.feats(tr, units), D.feats(te, units)
    with torch.no_grad():
        f_tr = m(torch.tensor(x_tr)).argmax(1).numpy()
        f_te = m(torch.tensor(x_te)).argmax(1).numpy()
    q_tr, q_te = np_predict(q, x_tr), np_predict(q, x_te)
    with torch.no_grad():
        pf_te = pre(torch.tensor(x_te)).argmax(1).numpy()
    pq_te = np_predict(quantize(get_params(pre), args.eps), x_te)  # snap the pre-snap model, no fine-tune
    return dict(pre_float_test=(pf_te == D.label[te]).mean(), pre_quant_test=(pq_te == D.label[te]).mean(),
                pre_agree_test=(pf_te == q_te).mean(), pf_te=pf_te,
                pre_pct_int=pct_integer(quantize(get_params(pre), args.eps))[0], units=units, model=m, params=p, qparams=q, loss=loss,
                float_train=(f_tr == D.label[tr]).mean(), quant_train=(q_tr == D.label[tr]).mean(),
                float_test=(f_te == D.label[te]).mean(), quant_test=(q_te == D.label[te]).mean(),
                agree_test=(f_te == q_te).mean(), f_te=f_te, q_te=q_te)


def pct_integer(q, eps=0.01):
    allw = np.concatenate([v.ravel() for v in q.values()])
    return float((np.abs(allw - np.round(allw)) < eps).mean()), allw.size


def stratified_folds(y, n, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i, j in enumerate(idx):
            folds[i % n].append(j)
    return [np.array(sorted(f)) for f in folds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(OUT / "data" / "trials.npz"))
    ap.add_argument("--win-lo", type=int, default=-100, help="window start, ms relative to movement onset")
    ap.add_argument("--win-hi", type=int, default=300)
    ap.add_argument("--bin-ms", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.15, help="nd quantization epsilon (nd default)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--hs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--l1s", type=float, nargs="+", default=[0.02, 0.1])
    ap.add_argument("--int-pulls", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--cv-seeds", type=int, default=1)
    ap.add_argument("--final-seeds", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=0, help="extra random 75/25 resplits for a robustness estimate")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = Data(args.data, args)
    _G.update(D=D, args=args)
    tr = np.where(D.split == "train")[0]
    te = np.where(D.split == "val")[0]
    y = D.label
    print(f"train trials {len(tr)}, held-out trials {len(te)}; held-out class counts {np.bincount(y[te], minlength=4).tolist()}, "
          f"majority-class chance {np.bincount(y[te]).max()/len(te):.3f} (uniform 0.25)", flush=True)

    # ---- 4. hyper-parameter selection by CV on TRAIN trials only (quantized accuracy)
    folds = stratified_folds(y[tr], 5, 0)
    grid = list(itertools.product(args.ks, args.hs, args.l1s, args.int_pulls))
    cv_rows = []
    t0 = time.time()
    jobs = [(np.delete(tr, f), tr[f], k, H, l1, ip, 100 + sd) for (k, H, l1, ip) in grid for f in folds for sd in range(args.cv_seeds)]
    res = pmap(jobs)
    per = len(folds) * args.cv_seeds
    for gi, (k, H, l1, ip) in enumerate(grid):
        rr = res[gi * per:(gi + 1) * per]
        a = (np.mean([r["float_test"] for r in rr]), np.mean([r["quant_test"] for r in rr]))
        cv_rows.append(dict(k=k, H=H, l1=l1, int_pull=ip, cv_float=float(a[0]), cv_quant=float(a[1])))
        print(f"  CV k={k:2d} H={H} l1={l1:<5} pull={ip:<4} float {a[0]:.3f} quant {a[1]:.3f}", flush=True)
    print(f"  CV done in {time.time()-t0:.0f}s", flush=True)
    best = max(cv_rows, key=lambda r: (r["cv_quant"], r["cv_float"], -r["H"], -r["k"]))
    print("selected by train-CV quantized accuracy:", best, flush=True)

    # ---- 5. final fit on all train trials; choose seed by TRAIN quantized accuracy then loss (no test peeking)
    cands = pmap([(tr, te, best["k"], best["H"], best["l1"], best["int_pull"], sd) for sd in range(args.final_seeds)])
    for s, r in enumerate(cands):
        print(f"  seed {s}: train quant {r['quant_train']:.3f} loss {r['loss']:.3f}", flush=True)
    order = sorted(range(len(cands)), key=lambda s: (-cands[s]["quant_train"], cands[s]["loss"]))
    seed = order[0]
    r = cands[seed]
    print(f"final seed {seed}", flush=True)

    k, units = best["k"], r["units"]
    q, p = r["qparams"], r["params"]
    # export the FLOAT weights; nd does the quantization itself (eps) — that is the decompiler's job
    weights = {"W_hh": p["W_hh"].tolist(), "W_hx": p["W_hx"].tolist(), "b_h": p["b_h"].tolist(),
               "W_y": p["W_y"].tolist(), "b_y": p["b_y"].tolist()}
    x_te = D.feats(te, units)
    tests = [{"inputs": x.astype(float).tolist(), "expected": int(lab)} for x, lab in zip(x_te, y[te])]
    (out / "reach_decoder.json").write_text(json.dumps(weights, indent=1))
    (out / "reach_decoder_tests.json").write_text(json.dumps(tests))
    pi, n_w = pct_integer(q)
    meta = dict(
        dataset="DANDI:000140 MC_Maze_Small (Neural Latents Benchmark), sub-Jenkins train+behavior file",
        task="reach quadrant of the active target: 0=left-down 1=left-up 2=right-down 3=right-up (sign of target x,y)",
        window_ms=[args.win_lo, args.win_hi], bin_ms=args.bin_ms, steps=(args.win_hi - args.win_lo) // args.bin_ms,
        input_index_to_dandi_unit_id=[int(D.unit_id[u]) for u in units],
        input_index_to_row_in_units_table=[int(u) for u in units],
        n_train=len(tr), n_test=len(te), test_trial_indices=te.tolist(),
        chance_majority=float(np.bincount(y[te]).max() / len(te)), chance_uniform=0.25,
        selected=best, final_seed=seed, eps=args.eps,
        pre_snap_float_test_acc=float(r["pre_float_test"]), pre_snap_quant_test_acc_no_finetune=float(r["pre_quant_test"]),
        pre_snap_float_vs_decompiled_agreement=float(r["pre_agree_test"]), pre_snap_pct_integer_eps_snap=float(r["pre_pct_int"]),
        pre_snap_float_test_pred=r["pf_te"].tolist(),
        float_train_acc=float(r["float_train"]), quant_train_acc=float(r["quant_train"]),
        float_test_acc=float(r["float_test"]), quant_test_acc_python_mirror=float(r["quant_test"]),
        float_quant_agreement_test=float(r["agree_test"]), pct_integer_quantized_eps0p01=pi, n_weights=n_w,
        float_test_pred=r["f_te"].tolist(), test_labels=y[te].tolist(), cv_grid=cv_rows)
    print(f"HELD-OUT: float {r['float_test']:.3f}  quantized(py mirror) {r['quant_test']:.3f}  agreement {r['agree_test']:.3f}  "
          f"%int {pi:.1%} of {n_w}", flush=True)

    # ---- optional: robustness over random resplits with the selected config (no re-selection, units re-ranked on train)
    if args.repeats:
        rs = []
        for i in range(args.repeats):
            rng = np.random.default_rng(1000 + i)
            idx = np.concatenate([rng.permutation(np.where(y == c)[0]) for c in range(4)])
            test_i = np.sort(np.concatenate([np.where(y == c)[0][rng.permutation((y == c).sum())[: max(1, round((y == c).sum() * 0.25))]] for c in range(4)]))
            train_i = np.setdiff1d(np.arange(len(y)), test_i)
            cs = pmap([(train_i, test_i, best["k"], best["H"], best["l1"], best["int_pull"], sd) for sd in range(3)])
            c = sorted(cs, key=lambda c: (-c["quant_train"], c["loss"]))[0]
            rs.append((c["float_test"], c["quant_test"], c["agree_test"], len(test_i), np.bincount(y[test_i]).max() / len(test_i)))
            print(f"  resplit {i}: float {rs[-1][0]:.3f} quant {rs[-1][1]:.3f} agree {rs[-1][2]:.3f} (n_test={len(test_i)})", flush=True)
        a = np.array(rs)
        meta["resplits"] = dict(n=args.repeats, float_mean=float(a[:, 0].mean()), float_std=float(a[:, 0].std()),
                                quant_mean=float(a[:, 1].mean()), quant_std=float(a[:, 1].std()),
                                agree_mean=float(a[:, 2].mean()), rows=a.tolist())
        print(f"RESPLITS x{args.repeats}: float {a[:,0].mean():.3f}±{a[:,0].std():.3f} quant {a[:,1].mean():.3f}±{a[:,1].std():.3f} agree {a[:,2].mean():.3f}", flush=True)
    (out / "reach_decoder_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
