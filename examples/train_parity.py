"""Find mod-2 parity transformer weights via random search."""

import json, math, random, time

SEED = 42
random.seed(SEED)

D_MODEL = 2
D_FF = 4
SEQ_LEN = 3
VOCAB_SIZE = 2
NUM_CLASSES = 2


def randn(n):
    return [
        math.sqrt(-2 * math.log(random.random() + 1e-10))
        * math.cos(2 * math.pi * random.random())
        for _ in range(n)
    ]


def rand_params(scale=0.5):
    return {
        "te0": [0.0, 0.0],
        "te1": [1.0, 0.0],
        "pe": [0.0] * (SEQ_LEN * D_MODEL),
        "wq": [v * scale for v in randn(D_MODEL * D_MODEL)],
        "wk": [v * scale for v in randn(D_MODEL * D_MODEL)],
        "wv": [v * scale for v in randn(D_MODEL * D_MODEL)],
        "wo": [v * scale for v in randn(D_MODEL * D_MODEL)],
        "wfi": [v * scale for v in randn(D_MODEL * D_FF)],
        "wfo": [v * scale for v in randn(D_FF * D_MODEL)],
        "wout": [v * scale for v in randn(D_MODEL * NUM_CLASSES)],
        "bout": [random.uniform(-0.5, 0.5) for _ in range(NUM_CLASSES)],
        "l1g": [1.0] * D_MODEL,
        "l1b": [0.0] * D_MODEL,
        "l2g": [1.0] * D_MODEL,
        "l2b": [0.0] * D_MODEL,
    }


def te(P, tok):
    return P["te" + str(tok)]


def mm(x, wf, in_d, out_d, bias=None):
    L = len(x)
    out = [[0.0] * out_d for _ in range(L)]
    for i in range(L):
        for j in range(out_d):
            s = bias[j] if bias else 0.0
            for k in range(in_d):
                s += x[i][k] * wf[k * out_d + j]
            out[i][j] = s
    return out


def ln(x, g, b):
    n = len(x)
    m = sum(x) / n
    v = sum((u - m) ** 2 for u in x) / n
    std = math.sqrt(v + 1e-5)
    return [(x[i] - m) / std * g[i] + b[i] for i in range(n)]


def forward(P, tokens):
    L = len(tokens)
    h = [
        [te(P, tokens[i])[j] + P["pe"][i * D_MODEL + j] for j in range(D_MODEL)]
        for i in range(L)
    ]

    xn = [ln(h[i], P["l1g"], P["l1b"]) for i in range(L)]
    q = mm(xn, P["wq"], D_MODEL, D_MODEL)
    k = mm(xn, P["wk"], D_MODEL, D_MODEL)
    v = mm(xn, P["wv"], D_MODEL, D_MODEL)

    hd = D_MODEL
    sc_ = [[0.0] * L for _ in range(L)]
    for i in range(L):
        for j in range(L):
            for d in range(hd):
                sc_[i][j] += q[i][d] * k[j][d]
            sc_[i][j] /= math.sqrt(hd)

    aw = []
    for row in sc_:
        m = max(row)
        ex = [math.exp(u - m) for u in row]
        s = sum(ex)
        aw.append([e / (s + 1e-10) for e in ex])

    ao = [[0.0] * D_MODEL for _ in range(L)]
    for i in range(L):
        for d in range(hd):
            for j in range(L):
                ao[i][d] += aw[i][j] * v[j][d]

    ap = mm(ao, P["wo"], D_MODEL, D_MODEL)
    for i in range(L):
        for j in range(D_MODEL):
            h[i][j] += ap[i][j]

    xn = [ln(h[i], P["l2g"], P["l2b"]) for i in range(L)]
    fh = mm(xn, P["wfi"], D_MODEL, D_FF)
    fh = [[max(0.0, u) for u in row] for row in fh]
    fo = mm(fh, P["wfo"], D_FF, D_MODEL)
    for i in range(L):
        for j in range(D_MODEL):
            h[i][j] += fo[i][j]

    logits = []
    for i in range(L):
        lg = list(P["bout"])
        for j in range(NUM_CLASSES):
            for kk in range(D_MODEL):
                lg[j] += h[i][kk] * P["wout"][kk * NUM_CLASSES + j]
        logits.append(lg)
    return logits


# Dataset
all_seqs = []
for i in range(2**SEQ_LEN):
    bits = [(i >> b) & 1 for b in range(SEQ_LEN)]
    all_seqs.append((bits, sum(bits) % 2))


def evaluate(P):
    correct = 0
    for tokens, label in all_seqs:
        lg = forward(P, tokens)[-1]
        if lg.index(max(lg)) == label:
            correct += 1
    return correct


# Brute force search
print(f"Searching for parity transformer weights ({SEQ_LEN}-bit)...")
t0 = time.time()
best_acc = 0
best_P = None

for trial in range(100000):
    P = rand_params(scale=0.3 + random.random() * 0.7)
    acc = evaluate(P)
    if acc > best_acc:
        best_acc = acc
        best_P = P
        elapsed = time.time() - t0
        print(f"  trial {trial}: acc={acc}/8 ({elapsed:.1f}s)")
        if acc == 8:
            print("  PERFECT!")
            break

    # Also try small perturbations of best
    if best_P and trial % 100 == 0 and trial > 0:
        for _ in range(20):
            P2 = {k: list(v) for k, v in best_P.items()}
            for k in ["wq", "wk", "wv", "wo", "wfi", "wfo", "wout", "bout"]:
                for i in range(len(P2[k])):
                    P2[k][i] += random.gauss(0, 0.1)
            acc2 = evaluate(P2)
            if acc2 > best_acc:
                best_acc = acc2
                best_P = P2
                elapsed = time.time() - t0
                print(f"  trial {trial}+refine: acc={acc2}/8 ({elapsed:.1f}s)")
                if acc2 == 8:
                    print("  PERFECT!")
                    break

P = best_P
print(f"\nBest accuracy: {best_acc}/8")

if best_acc == 8:
    print("\nAll predictions:")
    for tokens, label in all_seqs:
        lg = forward(P, tokens)[-1]
        p = lg.index(max(lg))
        print(f"  {tokens} → pred={p} exp={label} OK")

    # Export
    def reshape(lst, r, c):
        return [lst[i * c : (i + 1) * c] for i in range(r)]

    bp = best_P
    export = {
        "d_model": D_MODEL,
        "n_heads": 1,
        "d_ff": D_FF,
        "n_layers": 1,
        "max_seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "gelu": False,
        "token_emb": [bp["te0"], bp["te1"]],
        "pos_emb": reshape(bp["pe"], SEQ_LEN, D_MODEL),
        "layers": [
            {
                "w_q": reshape(bp["wq"], D_MODEL, D_MODEL),
                "w_k": reshape(bp["wk"], D_MODEL, D_MODEL),
                "w_v": reshape(bp["wv"], D_MODEL, D_MODEL),
                "w_o": reshape(bp["wo"], D_MODEL, D_MODEL),
                "w_ff_in": reshape(bp["wfi"], D_MODEL, D_FF),
                "w_ff_out": reshape(bp["wfo"], D_FF, D_MODEL),
                "ln1_gamma": bp["l1g"],
                "ln1_beta": bp["l1b"],
                "ln2_gamma": bp["l2g"],
                "ln2_beta": bp["l2b"],
            }
        ],
        "w_out": reshape(bp["wout"], D_MODEL, NUM_CLASSES),
        "b_out": bp["bout"],
    }

    tests = []
    for i in range(2**SEQ_LEN):
        bits = [(i >> b) & 1 for b in range(SEQ_LEN)]
        tests.append({"tokens": bits, "expected": sum(bits) % 2})

    out = "/Users/bbclaude/Projects/neural-decompile/examples"
    with open(f"{out}/parity_transformer.json", "w") as f:
        json.dump(export, f)
    with open(f"{out}/parity_transformer_tests.json", "w") as f:
        json.dump(tests, f, indent=2)
    print(f"\nExported to {out}/")
else:
    print("Could not find perfect weights. Trying next approach...")
