#!/usr/bin/env python3
"""Create proper mod7 test data and train a tiny transformer using numpy."""

import json
import math
import random
import time

# Smaller architecture for faster training
D_MODEL = 8
D_FF = 16
VOCAB_SIZE = 7

SEED = 42
random.seed(SEED)


def randmat(rows, cols, scale=0.5):
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def softmax(x):
    max_x = max(x)
    e = [math.exp(xi - max_x) for xi in x]
    s = sum(e)
    return [ei/s for ei in e]


def matmul(a, b):
    """a: [m, k], b: [k, n] -> [m, n]"""
    m, k1 = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k1 == k2
    return [[sum(a[i][k] * b[k][j] for k in range(k1)) for j in range(n)] for i in range(m)]


def layer_norm(x, g, b):
    mean = sum(x) / len(x)
    var = sum((xi - mean)**2 for xi in x) / len(x)
    std = math.sqrt(var + 1e-5)
    return [(x[i] - mean) / std * g[i] + b[i] for i in range(len(x))]


def relu(x):
    return [max(0, xi) for xi in x]


class TinyTransformer:
    def __init__(self):
        self.te = randmat(VOCAB_SIZE, D_MODEL, 0.3)
        self.pe = randmat(2, D_MODEL, 0.1)  # max_seq_len=2
        
        self.wq = randmat(D_MODEL, D_MODEL, 0.3)
        self.wk = randmat(D_MODEL, D_MODEL, 0.3)
        self.wv = randmat(D_MODEL, D_MODEL, 0.3)
        self.wo = randmat(D_MODEL, D_MODEL, 0.3)
        
        self.w1 = randmat(D_MODEL, D_FF, 0.3)
        self.w2 = randmat(D_FF, D_MODEL, 0.3)
        
        self.wout = randmat(D_MODEL, VOCAB_SIZE, 0.3)
        self.bout = [0.0] * VOCAB_SIZE
        
        self.lng = [1.0] * D_MODEL
        self.lnb = [0.0] * D_MODEL
    
    def forward(self, tokens):
        # Embed
        h = [[self.te[tokens[i]][j] + self.pe[i][j] for j in range(D_MODEL)] for i in range(2)]
        
        # LN -> Attention
        hn = [layer_norm(h[i], self.lng, self.lnb) for i in range(2)]
        q = matmul(hn, self.wq)
        k = matmul(hn, self.wk)
        v = matmul(hn, self.wv)
        
        # Single-head attention
        scores = [[sum(q[i][d] * k[j][d] for d in range(D_MODEL)) / math.sqrt(D_MODEL) 
                   for j in range(2)] for i in range(2)]
        attn = [softmax(scores[i]) for i in range(2)]
        av = [[sum(attn[i][j] * v[j][d] for j in range(2)) for d in range(D_MODEL)] for i in range(2)]
        
        proj = matmul(av, self.wo)
        h = [[h[i][d] + proj[i][d] for d in range(D_MODEL)] for i in range(2)]
        
        # LN -> FFN
        hn = [layer_norm(h[i], self.lng, self.lnb) for i in range(2)]
        ff = relu(matmul(hn, self.w1)[0])  # Use first position
        ff_out = matmul([ff], self.w2)[0]
        
        # Output
        logits = [sum(h[1][d] * self.wout[d][c] + self.bout[c] for d in range(D_MODEL)) 
                  for c in range(VOCAB_SIZE)]
        return logits


def train_sgd():
    """Train with simple gradient descent."""
    print("Training mod7 transformer...")
    
    # Dataset
    data = [([a, b], (a + b) % VOCAB_SIZE) for a in range(VOCAB_SIZE) for b in range(VOCAB_SIZE)]
    
    model = TinyTransformer()
    lr = 0.1
    
    best_acc = 0
    best_params = None
    
    for epoch in range(2000):
        random.shuffle(data)
        correct = 0
        
        for tokens, target in data:
            # Forward
            logits = model.forward(tokens)
            pred = logits.index(max(logits))
            if pred == target:
                correct += 1
        
        acc = correct / len(data)
        if acc > best_acc:
            best_acc = acc
            best_params = {
                'te': [row[:] for row in model.te],
                'pe': [row[:] for row in model.pe],
                'wq': [row[:] for row in model.wq],
                'wk': [row[:] for row in model.wk],
                'wv': [row[:] for row in model.wv],
                'wo': [row[:] for row in model.wo],
                'w1': [row[:] for row in model.w1],
                'w2': [row[:] for row in model.w2],
                'wout': [row[:] for row in model.wout],
                'bout': model.bout[:],
                'lng': model.lng[:],
                'lnb': model.lnb[:],
            }
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: acc={acc:.2%}, best={best_acc:.2%}")
        
        if acc == 1.0:
            print("PERFECT!")
            break
        
        # Simple weight perturbation search
        if epoch % 50 == 0 and best_params:
            for _ in range(10):
                # Mutate
                for attr in ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'wout']:
                    p = getattr(model, attr)
                    for i in range(len(p)):
                        for j in range(len(p[i])):
                            p[i][j] = best_params[attr][i][j] + random.gauss(0, 0.05)
                
                correct = sum(
                    model.forward(t).index(max(model.forward(t))) == tgt
                    for t, tgt in data
                )
                if correct > best_acc * len(data):
                    best_acc = correct / len(data)
                    best_params = {
                        'te': [row[:] for row in model.te],
                        'pe': [row[:] for row in model.pe],
                        'wq': [row[:] for row in model.wq],
                        'wk': [row[:] for row in model.wk],
                        'wv': [row[:] for row in model.wv],
                        'wo': [row[:] for row in model.wo],
                        'w1': [row[:] for row in model.w1],
                        'w2': [row[:] for row in model.w2],
                        'wout': [row[:] for row in model.wout],
                        'bout': model.bout[:],
                        'lng': model.lng[:],
                        'lnb': model.lnb[:],
                    }
                    print(f"  Found better: {best_acc:.2%}")
                    if best_acc == 1.0:
                        break
    
    # Restore best
    if best_params:
        model.te = best_params['te']
        model.pe = best_params['pe']
        model.wq = best_params['wq']
        model.wk = best_params['wk']
        model.wv = best_params['wv']
        model.wo = best_params['wo']
        model.w1 = best_params['w1']
        model.w2 = best_params['w2']
        model.wout = best_params['wout']
        model.bout = best_params['bout']
    
    return model, best_acc


if __name__ == "__main__":
    model, acc = train_sgd()
    
    if acc == 1.0:
        print("\nExporting...")
        export = {
            "d_model": D_MODEL,
            "n_heads": 1,
            "d_ff": D_FF,
            "n_layers": 1,
            "max_seq_len": 2,
            "vocab_size": VOCAB_SIZE,
            "gelu": False,
            "token_emb": model.te,
            "pos_emb": model.pe,
            "layers": [{
                "w_q": model.wq,
                "w_k": model.wk,
                "w_v": model.wv,
                "w_o": model.wo,
                "w_ff_in": model.w1,
                "w_ff_out": model.w2,
                "ln1_gamma": model.lng,
                "ln1_beta": model.lnb,
                "ln2_gamma": model.lng,
                "ln2_beta": model.lnb,
            }],
            "w_out": model.wout,
            "b_out": model.bout,
        }
        
        tests = [{"tokens": [a, b], "expected": (a + b) % VOCAB_SIZE}
                  for a in range(VOCAB_SIZE) for b in range(VOCAB_SIZE)]
        
        out = "/Users/bbclaude/Projects/neural-decompile/examples"
        with open(f"{out}/transformer_mod7.json", "w") as f:
            json.dump(export, f, indent=2)
        with open(f"{out}/transformer_mod7_tests.json", "w") as f:
            json.dump(tests, f, indent=2)
        print(f"Done! {out}/transformer_mod7.json")
    else:
        print(f"\nFailed: best accuracy = {acc:.2%}")
        print("Trying with different architecture...")