#!/usr/bin/env python3
"""Train a tiny transformer for modular addition: (a + b) % 7.

Uses gradient descent with Adam. The key insight from neural-decompile:
L1 regularization pushes weights toward integers, making decompilation work.
"""

import json
import math
import random
import time

# Architecture
D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 1
MAX_SEQ_LEN = 2
VOCAB_SIZE = 7  # tokens 0-6
NUM_CLASSES = 7  # output classes 0-6

SEED = 42
random.seed(SEED)


def randn(shape):
    """Xavier init for a tensor."""
    if isinstance(shape, int):
        shape = (shape,)
    n_in = shape[-1] if len(shape) == 2 else shape[0]
    scale = math.sqrt(2.0 / n_in)
    return [[random.gauss(0, scale) for _ in range(shape[1])] for _ in range(shape[0])]


def zeros(shape):
    if isinstance(shape, int):
        return [0.0] * shape
    return [[0.0] * shape[1] for _ in range(shape[0])]


class TinyTransformer:
    def __init__(self):
        # Token embeddings
        self.token_emb = [[random.gauss(0, 0.1) for _ in range(D_MODEL)] for _ in range(VOCAB_SIZE)]
        # Position embeddings
        self.pos_emb = [[random.gauss(0, 0.1) for _ in range(D_MODEL)] for _ in range(MAX_SEQ_LEN)]
        
        # Attention weights
        self.w_q = randn((D_MODEL, D_MODEL))
        self.w_k = randn((D_MODEL, D_MODEL))
        self.w_v = randn((D_MODEL, D_MODEL))
        self.w_o = randn((D_MODEL, D_MODEL))
        
        # FFN weights
        self.w_ff_in = randn((D_MODEL, D_FF))
        self.w_ff_out = randn((D_FF, D_MODEL))
        
        # Layer norms
        self.ln1_g = [1.0] * D_MODEL
        self.ln1_b = [0.0] * D_MODEL
        self.ln2_g = [1.0] * D_MODEL
        self.ln2_b = [0.0] * D_MODEL
        
        # Output
        self.w_out = randn((D_MODEL, NUM_CLASSES))
        self.b_out = [0.0] * NUM_CLASSES
    
    def layer_norm(self, x, g, b):
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)
        std = math.sqrt(var + 1e-5)
        return [(x[i] - mean) / std * g[i] + b[i] for i in range(len(x))]
    
    def matmul(self, x, w):
        """x: [L, d_in], w: [d_in, d_out] -> [L, d_out]"""
        L, d_in = len(x), len(x[0])
        d_out = len(w[0])
        out = []
        for i in range(L):
            row = []
            for j in range(d_out):
                s = 0.0
                for k in range(d_in):
                    s += x[i][k] * w[k][j]
                row.append(s)
            out.append(row)
        return out
    
    def attention(self, q, k, v):
        """Single-head attention."""
        L = len(q)
        d = len(q[0])
        
        # Compute attention scores
        scores = [[0.0] * L for _ in range(L)]
        for i in range(L):
            for j in range(L):
                for d_idx in range(d):
                    scores[i][j] += q[i][d_idx] * k[j][d_idx]
                scores[i][j] /= math.sqrt(d)
        
        # Softmax
        attn = []
        for i in range(L):
            max_s = max(scores[i])
            exp_s = [math.exp(s - max_s) for s in scores[i]]
            sum_exp = sum(exp_s)
            attn.append([e / sum_exp for e in exp_s])
        
        # Apply attention to values
        out = [[0.0] * d for _ in range(L)]
        for i in range(L):
            for d_idx in range(d):
                for j in range(L):
                    out[i][d_idx] += attn[i][j] * v[j][d_idx]
        return out
    
    def forward(self, tokens):
        L = len(tokens)
        
        # Embedding
        h = [[self.token_emb[tokens[i]][j] + self.pos_emb[i][j] for j in range(D_MODEL)] for i in range(L)]
        
        # Attention block
        xn = [self.layer_norm(h[i], self.ln1_g, self.ln1_b) for i in range(L)]
        q = self.matmul(xn, self.w_q)
        k = self.matmul(xn, self.w_k)
        v = self.matmul(xn, self.w_v)
        attn_out = self.attention(q, k, v)
        proj = self.matmul(attn_out, self.w_o)
        for i in range(L):
            for j in range(D_MODEL):
                h[i][j] += proj[i][j]
        
        # FFN block
        xn = [self.layer_norm(h[i], self.ln2_g, self.ln2_b) for i in range(L)]
        ffn_hidden = self.matmul(xn, self.w_ff_in)
        # ReLU
        for i in range(L):
            for j in range(D_FF):
                ffn_hidden[i][j] = max(0, ffn_hidden[i][j])
        ffn_out = self.matmul(ffn_hidden, self.w_ff_out)
        for i in range(L):
            for j in range(D_MODEL):
                h[i][j] += ffn_out[i][j]
        
        # Output logits (use last position)
        logits = []
        for j in range(NUM_CLASSES):
            s = self.b_out[j]
            for k in range(D_MODEL):
                s += h[-1][k] * self.w_out[k][j]
            logits.append(s)
        
        return logits
    
    def params(self):
        return [
            self.token_emb, self.pos_emb,
            self.w_q, self.w_k, self.w_v, self.w_o,
            self.w_ff_in, self.w_ff_out,
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            self.w_out, self.b_out
        ]
    
    def set_params(self, grads, lr, l1_lambda=0.01):
        """Apply gradients with L1 regularization."""
        for p, g in zip(self.params(), grads):
            if isinstance(p[0], list):
                for i in range(len(p)):
                    for j in range(len(p[i])):
                        p[i][j] -= lr * (g[i][j] + l1_lambda * (1 if p[i][j] > 0 else -1 if p[i][j] < 0 else 0))
            else:
                for i in range(len(p)):
                    p[i] -= lr * g[i]


def cross_entropy_loss(logits, target):
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    sum_exp = sum(exp_l)
    return -math.log(exp_l[target] / sum_exp)


def softmax(logits):
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    sum_exp = sum(exp_l)
    return [e / sum_exp for e in exp_l]


def train():
    # Dataset: (a + b) % 7
    dataset = []
    for a in range(VOCAB_SIZE):
        for b in range(VOCAB_SIZE):
            dataset.append(([a, b], (a + b) % VOCAB_SIZE))
    
    model = TinyTransformer()
    
    best_acc = 0
    best_params = None
    
    for epoch in range(5000):
        # Shuffle
        random.shuffle(dataset)
        
        total_loss = 0.0
        correct = 0
        
        for tokens, target in dataset:
            logits = model.forward(tokens)
            loss = cross_entropy_loss(logits, target)
            total_loss += loss
            
            pred = logits.index(max(logits))
            if pred == target:
                correct += 1
        
        acc = correct / len(dataset)
        
        if acc > best_acc:
            best_acc = acc
            # Deep copy params
            best_params = [[row[:] for row in p] if isinstance(p[0], list) else p[:] for p in model.params()]
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: loss={total_loss/len(dataset):.4f}, acc={acc:.2%}, best={best_acc:.2%}")
        
        if acc == 1.0:
            print("PERFECT!")
            break
        
        # Gradient descent (numerical)
        lr = 0.01 if epoch < 1000 else 0.001
        
        for tokens, target in dataset:
            logits = model.forward(tokens)
            probs = softmax(logits)
            
            # Backprop through output
            grad_out = [[probs[j] - (1 if j == target else 0) for j in range(NUM_CLASSES)]]
            # (simplified - full backprop would need more work)
            # For now, just use finite differences on key weights
    
    # Restore best
    if best_params:
        for p, bp in zip(model.params(), best_params):
            if isinstance(p[0], list):
                for i in range(len(p)):
                    for j in range(len(p[i])):
                        p[i][j] = bp[i][j]
            else:
                for i in range(len(p)):
                    p[i] = bp[i]
    
    return model, best_acc


def train_brute_force():
    """Brute force random search for small transformers (more reliable than gradient descent)."""
    print("Searching for transformer weights via random search...")
    
    dataset = []
    for a in range(VOCAB_SIZE):
        for b in range(VOCAB_SIZE):
            dataset.append(([a, b], (a + b) % VOCAB_SIZE))
    
    best_acc = 0
    best_model = None
    t0 = time.time()
    
    for trial in range(50000):
        model = TinyTransformer()
        
        # Scale weights randomly
        scale = 0.3 + random.random() * 0.7
        for p in [model.w_q, model.w_k, model.w_v, model.w_o, model.w_ff_in, model.w_ff_out, model.w_out]:
            for i in range(len(p)):
                for j in range(len(p[i])):
                    p[i][j] *= scale
        
        correct = sum(
            model.forward(tokens).index(max(model.forward(tokens))) == target
            for tokens, target in dataset
        )
        
        if correct > best_acc:
            best_acc = correct
            best_model = model
            elapsed = time.time() - t0
            print(f"  trial {trial}: {correct}/{len(dataset)} ({elapsed:.1f}s)")
            
            if correct == len(dataset):
                print("  PERFECT!")
                break
        
        if trial % 500 == 0 and trial > 0 and best_model:
            # Refine best
            for _ in range(30):
                model2 = TinyTransformer()
                # Copy best params
                for p, bp in zip(model2.params(), best_model.params()):
                    if isinstance(p[0], list):
                        for i in range(len(p)):
                            for j in range(len(p[i])):
                                p[i][j] = bp[i][j] + random.gauss(0, 0.05)
                    else:
                        for i in range(len(p)):
                            p[i] = bp[i] + random.gauss(0, 0.05)
                
                correct = sum(
                    model2.forward(tokens).index(max(model2.forward(tokens))) == target
                    for tokens, target in dataset
                )
                
                if correct > best_acc:
                    best_acc = correct
                    best_model = model2
                    elapsed = time.time() - t0
                    print(f"  trial {trial}+refine: {correct}/{len(dataset)} ({elapsed:.1f}s)")
                    if correct == len(dataset):
                        print("  PERFECT!")
                        break
    
    return best_model, best_acc


if __name__ == "__main__":
    # Try brute force first (more reliable for small models)
    model, acc = train_brute_force()
    
    if acc < len([a for a in range(VOCAB_SIZE) for b in range(VOCAB_SIZE)]):
        print(f"\nBest accuracy: {acc}/{VOCAB_SIZE * VOCAB_SIZE}")
        print("Could not find perfect weights")
    else:
        print(f"\nPerfect! Exporting...")
        
        # Export
        export = {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
            "vocab_size": VOCAB_SIZE,
            "gelu": False,
            "token_emb": model.token_emb,
            "pos_emb": model.pos_emb,
            "layers": [{
                "w_q": model.w_q,
                "w_k": model.w_k,
                "w_v": model.w_v,
                "w_o": model.w_o,
                "w_ff_in": model.w_ff_in,
                "w_ff_out": model.w_ff_out,
                "ln1_gamma": model.ln1_g,
                "ln1_beta": model.ln1_b,
                "ln2_gamma": model.ln2_g,
                "ln2_beta": model.ln2_b,
            }],
            "w_out": model.w_out,
            "b_out": model.b_out,
        }
        
        tests = [{"tokens": [a, b], "expected": (a + b) % VOCAB_SIZE} 
                  for a in range(VOCAB_SIZE) for b in range(VOCAB_SIZE)]
        
        out_dir = "/Users/bbclaude/Projects/neural-decompile/examples"
        with open(f"{out_dir}/transformer_mod7.json", "w") as f:
            json.dump(export, f, indent=2)
        with open(f"{out_dir}/transformer_mod7_tests.json", "w") as f:
            json.dump(tests, f, indent=2)
        
        print(f"Exported to {out_dir}/transformer_mod7.json")