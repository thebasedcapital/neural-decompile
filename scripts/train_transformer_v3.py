#!/usr/bin/env python3
"""
Train a minimal transformer on modular addition: (a + b) % 7
Uses pure numpy with proper manual backpropagation.
"""

import json
import math
import random
import numpy as np

class TinyTransformer:
    """Minimal transformer with manual backprop."""

    def __init__(self, vocab_size=7, d_model=16, n_heads=2, d_ff=32, max_seq_len=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads

        # Xavier init
        def xavier(fan_in, fan_out):
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(fan_in, fan_out) * std

        # Embeddings
        self.token_emb = xavier(vocab_size, d_model) * 0.5
        self.pos_emb = xavier(max_seq_len, d_model) * 0.5

        # Attention weights [d_model, d_model]
        self.w_q = xavier(d_model, d_model)
        self.w_k = xavier(d_model, d_model)
        self.w_v = xavier(d_model, d_model)
        self.w_o = xavier(d_model, d_model)

        # FFN
        self.w_ff_in = xavier(d_model, d_ff)
        self.w_ff_out = xavier(d_ff, d_model)

        # Layer norm params
        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)
        self.ln_final_g = np.ones(d_model)
        self.ln_final_b = np.zeros(d_model)

        # Output
        self.w_out = xavier(d_model, vocab_size)

        # Gradients
        self.reset_grads()

    def reset_grads(self):
        """Reset all gradients."""
        self.grads = {
            'token_emb': np.zeros_like(self.token_emb),
            'pos_emb': np.zeros_like(self.pos_emb),
            'w_q': np.zeros_like(self.w_q),
            'w_k': np.zeros_like(self.w_k),
            'w_v': np.zeros_like(self.w_v),
            'w_o': np.zeros_like(self.w_o),
            'w_ff_in': np.zeros_like(self.w_ff_in),
            'w_ff_out': np.zeros_like(self.w_ff_out),
            'w_out': np.zeros_like(self.w_out),
            'ln1_g': np.zeros_like(self.ln1_g),
            'ln1_b': np.zeros_like(self.ln1_b),
            'ln2_g': np.zeros_like(self.ln2_g),
            'ln2_b': np.zeros_like(self.ln2_b),
            'ln_final_g': np.zeros_like(self.ln_final_g),
            'ln_final_b': np.zeros_like(self.ln_final_b),
        }

    def layer_norm_fwd(self, x, gamma, beta):
        """Forward pass of layer norm."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-5)
        x_norm = (x - mean) / std
        out = gamma * x_norm + beta
        return out, (x, mean, var, std, x_norm, gamma)

    def layer_norm_bwd(self, dout, cache):
        """Backward pass of layer norm."""
        x, mean, var, std, x_norm, gamma = cache
        N = x.shape[-1]

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * x_norm, axis=0)

        dx_norm = dout * gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std**(-3), axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -1/std, axis=-1, keepdims=True) + dvar * np.sum(-2*(x-mean), axis=-1, keepdims=True) / N
        dx = dx_norm / std + dvar * 2*(x-mean)/N + dmean / N

        return dx, dgamma, dbeta

    def forward(self, tokens, training=False):
        """Forward pass with optional gradient storage."""
        self.cache = {'tokens': tokens}
        seq_len = len(tokens)

        # Embeddings
        x = self.token_emb[tokens] + self.pos_emb[:seq_len]
        self.cache['emb'] = x.copy()

        # Layer 1: Pre-norm attention
        h, ln1_cache = self.layer_norm_fwd(x, self.ln1_g, self.ln1_b)
        self.cache['ln1'] = ln1_cache

        # Q, K, V
        Q = h @ self.w_q
        K = h @ self.w_k
        V = h @ self.w_v
        self.cache['qkv'] = (Q.copy(), K.copy(), V.copy(), h.copy())

        # Multi-head attention
        attn_out = np.zeros_like(x)
        attn_caches = []

        for i in range(self.n_heads):
            start = i * self.head_dim
            end = start + self.head_dim

            q = Q[:, start:end]
            k = K[:, start:end]
            v = V[:, start:end]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            # Softmax
            max_score = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_score)
            sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
            attn = exp_scores / sum_exp

            head_out = attn @ v
            attn_out[:, start:end] = head_out

            # Store for backprop
            attn_caches.append((q, k, v, scores, attn, start, end))

        self.cache['attn_caches'] = attn_caches
        self.cache['attn_out_pre'] = attn_out.copy()

        # Output projection
        attn_proj = attn_out @ self.w_o
        x_new = x + attn_proj
        self.cache['after_attn'] = (x.copy(), attn_out.copy(), x_new.copy())

        # FFN
        h2, ln2_cache = self.layer_norm_fwd(x_new, self.ln2_g, self.ln2_b)
        self.cache['ln2'] = ln2_cache

        ffn_hidden = h2 @ self.w_ff_in
        ffn_activated = np.maximum(0, ffn_hidden)  # ReLU
        ffn_out = ffn_activated @ self.w_ff_out
        x_final = x_new + ffn_out
        self.cache['after_ffn'] = (x_new.copy(), ffn_activated.copy(), ffn_out.copy(), x_final.copy())

        # Final LN
        x_norm, ln_final_cache = self.layer_norm_fwd(x_final, self.ln_final_g, self.ln_final_b)
        self.cache['ln_final'] = ln_final_cache

        # Output projection
        logits = x_norm @ self.w_out
        self.cache['logits'] = (x_norm.copy(), logits.copy())

        return logits

    def backward(self, target):
        """Backward pass to compute gradients."""
        # Get logits and compute loss gradient
        x_norm, logits = self.cache['logits']
        seq_len = logits.shape[0]

        # Softmax cross-entropy gradient
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        d_logits = probs.copy()
        d_logits[-1, target] -= 1  # Gradient at last position only

        # Backprop through w_out
        self.grads['w_out'] += np.outer(x_norm[-1], d_logits[-1])
        d_x_norm = d_logits @ self.w_out.T

        # Backprop through final LN
        d_x_final, dg, db = self.layer_norm_bwd(d_x_norm, self.cache['ln_final'])
        self.grads['ln_final_g'] += dg
        self.grads['ln_final_b'] += db

        # Backprop through FFN residual
        x_new, ffn_activated, ffn_out, _ = self.cache['after_ffn']
        d_x_new = d_x_final.copy()
        d_ffn_out = d_x_final.copy()

        # Backprop through w_ff_out
        self.grads['w_ff_out'] += np.outer(ffn_activated[-1], d_ffn_out[-1])
        d_ffn_activated = d_ffn_out @ self.w_ff_out.T

        # Backprop through ReLU
        d_ffn_hidden = d_ffn_activated * (ffn_activated > 0)

        # Backprop through w_ff_in
        h2 = self.cache['ln2'][0]  # x_norm from ln2
        self.grads['w_ff_in'] += np.outer(h2[-1], d_ffn_hidden[-1])
        d_h2 = d_ffn_hidden @ self.w_ff_in.T

        # Backprop through ln2
        d_x_new_late, dg, db = self.layer_norm_bwd(d_h2, self.cache['ln2'])
        self.grads['ln2_g'] += dg
        self.grads['ln2_b'] += db
        d_x_new += d_x_new_late

        # Backprop through attention residual
        x, attn_out, _ = self.cache['after_attn']
        d_x = d_x_new.copy()
        d_attn_out = d_x_new @ self.w_o.T

        # Backprop through w_o
        attn_out_pre = self.cache['attn_out_pre']
        self.grads['w_o'] += np.outer(attn_out_pre[-1], d_x_new[-1])

        # Backprop through attention heads
        Q, K, V, h = self.cache['qkv']
        d_Q = np.zeros_like(Q)
        d_K = np.zeros_like(K)
        d_V = np.zeros_like(V)

        for i, (q, k, v, scores, attn, start, end) in enumerate(self.cache['attn_caches']):
            d_head_out = d_attn_out[:, start:end]

            # Backprop through attn @ v
            d_attn = d_head_out @ v.T
            d_v = attn.T @ d_head_out
            d_V[:, start:end] = d_v

            # Backprop through softmax
            # d_softmax = attn * (d_attn - sum(d_attn * attn, axis=-1))
            d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
            d_scores /= np.sqrt(self.head_dim)

            # Backprop through q @ k.T
            d_q = d_scores @ k
            d_k = d_scores.T @ q

            d_Q[:, start:end] = d_q
            d_K[:, start:end] = d_k

        # Backprop through Q, K, V projections
        self.grads['w_q'] += np.outer(h[-1], d_Q[-1])
        self.grads['w_k'] += np.outer(h[-1], d_K[-1])
        self.grads['w_v'] += np.outer(h[-1], d_V[-1])

        d_h = d_Q @ self.w_q.T + d_K @ self.w_k.T + d_V @ self.w_v.T

        # Backprop through ln1
        d_x_early, dg, db = self.layer_norm_bwd(d_h, self.cache['ln1'])
        self.grads['ln1_g'] += dg
        self.grads['ln1_b'] += db
        d_x += d_x_early

        # Backprop through embeddings
        tokens = self.cache['tokens']
        for i, tok in enumerate(tokens):
            self.grads['token_emb'][tok] += d_x[i]
            self.grads['pos_emb'][i] += d_x[i]

    def update(self, lr=0.01):
        """Update weights using gradients."""
        self.token_emb -= lr * self.grads['token_emb']
        self.pos_emb -= lr * self.grads['pos_emb']
        self.w_q -= lr * self.grads['w_q']
        self.w_k -= lr * self.grads['w_k']
        self.w_v -= lr * self.grads['w_v']
        self.w_o -= lr * self.grads['w_o']
        self.w_ff_in -= lr * self.grads['w_ff_in']
        self.w_ff_out -= lr * self.grads['w_ff_out']
        self.w_out -= lr * self.grads['w_out']
        self.ln1_g -= lr * self.grads['ln1_g']
        self.ln1_b -= lr * self.grads['ln1_b']
        self.ln2_g -= lr * self.grads['ln2_g']
        self.ln2_b -= lr * self.grads['ln2_b']
        self.ln_final_g -= lr * self.grads['ln_final_g']
        self.ln_final_b -= lr * self.grads['ln_final_b']

        self.reset_grads()

    def train_step(self, tokens, target, lr=0.01):
        """Single training step."""
        self.reset_grads()
        logits = self.forward(tokens, training=True)

        # Compute loss
        last_logits = logits[-1]
        exp = np.exp(last_logits - np.max(last_logits))
        probs = exp / np.sum(exp)
        loss = -np.log(probs[target] + 1e-10)

        # Backward and update
        self.backward(target)
        self.update(lr)

        return loss, probs.argmax() == target

    def save(self, path):
        """Save to JSON format."""
        def to_list(arr):
            return arr.tolist()

        data = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": 1,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "gelu": False,
            "token_emb": to_list(self.token_emb),
            "pos_emb": to_list(self.pos_emb),
            "layers": [{
                "w_q": to_list(self.w_q),
                "w_k": to_list(self.w_k),
                "w_v": to_list(self.w_v),
                "w_o": to_list(self.w_o),
                "b_q": None, "b_k": None, "b_v": None, "b_o": None,
                "w_ff_in": to_list(self.w_ff_in),
                "b_ff_in": None,
                "w_ff_out": to_list(self.w_ff_out),
                "b_ff_out": None,
                "ln1_gamma": to_list(self.ln1_g),
                "ln1_beta": to_list(self.ln1_b),
                "ln2_gamma": to_list(self.ln2_g),
                "ln2_beta": to_list(self.ln2_b),
            }],
            "ln_final_gamma": to_list(self.ln_final_g),
            "ln_final_beta": to_list(self.ln_final_b),
            "w_out": to_list(self.w_out),
            "b_out": None,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

def train():
    random.seed(42)
    np.random.seed(42)

    model = TinyTransformer(vocab_size=7, d_model=16, n_heads=2, d_ff=32)

    all_data = []
    for a in range(7):
        for b in range(7):
            all_data.append(([a, b], (a + b) % 7))

    print(f"Training: {len(all_data)} samples")
    print(f"Params: {sum(w.size for w in [model.token_emb, model.pos_emb, model.w_q, model.w_k, model.w_v, model.w_o, model.w_ff_in, model.w_ff_out, model.w_out])}")
    print()

    best_acc = 0
    lr = 0.1

    for epoch in range(1000):
        random.shuffle(all_data)
        total_loss = 0
        correct = 0

        for tokens, target in all_data:
            loss, ok = model.train_step(tokens, target, lr=lr)
            total_loss += loss
            if ok:
                correct += 1

        acc = correct / len(all_data) * 100

        if epoch % 100 == 0 or acc > 95:
            print(f"Epoch {epoch:4d}: loss={total_loss/len(all_data):.4f} acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc

        if acc == 100.0:
            print(f"\n✓ Perfect at epoch {epoch}!")
            break

        # Decay learning rate
        if epoch % 200 == 199:
            lr *= 0.5

    print(f"\nBest: {best_acc:.1f}%")

    # Final eval
    correct = 0
    for tokens, target in all_data:
        logits = model.forward(tokens)
        if logits[-1].argmax() == target:
            correct += 1

    final_acc = correct / len(all_data) * 100
    print(f"Final: {correct}/{len(all_data)} = {final_acc:.1f}%")

    if final_acc >= 95:
        model.save("examples/trained_mod7_transformer.json")
        tests = [{"tokens": t, "expected": tgt} for t, tgt in all_data]
        with open("examples/trained_mod7_tests.json", 'w') as f:
            json.dump(tests, f, indent=2)
        print("Saved test cases")
        return True
    return False

if __name__ == "__main__":
    train()
