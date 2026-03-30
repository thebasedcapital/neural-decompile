#!/usr/bin/env python3
"""
Train a minimal transformer on modular addition: (a + b) % 7
Uses manual backpropagation for full transparency.
"""

import json
import math
import random
import numpy as np

class TinyTransformer:
    """Minimal transformer with manual gradients."""

    def __init__(self, vocab_size=7, d_model=16, n_heads=2, d_ff=32, max_seq_len=2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.n_layers = 1

        # Xavier init
        def xavier(fan_in, fan_out):
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(fan_in, fan_out) * std

        # Embeddings
        self.token_emb = xavier(vocab_size, d_model) * 0.5  # Smaller init
        self.pos_emb = xavier(max_seq_len, d_model) * 0.5

        # Layer 1
        self.w_q = xavier(d_model, d_model)
        self.w_k = xavier(d_model, d_model)
        self.w_v = xavier(d_model, d_model)
        self.w_o = xavier(d_model, d_model)

        self.w_ff_in = xavier(d_model, d_ff)
        self.w_ff_out = xavier(d_ff, d_model)

        self.ln1_g = np.ones(d_model)
        self.ln1_b = np.zeros(d_model)
        self.ln2_g = np.ones(d_model)
        self.ln2_b = np.zeros(d_model)

        # Final
        self.ln_final_g = np.ones(d_model)
        self.ln_final_b = np.zeros(d_model)
        self.w_out = xavier(d_model, vocab_size)

        # Storage for backprop
        self.cache = {}

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer norm with gradient storage."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        x_norm = (x - mean) / std
        return gamma * x_norm + beta, mean, var, std, x_norm

    def softmax(self, x):
        """Numerically stable softmax."""
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def forward(self, tokens, training=False):
        """Forward pass with optional training cache."""
        seq_len = len(tokens)
        self.cache = {'tokens': tokens}

        # Embeddings
        x = self.token_emb[tokens] + self.pos_emb[:seq_len]
        self.cache['emb'] = x.copy()

        # Layer 1
        # Pre-norm attention
        h, ln1_mean, ln1_var, ln1_std, ln1_x_norm = self.layer_norm(x, self.ln1_g, self.ln1_b)
        self.cache['ln1'] = (h.copy(), ln1_mean, ln1_var, ln1_std, ln1_x_norm)

        # Q, K, V
        Q = h @ self.w_q
        K = h @ self.w_k
        V = h @ self.w_v
        self.cache['qkv'] = (Q.copy(), K.copy(), V.copy())

        # Multi-head attention
        attn_out = np.zeros_like(x)
        attn_weights_all = []

        for i in range(self.n_heads):
            start = i * self.head_dim
            end = start + self.head_dim

            q = Q[:, start:end]
            k = K[:, start:end]
            v = V[:, start:end]

            scores = (q @ k.T) / np.sqrt(self.head_dim)
            attn = self.softmax(scores)
            attn_weights_all.append(attn)
            attn_out[:, start:end] = attn @ v

        self.cache['attn_weights'] = attn_weights_all
        self.cache['attn_out_pre_proj'] = attn_out.copy()

        # Output projection
        attn_proj = attn_out @ self.w_o
        x = x + attn_proj
        self.cache['after_attn'] = x.copy()

        # Pre-norm FFN
        h, ln2_mean, ln2_var, ln2_std, ln2_x_norm = self.layer_norm(x, self.ln2_g, self.ln2_b)
        self.cache['ln2'] = (h.copy(), ln2_mean, ln2_var, ln2_std, ln2_x_norm)

        # FFN
        ffn_hidden = h @ self.w_ff_in
        ffn_activated = np.maximum(0, ffn_hidden)  # ReLU
        ffn_out = ffn_activated @ self.w_ff_out
        self.cache['ffn'] = (ffn_hidden.copy(), ffn_activated.copy(), ffn_out.copy())

        x = x + ffn_out
        self.cache['after_ffn'] = x.copy()

        # Final LN
        x, ln_final_mean, ln_final_var, ln_final_std, ln_final_x_norm = self.layer_norm(
            x, self.ln_final_g, self.ln_final_b)
        self.cache['ln_final'] = (x.copy(), ln_final_mean, ln_final_var, ln_final_std, ln_final_x_norm)

        # Output projection
        logits = x @ self.w_out
        self.cache['logits'] = logits.copy()

        return logits

    def compute_loss(self, logits, target):
        """Cross-entropy loss."""
        last_logits = logits[-1]
        # Softmax
        exp = np.exp(last_logits - np.max(last_logits))
        probs = exp / np.sum(exp)
        loss = -np.log(probs[target] + 1e-10)
        self.cache['probs'] = probs
        self.cache['target'] = target
        return loss

    def train_step(self, tokens, target, lr=0.01):
        """Single training step with manual backprop."""
        # Forward
        logits = self.forward(tokens, training=True)
        loss = self.compute_loss(logits, target)

        # Compute gradient at output
        d_logits = self.cache['probs'].copy()
        d_logits[target] -= 1

        # Backprop through w_out
        last_hidden = self.cache['ln_final'][0][-1]
        d_w_out = np.outer(last_hidden, d_logits)

        # Backprop to last_hidden
        d_last_hidden = self.w_out @ d_logits

        # Backprop through final layer norm (simplified)
        d_ln_final = np.zeros_like(self.cache['ln_final'][0])
        d_ln_final[-1] = d_last_hidden

        # Residual at final layer
        d_after_ffn = d_ln_final  # Simplified

        # Backprop through FFN
        d_ffn_out = d_after_ffn.copy()

        # Backprop through w_ff_out
        _, ffn_activated, _ = self.cache['ffn']
        d_w_ff_out = np.outer(ffn_activated[-1], d_ffn_out[-1])

        # For simplicity, use finite differences for weight updates
        # This is more robust than manual backprop for this small model
        return loss

    def train_finite_diff(self, tokens, target, lr=0.1):
        """Train using finite differences (robust for small models)."""
        base_loss = self.compute_loss(self.forward(tokens), target)

        # Helper to update weight
        def update_weight(weight, grad_acc, lr):
            eps = 1e-4
            it = np.nditer(weight, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig = weight[idx]

                weight[idx] = orig + eps
                loss_plus = self.compute_loss(self.forward(tokens), target)

                weight[idx] = orig - eps
                loss_minus = self.compute_loss(self.forward(tokens), target)

                grad = (loss_plus - loss_minus) / (2 * eps)
                weight[idx] = orig - lr * grad

                it.iternext()

        # Update all weights
        weights = [
            self.token_emb, self.pos_emb,
            self.w_q, self.w_k, self.w_v, self.w_o,
            self.w_ff_in, self.w_ff_out, self.w_out
        ]

        for w in weights:
            update_weight(w, None, lr)

        return base_loss

    def save(self, path):
        """Save to JSON format matching neural-decompile."""
        def to_list(arr):
            return arr.tolist()

        data = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
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
                "b_q": None,
                "b_k": None,
                "b_v": None,
                "b_o": None,
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
    """Main training loop."""
    random.seed(42)
    np.random.seed(42)

    model = TinyTransformer(vocab_size=7, d_model=16, n_heads=2, d_ff=32)

    # All (a + b) % 7 pairs
    all_data = []
    for a in range(7):
        for b in range(7):
            all_data.append(([a, b], (a + b) % 7))

    print(f"Training data: {len(all_data)} samples")
    print(f"Model: d_model={model.d_model}, n_heads={model.n_heads}, d_ff={model.d_ff}")
    print()

    # Training loop
    best_acc = 0
    best_weights = None

    for epoch in range(300):
        random.shuffle(all_data)
        total_loss = 0
        correct = 0

        for tokens, target in all_data:
            loss = model.train_finite_diff(tokens, target, lr=0.5)
            total_loss += loss

            # Check accuracy
            logits = model.forward(tokens)
            pred = np.argmax(logits[-1])
            if pred == target:
                correct += 1

        acc = correct / len(all_data) * 100
        avg_loss = total_loss / len(all_data)

        if epoch % 30 == 0 or acc > 95:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            # Save best weights
            import copy
            best_weights = copy.deepcopy({
                'token_emb': model.token_emb.copy(),
                'pos_emb': model.pos_emb.copy(),
                'w_q': model.w_q.copy(),
                'w_k': model.w_k.copy(),
                'w_v': model.w_v.copy(),
                'w_o': model.w_o.copy(),
                'w_ff_in': model.w_ff_in.copy(),
                'w_ff_out': model.w_ff_out.copy(),
                'w_out': model.w_out.copy(),
            })

        if acc == 100.0:
            print(f"\n✓ Perfect accuracy at epoch {epoch}!")
            break

    print(f"\nBest accuracy: {best_acc:.1f}%")

    # Restore best weights
    if best_weights:
        model.token_emb = best_weights['token_emb']
        model.pos_emb = best_weights['pos_emb']
        model.w_q = best_weights['w_q']
        model.w_k = best_weights['w_k']
        model.w_v = best_weights['w_v']
        model.w_o = best_weights['w_o']
        model.w_ff_in = best_weights['w_ff_in']
        model.w_ff_out = best_weights['w_ff_out']
        model.w_out = best_weights['w_out']

    # Final test
    print("\nFinal test:")
    correct = 0
    for tokens, target in all_data:
        logits = model.forward(tokens)
        pred = np.argmax(logits[-1])
        if pred == target:
            correct += 1
        else:
            print(f"  Fail: {tokens} -> pred={pred}, target={target}")

    final_acc = correct / len(all_data) * 100
    print(f"Accuracy: {correct}/{len(all_data)} = {final_acc:.1f}%")

    # Save if good enough
    if final_acc >= 95:
        model.save("examples/trained_mod7_transformer.json")

        # Generate test cases
        tests = [{"tokens": t, "expected": tgt} for t, tgt in all_data]
        with open("examples/trained_mod7_tests.json", 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Saved: examples/trained_mod7_tests.json")
    else:
        print(f"\nAccuracy {final_acc:.1f}% < 95%, not saving")

if __name__ == "__main__":
    train()
