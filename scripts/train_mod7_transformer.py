#!/usr/bin/env python3
"""
Train a minimal transformer on modular addition: (a + b) mod 7.
Uses pure numpy for transparency.
"""

import json
import math
import random
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

class TinyTransformer:
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
        self.token_emb = xavier(vocab_size, d_model) * 0.1
        self.pos_emb = xavier(max_seq_len, d_model) * 0.1

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

    def forward(self, tokens):
        seq_len = len(tokens)

        # Embeddings
        x = self.token_emb[tokens] + self.pos_emb[:seq_len]

        # Pre-norm attention
        h = layer_norm(x, self.ln1_g, self.ln1_b)

        # Q, K, V
        Q = h @ self.w_q
        K = h @ self.w_k
        V = h @ self.w_v

        # Multi-head attention
        attn_out = np.zeros_like(x)
        for i in range(self.n_heads):
            start = i * self.head_dim
            end = start + self.head_dim

            q = Q[:, start:end]
            k = K[:, start:end]
            v = V[:, start:end]

            scores = (q @ k.T) / math.sqrt(self.head_dim)
            attn = softmax(scores)
            attn_out[:, start:end] = attn @ v

        # Output projection + residual
        x = x + attn_out @ self.w_o

        # Pre-norm FFN
        h = layer_norm(x, self.ln2_g, self.ln2_b)
        ffn_out = gelu(h @ self.w_ff_in) @ self.w_ff_out
        x = x + ffn_out

        # Final LN
        x = layer_norm(x, self.ln_final_g, self.ln_final_b)

        # Output projection
        logits = x @ self.w_out
        return logits

    def loss(self, tokens, target):
        logits = self.forward(tokens)
        last_logit = logits[-1]

        # Cross-entropy
        exp = np.exp(last_logit - np.max(last_logit))
        probs = exp / np.sum(exp)
        return -np.log(probs[target] + 1e-10)

    def train_step(self, tokens, target, lr=0.01):
        # Simple gradient estimation via finite differences
        eps = 1e-5
        params = [
            self.token_emb, self.pos_emb,
            self.w_q, self.w_k, self.w_v, self.w_o,
            self.w_ff_in, self.w_ff_out, self.w_out
        ]

        base_loss = self.loss(tokens, target)

        for param in params:
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]

                param[idx] = orig + eps
                loss_plus = self.loss(tokens, target)

                param[idx] = orig - eps
                loss_minus = self.loss(tokens, target)

                grad = (loss_plus - loss_minus) / (2 * eps)
                param[idx] = orig - lr * grad

                it.iternext()

        return base_loss

def train():
    random.seed(42)
    np.random.seed(42)

    model = TinyTransformer(vocab_size=7, d_model=16, n_heads=2, d_ff=32)

    # Generate all training data: (a + b) mod 7
    data = []
    for a in range(7):
        for b in range(7):
            data.append(([a, b], (a + b) % 7))

    print(f"Training data: {len(data)} samples")

    # Train
    for epoch in range(500):
        random.shuffle(data)
        total_loss = 0
        correct = 0

        for tokens, target in data:
            loss = model.train_step(tokens, target, lr=0.05)
            total_loss += loss

            logits = model.forward(tokens)
            pred = np.argmax(logits[-1])
            if pred == target:
                correct += 1

        if epoch % 50 == 0:
            acc = correct / len(data) * 100
            print(f"Epoch {epoch}: loss={total_loss/len(data):.4f} acc={acc:.1f}%")

    # Final test
    correct = 0
    for tokens, target in data:
        logits = model.forward(tokens)
        pred = np.argmax(logits[-1])
        if pred == target:
            correct += 1

    print(f"\nFinal accuracy: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

    # Save weights
    def to_list(arr):
        return arr.tolist()

    weights = {
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "d_ff": model.d_ff,
        "n_layers": 1,
        "max_seq_len": model.max_seq_len,
        "vocab_size": model.vocab_size,
        "gelu": True,

        "token_emb": to_list(model.token_emb),
        "pos_emb": to_list(model.pos_emb),

        "layers": [{
            "w_q": to_list(model.w_q),
            "w_k": to_list(model.w_k),
            "w_v": to_list(model.w_v),
            "w_o": to_list(model.w_o),
            "b_q": None,
            "b_k": None,
            "b_v": None,
            "b_o": None,
            "w_ff_in": to_list(model.w_ff_in),
            "b_ff_in": None,
            "w_ff_out": to_list(model.w_ff_out),
            "b_ff_out": None,
            "ln1_gamma": to_list(model.ln1_g),
            "ln1_beta": to_list(model.ln1_b),
            "ln2_gamma": to_list(model.ln2_g),
            "ln2_beta": to_list(model.ln2_b),
        }],

        "ln_final_gamma": to_list(model.ln_final_g),
        "ln_final_beta": to_list(model.ln_final_b),
        "w_out": to_list(model.w_out),
        "b_out": None,
    }

    with open("examples/trained_mod7_transformer.json", "w") as f:
        json.dump(weights, f, indent=2)

    print("\nSaved: examples/trained_mod7_transformer.json")

    # Generate test cases
    tests = [{"tokens": list(t), "expected": tgt} for t, tgt in data]
    with open("examples/trained_mod7_tests.json", "w") as f:
        json.dump(tests, f, indent=2)

    print("Saved: examples/trained_mod7_tests.json")

if __name__ == "__main__":
    train()