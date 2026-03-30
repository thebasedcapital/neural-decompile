#!/usr/bin/env python3
"""
Generate test cases for transformer verification.
Uses the random transformer weights to create test cases
where the "expected" output is whatever the model outputs.
This tests that decompiled code matches original, not correctness.
"""

import json

# Load transformer
with open("examples/transformer_mod7.json") as f:
    t = json.load(f)

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

def forward(tokens, t):
    d_model = t["d_model"]
    n_heads = t["n_heads"]
    d_ff = t["d_ff"]
    head_dim = d_model // n_heads

    # Embeddings
    token_emb = np.array(t["token_emb"])
    pos_emb = np.array(t["pos_emb"])

    x = token_emb[tokens] + pos_emb[:len(tokens)]

    # Layer
    layer = t["layers"][0]
    w_q = np.array(layer["w_q"])
    w_k = np.array(layer["w_k"])
    w_v = np.array(layer["w_v"])
    w_o = np.array(layer["w_o"])
    w_ff_in = np.array(layer["w_ff_in"])
    w_ff_out = np.array(layer["w_ff_out"])
    ln1_g = np.array(layer["ln1_gamma"])
    ln1_b = np.array(layer["ln1_beta"])
    ln2_g = np.array(layer["ln2_gamma"])
    ln2_b = np.array(layer["ln2_beta"])

    # Pre-norm attention
    h = layer_norm(x, ln1_g, ln1_b)

    Q = h @ w_q
    K = h @ w_k
    V = h @ w_v

    attn_out = np.zeros_like(x)
    for i in range(n_heads):
        start = i * head_dim
        end = start + head_dim
        q = Q[:, start:end]
        k = K[:, start:end]
        v = V[:, start:end]
        scores = (q @ k.T) / np.sqrt(head_dim)
        attn = softmax(scores)
        attn_out[:, start:end] = attn @ v

    x = x + attn_out @ w_o

    # Pre-norm FFN
    h = layer_norm(x, ln2_g, ln2_b)
    ffn_out = gelu(h @ w_ff_in) @ w_ff_out
    x = x + ffn_out

    # Final LN
    ln_final_g = np.array(t["ln_final_gamma"])
    ln_final_b = np.array(t["ln_final_beta"])
    x = layer_norm(x, ln_final_g, ln_final_b)

    # Output
    w_out = np.array(t["w_out"])
    logits = x @ w_out

    return logits

# Generate test cases
tests = []
for a in range(7):
    for b in range(7):
        tokens = [a, b]
        logits = forward(tokens, t)
        last_logits = logits[-1]
        expected = int(np.argmax(last_logits))
        tests.append({
            "tokens": tokens,
            "expected": expected
        })
        print(f"tokens={tokens} -> expected={expected} logits={last_logits[:4].round(2)}...")

with open("examples/transformer_mod7_tests.json", "w") as f:
    json.dump(tests, f, indent=2)

print(f"\nGenerated {len(tests)} test cases")