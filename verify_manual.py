#!/usr/bin/env python3
"""Manual verification of decompiled transformer on parity task."""
import json
import math

# Load test cases
with open('examples/parity_transformer_tests.json') as f:
    tests = json.load(f)

# Load original weights
with open('examples/parity_transformer.json') as f:
    weights = json.load(f)

# Decompiled weights (from nd output)
token_emb_decompiled = [[0.0, 0.0], [1.0, 0.0]]
pos_emb_decompiled = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
w_q_0 = [-0.2571, -0.2123, -0.5551, -1.4796]
w_k_0 = [0.6979, 0.2979, 0.8150, -0.6155]
w_v_0 = [-0.2468, 0.0, 0.5529, 0.7626]
w_o_0 = [1.0, 0.5597, 1.0, 0.0]
w_ff_in_0 = [-0.4756, -0.2617, 0.4024, -0.3190, 0.4798, 0.7276, 1.0, 0.5600]
w_ff_out_0 = [0.7442, 1.2803, 0.0, -1.0, 0.4139, 0.0, 0.3134, 0.0]
ln1_gamma_0 = [1, 1]
ln1_beta_0 = [0, 0]
ln2_gamma_0 = [1, 1]
ln2_beta_0 = [0, 0]
w_out_decompiled = [-1.0, -0.1821, 0.7617, 0.0]
b_out_decompiled = [0.3346, 0.0]

def layer_norm(x, gamma, beta):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    std = math.sqrt(var + 1e-5)
    return [(x[i] - mean) / std * gamma[i] + beta[i] for i in range(len(x))]

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

def decompiled(tokens):
    """Transformer decompilation: 1 layers, 1 heads"""
    seq_len = len(tokens)
    assert seq_len <= 3
    # Token + position embeddings
    hidden = []
    for i, tok in enumerate(tokens):
        emb = token_emb_decompiled[tok][:]  # copy
        for j in range(2):
            emb[j] += pos_emb_decompiled[i][j]
        hidden.append(emb)

    # Layer 0
    # Pre-norm attention
    x = [layer_norm(h, ln1_gamma_0[:], ln1_beta_0[:]) for h in hidden]
    # Multi-head attention (1 heads, head_dim=2)
    q = [[0.0] * 2 for _ in range(seq_len)]
    k = [[0.0] * 2 for _ in range(seq_len)]
    v = [[0.0] * 2 for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(2):
            for h_idx in range(2):
                q[i][j] += x[i][h_idx] * w_q_0[h_idx * 2 + j]
                k[i][j] += x[i][h_idx] * w_k_0[h_idx * 2 + j]
                v[i][j] += x[i][h_idx] * w_v_0[h_idx * 2 + j]
    attn_out = [[0.0] * 2 for _ in range(seq_len)]
    for h_idx in range(1):
        h_off = h_idx * 2
        # Compute attention scores
        scores = [[0.0] * seq_len for _ in range(seq_len)]
        for i in range(seq_len):
            for j in range(seq_len):
                for d in range(2):
                    scores[i][j] += q[i][h_off + d] * k[j][h_off + d]
                scores[i][j] /= 1.4142135623730951  # sqrt(head_dim)
        # Softmax
        attn_weights = [softmax(row) for row in scores]
        # Apply attention to values
        for i in range(seq_len):
            for d in range(2):
                for j in range(seq_len):
                    attn_out[i][h_off + d] += attn_weights[i][j] * v[j][h_off + d]
    # Output projection
    attn_proj = [[0.0] * 2 for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(2):
            for h_idx in range(2):
                attn_proj[i][j] += attn_out[i][h_idx] * w_o_0[h_idx * 2 + j]
    # Residual
    for i in range(seq_len):
        for j in range(2):
            hidden[i][j] += attn_proj[i][j]
    # Pre-norm FFN
    x = [layer_norm(h, ln2_gamma_0[:], ln2_beta_0[:]) for h in hidden]
    ffn_out = []
    for h in x:
        # FFN layer 1
        hidden_ff = [0.0] * 4
        for j in range(4):
            for i in range(2):
                hidden_ff[j] += h[i] * w_ff_in_0[i * 4 + j]
        # Activation
        hidden_ff = [max(0, v) for v in hidden_ff]
        # FFN layer 2
        out = [0.0] * 2
        for j in range(2):
            for i in range(4):
                out[j] += hidden_ff[i] * w_ff_out_0[i * 2 + j]
        ffn_out.append(out)
    # Residual
    for i in range(seq_len):
        for j in range(2):
            hidden[i][j] += ffn_out[i][j]

    # Output projection
    logits = []
    for h in hidden:
        out = []
        for j in range(2):
            s = 0.0
            for i in range(2):
                s += h[i] * w_out_decompiled[i * 2 + j]
            s += b_out_decompiled[j]
        out.append(s)
        logits.append(out)
    return logits

def predict(logits):
    """Get prediction from logits (argmax of last token)"""
    final_logits = logits[-1]
    return final_logits.index(max(final_logits))

# Run all tests
print("Manual verification of decompiled transformer:")
print("=" * 50)
passed = 0
for i, test in enumerate(tests):
    tokens = test['tokens']
    expected = test['expected']
    logits = decompiled(tokens)
    pred = predict(logits)
    status = "PASS" if pred == expected else "FAIL"
    if pred == expected:
        passed += 1
    print(f"Test {i+1}: tokens={tokens}, expected={expected}, pred={pred} [{status}]")

print("=" * 50)
print(f"Result: {passed}/{len(tests)} passed ({100*passed/len(tests):.0f}%)")
if passed == len(tests):
    print("✓ PERFECT — manual verification confirms 100% accuracy")
else:
    print("✗ FAILED — decompiled code does not match expectations")
