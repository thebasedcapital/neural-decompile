#!/usr/bin/env python3
"""
Train a minimal 1-layer transformer on modular addition.
Generates weights in neural-decompile's transformer format.
"""

import json
import math
import random

def main():
    # Task: (a + b) mod 7, where a, b ∈ {0,1,2,3,4,5,6}
    # Input: one-hot [a, b] as two tokens
    # Output: one-hot of (a + b) mod 7
    
    vocab_size = 7  # tokens 0-6
    d_model = 16    # small hidden dim
    n_heads = 2
    d_ff = 32
    max_seq_len = 2
    n_layers = 1
    
    # Random init
    random.seed(42)
    
    def rand_mat(rows, cols):
        return [[random.gauss(0, 0.1) for _ in range(cols)] for _ in range(rows)]
    
    def rand_vec(size):
        return [random.gauss(0, 0.1) for _ in range(size)]
    
    # Build transformer
    layers = []
    for _ in range(n_layers):
        layers.append({
            "w_q": rand_mat(d_model, d_model),
            "w_k": rand_mat(d_model, d_model),
            "w_v": rand_mat(d_model, d_model),
            "w_o": rand_mat(d_model, d_model),
            "b_q": None,
            "b_k": None,
            "b_v": None,
            "b_o": None,
            "w_ff_in": rand_mat(d_model, d_ff),
            "b_ff_in": None,
            "w_ff_out": rand_mat(d_ff, d_model),
            "b_ff_out": None,
            "ln1_gamma": [1.0] * d_model,
            "ln1_beta": [0.0] * d_model,
            "ln2_gamma": [1.0] * d_model,
            "ln2_beta": [0.0] * d_model,
        })
    
    transformer = {
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "n_layers": n_layers,
        "max_seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "gelu": False,
        
        "token_emb": rand_mat(vocab_size, d_model),
        "pos_emb": rand_mat(max_seq_len, d_model),
        
        "layers": layers,
        
        "ln_final_gamma": [1.0] * d_model,
        "ln_final_beta": [0.0] * d_model,
        
        "w_out": rand_mat(d_model, vocab_size),
        "b_out": None,
    }
    
    # Save
    with open("examples/transformer_mod7.json", "w") as f:
        json.dump(transformer, f, indent=2)
    
    print(f"Saved transformer_mod7.json")
    print(f"  d_model={d_model}, n_heads={n_heads}, vocab_size={vocab_size}")
    print(f"  {n_layers} layer(s), {d_ff} FFN dim")

if __name__ == "__main__":
    main()