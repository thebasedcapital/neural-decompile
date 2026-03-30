#!/usr/bin/env python3
"""
Train a minimal transformer on modular addition: (a + b) % 7
Uses PyTorch for proper automatic differentiation.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=7, d_model=16, n_heads=2, d_ff=32, max_seq_len=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Attention
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.w_ff_in = nn.Linear(d_model, d_ff, bias=False)
        self.w_ff_out = nn.Linear(d_ff, d_model, bias=False)

        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln_final = nn.LayerNorm(d_model)

        # Output
        self.w_out = nn.Linear(d_model, vocab_size, bias=False)

        # Init
        nn.init.xavier_uniform_(self.token_emb.weight, gain=0.5)
        nn.init.xavier_uniform_(self.pos_emb.weight, gain=0.5)

    def forward(self, tokens):
        # tokens: [seq_len]
        seq_len = tokens.shape[0]
        positions = torch.arange(seq_len, device=tokens.device)

        x = self.token_emb(tokens) + self.pos_emb(positions)

        # Attention block
        residual = x
        x = self.ln1(x)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Multi-head attention
        # Reshape for multi-head: [seq_len, n_heads, head_dim]
        q = q.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)  # [n_heads, seq_len, head_dim]
        k = k.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.n_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)  # [n_heads, seq_len, head_dim]

        # Concatenate heads
        attn_out = attn_out.transpose(0, 1).reshape(seq_len, self.d_model)
        attn_proj = self.w_o(attn_out)
        x = residual + attn_proj

        # FFN block
        residual = x
        x = self.ln2(x)
        ffn_out = self.w_ff_out(F.relu(self.w_ff_in(x)))
        x = residual + ffn_out

        # Final LN and output
        x = self.ln_final(x)
        logits = self.w_out(x)

        return logits

    def save_to_json(self, path):
        """Save to neural-decompile JSON format."""
        def to_list(t):
            return t.detach().cpu().numpy().tolist()

        data = {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.w_ff_in.weight.shape[0],
            "n_layers": 1,
            "max_seq_len": self.pos_emb.weight.shape[0],
            "vocab_size": self.token_emb.weight.shape[0],
            "gelu": False,

            "token_emb": to_list(self.token_emb.weight),
            "pos_emb": to_list(self.pos_emb.weight),

            "layers": [{
                "w_q": to_list(self.w_q.weight.T),  # Transpose for [d_model, d_model]
                "w_k": to_list(self.w_k.weight.T),
                "w_v": to_list(self.w_v.weight.T),
                "w_o": to_list(self.w_o.weight.T),
                "b_q": None,
                "b_k": None,
                "b_v": None,
                "b_o": None,
                "w_ff_in": to_list(self.w_ff_in.weight.T),
                "b_ff_in": None,
                "w_ff_out": to_list(self.w_ff_out.weight.T),
                "b_ff_out": None,
                "ln1_gamma": to_list(self.ln1.weight),
                "ln1_beta": to_list(self.ln1.bias),
                "ln2_gamma": to_list(self.ln2.weight),
                "ln2_beta": to_list(self.ln2.bias),
            }],

            "ln_final_gamma": to_list(self.ln_final.weight),
            "ln_final_beta": to_list(self.ln_final.bias),
            "w_out": to_list(self.w_out.weight.T),
            "b_out": None,
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved: {path}")

def train():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cpu')
    model = TinyTransformer(vocab_size=7, d_model=16, n_heads=2, d_ff=32).to(device)

    # All (a + b) % 7 pairs
    all_data = []
    for a in range(7):
        for b in range(7):
            all_data.append(([a, b], (a + b) % 7))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"Training data: {len(all_data)} samples")
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    print()

    best_acc = 0
    for epoch in range(500):
        random.shuffle(all_data)
        total_loss = 0
        correct = 0

        for tokens, target in all_data:
            tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)
            target_t = torch.tensor(target, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(tokens_t)
            loss = F.cross_entropy(logits[-1:], target_t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                pred = logits[-1].argmax().item()
                if pred == target:
                    correct += 1

        acc = correct / len(all_data) * 100

        if epoch % 50 == 0 or acc > 95:
            print(f"Epoch {epoch:3d}: loss={total_loss/len(all_data):.4f} acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc

        if acc == 100.0:
            print(f"\n✓ Perfect accuracy at epoch {epoch}!")
            break

    print(f"\nBest accuracy: {best_acc:.1f}%")

    # Final test
    print("\nFinal test:")
    model.eval()
    correct = 0
    with torch.no_grad():
        for tokens, target in all_data:
            tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)
            logits = model(tokens_t)
            pred = logits[-1].argmax().item()
            if pred == target:
                correct += 1
            else:
                print(f"  Fail: {tokens} -> pred={pred}, target={target}")

    final_acc = correct / len(all_data) * 100
    print(f"Accuracy: {correct}/{len(all_data)} = {final_acc:.1f}%")

    # Save if good enough
    if final_acc >= 95:
        model.save_to_json("examples/trained_mod7_transformer.json")

        # Generate test cases
        tests = [{"tokens": t, "expected": tgt} for t, tgt in all_data]
        with open("examples/trained_mod7_tests.json", 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Saved: examples/trained_mod7_tests.json")

        return True
    else:
        print(f"\nAccuracy {final_acc:.1f}% < 95%, not saving")
        return False

if __name__ == "__main__":
    train()
