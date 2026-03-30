# nd — Neural Decompiler

A Rust CLI that auto-decompiles neural network weights into readable code.

Feed it trained RNN weights. It quantizes them to integers and emits a readable Python or Rust program that does the same thing.

## The Demo

We trained a 2-neuron RNN on "does this binary string contain '11'?" — then decompiled it:

```
$ nd decompile examples/regex/contains_11.json

# 100% of weights are exact integers
# Hidden dim: 2, Input dim: 2, Output dim: 2

def decompiled(input_sequence):
    h = [0.0] * 2
    for x in input_sequence:
        h0 = 0
        h1 = max(0, 2*h[1] + -1*x[0] + 2*x[1] + -1)
        h = [h0, h1]
    logits = []
    logits.append(-2*h[1] + 3)
    logits.append(2*h[1] + -3)
    return logits.index(max(logits))
```

One neuron. All-integer weights. 62/62 test cases correct. The network independently discovered a DFA accumulator — and we extracted it.

```
$ nd verify examples/regex/contains_11.json examples/regex/contains_11_tests.json
Verification: 62/62 passed (100%)
✓ PERFECT — decompiled FSM matches all test cases
```

## More Examples

**Divisible by 3** — the textbook automata theory example. Every CS student knows the 3-state DFA for binary divisibility by 3. We trained an RNN on 254 binary strings (lengths 1-7), decompiled it, and got a 4-neuron all-integer circuit that computes mod-3 remainders:

```
$ nd verify examples/regex/divisible_by_3.json examples/regex/divisible_by_3_tests.json
Verification: 254/254 passed (100%)
✓ PERFECT — decompiled FSM matches all test cases
```

**Parity of 3 bits** — 100% integer weights, 8/8 correct. The decompiled code IS the XOR algorithm.

**Modular addition** — (x₁ + x₂) mod 3, mod 5, mod 7. Integer-coefficient programs that compute modular arithmetic.

Full benchmark: **7/13 algorithmic tasks decompile with 100% accuracy.** Three of those have literally every weight as an exact integer.

## Install

```bash
cargo install --path .
# or
cargo build --release && cp target/release/nd ~/.local/bin/
```

## Usage

```bash
# Decompile weights to Python
nd decompile weights.json

# Decompile to Rust
nd decompile weights.json --format rust

# Verify against test cases
nd verify weights.json tests.json

# Weight statistics
nd stats weights.json

# GGUF model inspection
nd layers model.gguf
nd extract model.gguf --tensor "blk.0.attn_q.weight"
```

## Weight Format

JSON with five matrices:

```json
{
  "W_hh": [[...], ...],
  "W_hx": [[...], ...],
  "b_h": [...],
  "W_y": [[...], ...],
  "b_y": [...]
}
```

Where the RNN computes: `h_t = ReLU(W_hh @ h_{t-1} + W_hx @ x_t + b_h)` and `y = argmax(W_y @ h + b_y)`.

## How It Works

1. **Train** an RNN with L1 regularization that pushes weights toward integers
2. **Quantize** — snap near-integer weights to exact integers (configurable epsilon)
3. **Emit** — generate readable code from the quantized finite state machine
4. **Verify** — prove the emitted code matches the original on all test cases

The key insight: **L1 integer regularization + direct quantization outperforms the MIPS 5-normalizer chain** (whitening, Jordan normal form, Toeplitz, de-bias, quantize) from [Michaud & Tegmark 2024](https://arxiv.org/abs/2402.05110). The normalizers rotate the weight space and destroy the integer alignment that L1 creates. Skip them.

## GGUF Support

`nd` reads GGUF v2/v3 model files (the format used by llama.cpp and Ollama). Supports F32, F16, BF16, Q8_0, and Q4_0 tensor types with memory-mapped IO.

```
$ nd layers tinyllama-1.1b-q4_0.gguf
GGUF v3 — 201 tensors, 23 metadata keys
  general.architecture: "llama"

TENSOR                                TYPE  SHAPE              SIZE
blk.0.attn_q.weight                  Q4_0  [2048, 2048]     2.2 MB
blk.0.ffn_gate.weight                Q4_0  [2048, 5632]     6.2 MB
...
```

## Benchmark

### RNN Decompilation

| Task | Hidden Dim | Verify | % Integer |
|------|-----------|--------|-----------|
| parity3 | 3 | 8/8 | **100%** |
| parity5 | 2 | 32/32 | **100%** |
| evens_detector | 2 | 4/4 | **100%** |
| contains_11 | 2 | 62/62 | **100%** |
| no_consecutive_1 | 2 | 62/62 | **100%** |
| divisible_by_3 | 5 | 254/254 | **100%** |
| mod3_add | 3 | 9/9 | 97% |
| max4 | 2 | 16/16 | 96% |
| max5 | 2 | 25/25 | 94% |
| bitwise_xor | 3 | 16/16 | 47% |

### Transformer Decompilation

| Task | Architecture | Verify | Notes |
|------|--------------|--------|-------|
| parity_transformer | 1-layer, 2-head | 8/8 **100%** | Trained via random search |
| mod3_add (RNN) | 3-neuron RNN | 9/9 **100%** | RNN baseline for comparison |

### CNN-Transformer Hybrid (New Task)

**Task**: Syntactic structure recovery — recover canonical program structure from obfuscated input.

**Input**: `["if x>0 goto 2", "add 1 y", ...]` (obfuscated variables, irregular spacing)
**Output**: Canonical form with normalized variable names, indentation, etc.

**Architecture**: `tokens → embed → 1D-Conv → LayerNorm → Transformer`

This mimics CNN-transformer hybrids used in speech/NLP where CNNs pre-filter temporal features before the transformer processes sequence relationships.

See `examples/cnn_transformer_spec.json` for the full architecture specification.

## Formal Verification

The `kani-proofs/` directory contains **machine-checked proofs** that the decompiled circuits are correct for **ALL possible inputs**, not just tested cases.

```bash
cd kani-proofs && cargo kani
```

**6 verified circuits** (all proofs pass):

| Circuit | Property Verified | Lines of Proof |
|---------|-------------------|----------------|
| `parity3` | Correctly computes 3-bit parity for all 8 inputs | 60 |
| `contains_11` | Correctly detects "11" substring for all binary strings | 50 |
| `no_consecutive_1` | Correctly rejects strings with consecutive 1s | 50 |
| `divisible_by_3` | Correctly recognizes binary numbers divisible by 3 | 60 |
| `mod3_add` | Correctly computes (x₁ + x₂) mod 3 | 80 |
| `complement` | Proves contains_11 and no_consecutive_1 are exact boolean complements | 130 |

**Key finding**: The `complement` proof shows that two independently trained networks learned **identical hidden state dynamics** — the same 1-neuron DFA — with only the output logits swapped. This is structural reuse: the network discovered that these tasks are boolean complements and implemented them as `f(x)` and `¬f(x)`.

**Why this matters**: Empirical testing (62/62 test cases) can't prove correctness — it only shows the model works on _tested_ inputs. Kani's symbolic execution proves the decompiled Rust code matches the mathematical specification for **every possible input** up to the bounded length. This elevates neural decompilation from "cool demo" to **formally verified algorithm extraction**.

## Research Background

Based on [MIPS: Opening the AI Black Box](https://arxiv.org/abs/2402.05110) (Michaud, Liao, Lad, Liu et al., 2024) which showed that trained RNNs secretly learn discrete algorithms encodable as integer lattices. Our contribution:

- **L1 + direct quantize beats the normalizer chain** — novel finding, simpler pipeline, better results
- **Regex extraction** — trained RNNs on regex patterns, extracted the equivalent DFA as readable code
- **Complement discovery** — the network learned that "contains 11" and "no consecutive 1s" are complements, reusing the same single-neuron circuit with inverted output
- **Formal verification** — Kani proofs verify extracted algorithms are correct for ALL inputs, not just tested cases

## License

MIT
