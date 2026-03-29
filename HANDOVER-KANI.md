# Handover: Formal Verification with Kani

## What This Is

The `nd` toolchain decompiles trained RNN weight matrices into readable Rust/Python FSMs. All 13 benchmark tasks produce PERFECT decompilations verified against test cases. But test-case verification only proves correctness on *sampled* inputs — it can't prove the FSM is correct for *all* possible inputs.

**Kani** is a Rust model checker (based on CBMC) that can prove properties about Rust code for all possible inputs. The goal: take the Rust code emitted by `nd decompile --format rust` and formally verify that it implements the intended algorithm.

## What Exists Today

### Emitted Rust Code Shape

Every decompiled circuit produces this structure:

```rust
fn decompiled(input_sequence: &[Vec<f64>]) -> usize {
    let mut h = vec![0.0_f64; N];       // N = hidden_dim
    for x in input_sequence {
        let h0 = (EXPR).max(0.0);       // ReLU(linear combo of h + x)
        let h1 = (EXPR).max(0.0);
        // ...
        h = vec![h0, h1, ...];
    }
    let logits: Vec<f64> = vec![EXPR, EXPR, ...];  // M = output_dim
    logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}
```

Key properties:
- All 100%-integer circuits use only integer coefficients (1, -2, 3, etc.) with f64 arithmetic
- ReLU is just `.max(0.0)` — no branches, no discontinuities beyond the clamp
- Input is one-hot encoded: each `x` is a vector like `[1,0]` or `[0,1]` for binary, `[1,0,0]`/`[0,1,0]`/`[0,0,1]` for ternary
- Output is argmax of the logit vector
- Sequence length is fixed per task (e.g., parity3 = 3 steps, mod3_add = 2 steps)

### The Cleanest Circuits (Start Here)

**contains_11** (1 neuron, 100% integer, sliced):
```rust
fn contains_11(input_sequence: &[Vec<f64>]) -> usize {
    let mut h = vec![0.0_f64; 1];
    for x in input_sequence {
        let h0 = (2.0 * h[0] + -1.0 * x[0] + 2.0 * x[1] + -1.0).max(0.0);
        h = vec![h0];
    }
    let logits: Vec<f64> = vec![-2.0 * h[0] + 3.0, 2.0 * h[0] + -3.0];
    logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
}
```

This should be provably equivalent to: "return 1 if input contains consecutive 1s, else 0".

**parity circuits** (2 neurons, 100% integer): should be provably equivalent to `sum(bits) % 2`.

**divisible_by_3** (3 neurons, textbook mod-3 DFA): should be provably equivalent to `binary_to_int(bits) % 3 == 0`.

### What `nd verify` Does Today

Runs the FSM on every test case and checks argmax matches. Test cases are exhaustive for small input spaces (parity3: all 8 inputs, mod3_add: all 9, contains_11: all binary strings length 1-6 = 62 inputs). So for these tasks, `nd verify` is already essentially a proof — but it's brute-force, not structural. Kani would give a real proof certificate.

## The Plan

### Phase 1: Prove Contains_11

1. **Install Kani**: `cargo install --locked kani-verifier && cargo kani setup`

2. **Create a verification crate** at `~/Projects/neural-decompile/kani-proofs/`:
```
kani-proofs/
  Cargo.toml          # depends on kani
  src/
    lib.rs            # trait ProveEquivalent
    contains_11.rs    # paste decompiled fn + kani::proof
    parity3.rs
    divisible_by_3.rs
```

3. **Write the proof harness** for contains_11:
```rust
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(7)]  // max sequence length = 6
fn verify_contains_11() {
    // Generate arbitrary binary sequence of length 1-6
    let len: usize = kani::any();
    kani::assume(len >= 1 && len <= 6);

    let mut input_sequence = Vec::new();
    let mut has_consecutive_ones = false;
    let mut prev_bit = 0u8;

    for i in 0..6 {
        if i >= len { break; }
        let bit: u8 = kani::any();
        kani::assume(bit <= 1);

        if bit == 1 && prev_bit == 1 {
            has_consecutive_ones = true;
        }
        prev_bit = bit;

        // One-hot encode
        if bit == 0 {
            input_sequence.push(vec![1.0, 0.0]);
        } else {
            input_sequence.push(vec![0.0, 1.0]);
        }
    }

    let result = contains_11_decompiled(&input_sequence);
    let expected = if has_consecutive_ones { 1 } else { 0 };

    assert_eq!(result, expected);
}
```

4. **Run**: `cargo kani --function verify_contains_11`

### Phase 2: Prove Parity and Mod Circuits

Same pattern: paste decompiled Rust, write reference implementation, prove equivalence.

**Challenges for larger circuits:**
- Kani unwind depth scales with sequence length (bitwise_xor needs unwind=4, but mod7_add inputs are one-hot of size 7 × 2 steps)
- Non-integer weights (bitwise_xor has 0.69 coefficient) make proofs harder — Kani uses CBMC which handles floats via bit-blasting, which can be slow
- For circuits with non-integer weights, consider proving on the *integer-snapped* version and bounding the residual error separately

### Phase 3: Auto-Generate Proof Harnesses

New `nd` command: `nd prove <weights.json> <tests.json> --spec "parity" -o proof.rs`

This would:
1. Decompile + slice the circuit
2. Look up a known spec (parity, mod_add, contains_pattern, divisible_by_N)
3. Generate the Kani proof harness automatically
4. Optionally run `cargo kani` and report the result

The spec library would be a set of reference implementations:
```rust
fn spec_parity(bits: &[u8]) -> usize { bits.iter().sum::<u8>() as usize % 2 }
fn spec_mod_add(a: usize, b: usize, base: usize) -> usize { (a + b) % base }
fn spec_contains(seq: &[u8], pattern: &[u8]) -> bool { seq.windows(pattern.len()).any(|w| w == pattern) }
fn spec_divisible(bits: &[u8], n: usize) -> bool { bits_to_int(bits) % n == 0 }
```

### Phase 4: Prove Novel Properties

Beyond equivalence:
- **Termination**: trivially true (fixed-length loop), but good to have Kani confirm
- **Bounded hidden state**: prove that hidden activations stay within [0, K] for all inputs (useful for fixed-point deployment)
- **Monotonicity**: for contains_11, once h0 > 1.5, it stays > 1.5 — prove this structural property of the DFA
- **Complement duality**: formally prove that contains_11 and no_consecutive_1 always return opposite outputs

## Known Gotchas

1. **f64 in Kani**: Kani supports floats but they're expensive for the solver. Since our best circuits are 100% integer, consider casting to integer arithmetic for the proof (the integers are exact in f64 up to 2^53).

2. **Variable-length sequences**: Kani needs a fixed unwind bound. Use `kani::assume(len <= MAX)` and set `#[kani::unwind(MAX + 1)]`. Start small (MAX=6 for contains_11) and increase.

3. **Vec allocations**: Kani tracks heap allocations. The `h = vec![h0, h1]` pattern works but may be slow for the solver. Consider rewriting to use fixed-size arrays: `let h: [f64; 2] = [h0, h1]` — this is a natural emit.rs improvement anyway.

4. **The argmax pattern**: `logits.iter().enumerate().max_by(...)` is correct but complex for the solver. For 2-class problems, simplify to `if logits[0] > logits[1] { 0 } else { 1 }`.

## Suggested Emit Changes

To make Kani happier, add a `--format rust-kani` emit mode that:
- Uses fixed-size arrays instead of Vec
- Simplifies argmax for small output dims
- Adds `#[cfg(kani)]` proof harness template
- Uses integer arithmetic where all weights are integers

## Files to Read

| File | Why |
|------|-----|
| `src/emit.rs` | Current Rust code emitter — modify for kani-friendly output |
| `src/slice.rs` | Slice before proving — smaller circuit = faster solver |
| `src/xray.rs` | Hybrid decomposition tells you which neurons are pure integer |
| `src/quantize.rs` | The epsilon/snapping logic — Kani proofs should use the quantized weights |
| `batch_weights/*.json` | The actual weight files to prove |
| `batch_weights/*_tests.json` | Test cases — Kani should prove what these test empirically |

## Priority Order

1. **contains_11** — 1 neuron, 100% integer, simplest possible proof
2. **parity3/parity5** — 2 neurons, 100% integer, well-known algorithm
3. **divisible_by_3** — 3 neurons, textbook DFA, beautiful if proven
4. **Complement proof** — contains_11 XOR no_consecutive_1 = always true
5. **mod3_add** — 100% integer after L1, first multi-class proof
6. **bitwise_xor** — has 0.69 residual, first proof with non-integer weights

Good luck. Have fun. Follow your soul.
