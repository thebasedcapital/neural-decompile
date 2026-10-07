# Proof results and exact domains

Executed with **Kani 0.68.0 / CBMC 6.11.0** on this checkout after regenerating circuits from the current release binary:

```text
./check_circuits.sh --update
./check_circuits.sh
  OK: src/generated.rs matches current nd output for 5 circuits
cargo kani -j 2 --output-format terse
  Complete - 33 successfully verified harnesses, 0 failures, 33 total.
```

The final generation/check/proof command completed in 32.36 seconds on the local workstation. Kani reported its toolchain-injected `register_tool` unstable-feature warning; it did not report a failed property. The separate proof-crate unit suite previously ran 12 tests successfully.

## What the 33 harnesses establish

| Module | Harnesses | Domain and conclusion |
|---|---:|---|
| `contains_11` | 9 | Exhaustive lengths 0–5 over bit-0/bit-1/padding; DFA relation; inductive correctness of a saturated transition system; literal `i64` step/output bounds through 62 steps |
| `no_consecutive_1` | 6 | Exhaustive lengths 0–5; saturated-system base/step/output relation; literal `i64` step/output bounds through 62 steps |
| `complement` | 4 | Complementarity through five symbols; identical transition and complementary-output lemmas over declared integer ranges |
| `divisible_by_3` | 5 | Correct on lengths 0–7 with trailing padding; **incorrect at `10100001`**; base/closure/output properties of its actual nine-state reachable set |
| `parity3` | 3 | Correct at its fixed three-bit domain; witnessed failures at other sequence lengths, not an arbitrary-length parity proof |
| `mod3_add` | 1 | All nine pairs of one-hot ternary digits, fixed sequence length two |
| `equiv` | 5 | Mechanically split step/output agrees with emitted function: binary/padding sequences through length four; all nine ternary pairs for modular addition |

The modulo-three counterexample harness **passes because it proves a mismatch**, not because the original model implements divisibility for every length. There can be no sound unrestricted correctness proof of that original claim.

## Source linkage

`check_circuits.sh` obtains `rust-kani` output for five checked-in weight files, extracts each complete `decompiled` function, and mechanically splits its loop body and output expression. `src/generated.rs` is generated, not a hand-entered substitute. Running the check without `--update` rejects drift. The `equiv` harnesses additionally compare the split functions with the emitted functions over their declared bounded domains.

`nd decompile --format rust-kani` now emits the circuit only; it does not append a placeholder specification or claim a proof for arbitrary weights. The actual specifications and assumptions live in this crate.

## Saturation is not unlimited machine arithmetic

The hand-written induction harnesses use `CAP = 2**60` and explicitly prove properties of a saturated machine. `clamp_commutes_with_step` checks a finite machine-integer range; the source comments distinguish the additional mathematical argument outside that range. Neither those lemmas nor the bounded split checks certify arbitrary-length execution of the growing raw `i64` recurrence. The literal substring recurrence can overflow after sufficiently many symbols.

For a separate, automatically discovered cap-2 quotient with an SMT query over **all nonnegative mathematical integers**, run `make breakthrough`. Its certificates establish the homomorphism without a finite hidden-state assumption. The emitted three-state machines do not execute the growing recurrence. See [automatic extraction and repair](../results/algorithm-extraction.md).

The repaired modulo-three model in `results/automata/` is certified separately by exact reachable-product closure and an arithmetic bound; it is not substituted for the original model in these Kani harnesses.

## Reproduction

```bash
cargo install --locked kani-verifier
cargo kani setup
make proofs
```

No harness in this directory certifies the BCI decoder, a full LLM, a medical device, or unrestricted floating-point execution. Task accuracy, emitted-code fidelity, specification equivalence, and overflow-freedom are different properties.
