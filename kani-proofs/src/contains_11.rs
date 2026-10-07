//! contains_11 — does the bit stream contain two consecutive 1-bits?
//!
//! The circuit under proof is `crate::generated::contains_11`, i.e. what
//! `nd decompile examples/regex/contains_11.json --format rust-kani` prints
//! (see `check_circuits.sh`). With the dead neuron removed:
//!
//! ```text
//!   h' = max(2*h - x0 + 2*x1 - 1, 0)        logits = [-2h + 3, 2h - 3]   => class 1  iff  h >= 2
//! ```
//!
//! Input symbols: `(1,0)` = bit 0, `(0,1)` = bit 1, `(0,0)` = padding (a no-op for the spec).
//!
//! # What is proved
//!
//! 1. **Bounded, exhaustive** (`verify_contains_11`): every sequence of 5 symbols from
//!    {bit0, bit1, pad} (padding anywhere), 0..=5 steps.
//! 2. **Inductive, any length** (`unbounded_*`): relate the hidden value `h` to the 3-state
//!    spec DFA `(seen, prev1)` by the invariant
//!
//!    ```text
//!      inv(h, s) :=   s.seen            =>  h >= 2     (absorbing "accept" region)
//!                 ∧  !s.seen ∧ s.prev1  =>  h == 1     (last real bit was 1)
//!                 ∧  !s.seen ∧ !s.prev1 =>  h == 0     (last real bit was 0 / nothing yet)
//!    ```
//!
//!    and prove with Kani: base (`inv(0, init)`), step (for ANY h, ANY DFA state, ANY symbol
//!    incl. padding: `inv` is preserved) and output (`inv(h,s)` ⇒ circuit output == DFA accept).
//!    Together this is correctness for every length — see the precise scope note on overflow below.
//! 3. **Overflow-faithful version on the literal i64 circuit** (`literal_*`): the doubling
//!    `h' = 2h - 1` in the absorbing region exceeds i64 after 62 steps. Strengthening the
//!    invariant with `h <= 2^n - 1` (n = steps taken) proves, on the literal emitted code with
//!    Kani's overflow checks on, correctness and overflow-freedom for all sequences of ≤ 62 steps.
//!
//! # Overflow scope (be precise)
//!
//! * The *saturated* machine `step_sat(h) = min(step(h), CAP)` (CAP = 2^60) has a closed state
//!   space `[0, CAP]` so the induction (2) covers every length for it. It models the circuit over
//!   the mathematical integers because clamping commutes with the step
//!   (`clamp_commutes_with_step`, Kani-checked for h <= 2^61; for h > 2^61 the step is >= 2h-2 > CAP
//!   on both sides — a one-line argument, not machine-checked).
//! * The literal i64 code overflows (debug: panic, release: wrap) on the 63rd step of the all-ones
//!   input; nothing here claims anything about that region. The f64 code `nd decompile --format rust`
//!   emits does not wrap (it reaches +inf, which stays >= 2).

use crate::common::CAP;
#[cfg(test)]
use crate::common::valid_symbol;
use crate::generated::contains_11 as nd;

/// Spec DFA state: have we already seen "11", and was the last *real* bit a 1?
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(kani, derive(kani::Arbitrary))]
pub struct Dfa {
    pub seen: bool,
    pub prev1: bool,
}

pub const DFA_INIT: Dfa = Dfa { seen: false, prev1: false };

/// Spec transition; the body is the loop body of `spec_contains_11` below.
pub fn dfa_step(s: Dfa, x0: i64, x1: i64) -> Dfa {
    let is_one = x0 == 0 && x1 == 1;
    let is_bit = x0 + x1 > 0;
    Dfa {
        seen: s.seen || (is_one && s.prev1),
        prev1: if is_bit { is_one } else { s.prev1 },
    }
}

pub fn dfa_accept(s: Dfa) -> usize {
    s.seen as usize
}

/// The abstraction relation between hidden state and spec state.
pub fn inv(h: i64, s: Dfa) -> bool {
    if s.seen {
        h >= 2
    } else if s.prev1 {
        h == 1
    } else {
        h == 0
    }
}

/// Saturated step: the emitted `step`, clamped at CAP. Only ever called with 0 <= h <= CAP.
pub fn step_sat(h: i64, x0: i64, x1: i64) -> i64 {
    nd::step([h], x0, x1)[0].min(CAP)
}

/// Independent reference (a direct transcription of the English spec): scan the active bits and
/// report whether two consecutive real bits are both 1. Padding is skipped.
pub fn spec_contains_11(seq: &[[i64; 2]], len: usize) -> usize {
    let mut prev_is_one = false;
    let mut i = 0;
    while i < len {
        let is_one = seq[i][0] == 0 && seq[i][1] == 1;
        if is_one && prev_is_one {
            return 1;
        }
        let is_bit = seq[i][0] + seq[i][1] > 0;
        if is_bit {
            prev_is_one = is_one;
        }
        i += 1;
    }
    0
}

#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::common::any_symbol;

    const MAX_LEN: usize = 5;

    /// BOUNDED (exhaustive): nd's verbatim `decompiled` == spec for every sequence of
    /// length `len <= 5` over {bit0, bit1, pad}, padding in any position.
    #[kani::proof]
    #[kani::unwind(7)]
    fn verify_contains_11() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);
        let mut seq = [[0i64; 2]; MAX_LEN];
        let mut i = 0;
        while i < MAX_LEN {
            let (x0, x1) = any_symbol();
            seq[i] = [x0, x1];
            i += 1;
        }
        assert_eq!(nd::decompiled(&seq, len), spec_contains_11(&seq, len));
    }

    /// BOUNDED: the 3-state DFA used as the induction target agrees with the independent
    /// reference loop (so the unbounded proofs below are about the right spec).
    #[kani::proof]
    #[kani::unwind(7)]
    fn dfa_matches_reference_spec() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);
        let mut seq = [[0i64; 2]; MAX_LEN];
        let mut s = DFA_INIT;
        let mut i = 0;
        while i < MAX_LEN {
            let (x0, x1) = any_symbol();
            seq[i] = [x0, x1];
            if i < len {
                s = dfa_step(s, x0, x1);
            }
            i += 1;
        }
        assert_eq!(dfa_accept(s), spec_contains_11(&seq, len));
    }

    // ------------------------------------------------------------------------------------
    // UNBOUNDED (any sequence length) — induction on the saturated/ideal-integer machine.
    // ------------------------------------------------------------------------------------

    /// Base: the initial hidden state (0) is related to the initial DFA state.
    #[kani::proof]
    fn unbounded_base() {
        assert!(inv(0, DFA_INIT));
    }

    /// Step: for ANY related (h, s) and ANY symbol (bit0 / bit1 / padding), one circuit step and
    /// one DFA step give related states, and the state stays inside [0, CAP].
    #[kani::proof]
    fn unbounded_step() {
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= CAP);
        kani::assume(inv(h, s));
        let (x0, x1) = any_symbol();

        let h2 = step_sat(h, x0, x1);
        let s2 = dfa_step(s, x0, x1);
        assert!(0 <= h2 && h2 <= CAP);
        assert!(inv(h2, s2));
    }

    /// Output: related states give the DFA's verdict.
    #[kani::proof]
    fn unbounded_output() {
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= CAP);
        kani::assume(inv(h, s));
        assert_eq!(nd::output([h]), dfa_accept(s));
    }

    /// Clamping commutes with the emitted step (justifies reading the saturated machine as the
    /// integer circuit). Checked for 0 <= h <= 2^61; beyond that both sides are CAP because
    /// step(h) >= 2h - 2 > CAP.
    #[kani::proof]
    fn clamp_commutes_with_step() {
        let h: i64 = kani::any();
        kani::assume(0 <= h && h <= (1i64 << 61));
        let (x0, x1) = any_symbol();
        let lhs = nd::step([h], x0, x1)[0].min(CAP);
        let rhs = step_sat(h.min(CAP), x0, x1);
        assert_eq!(lhs, rhs);
    }

    /// Absorption (the old `verify_monotonicity`, now for any h): once h >= 2 it stays >= 2
    /// under every symbol.
    #[kani::proof]
    fn unbounded_absorbing() {
        let h: i64 = kani::any();
        kani::assume(2 <= h && h <= CAP);
        let (x0, x1) = any_symbol();
        assert!(step_sat(h, x0, x1) >= 2);
    }

    // ------------------------------------------------------------------------------------
    // LITERAL i64 circuit, all sequences of up to 62 steps, Kani overflow checks active.
    // ------------------------------------------------------------------------------------

    /// Step for the literal emitted code: after `n` steps h <= 2^n - 1; the next step cannot
    /// overflow for n <= 61 and re-establishes both the relation and the size bound.
    #[kani::proof]
    fn literal_step() {
        let n: u32 = kani::any();
        kani::assume(n <= 61);
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= (1i64 << n) - 1);
        kani::assume(inv(h, s));
        let (x0, x1) = any_symbol();

        let h2 = nd::step([h], x0, x1)[0]; // overflow-checked by Kani
        assert!(0 <= h2 && h2 <= (1i64 << (n + 1)) - 1);
        assert!(inv(h2, dfa_step(s, x0, x1)));
    }

    /// Base + output for the literal code (n = 0 gives h = 0; output needs |h| small enough
    /// that 2*h does not overflow, which h <= 2^62 - 1 guarantees).
    #[kani::proof]
    fn literal_base_and_output() {
        assert!(inv(0, DFA_INIT) && (1i64 << 0) - 1 == 0);
        let n: u32 = kani::any();
        kani::assume(n <= 62);
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= (1i64 << n) - 1);
        kani::assume(inv(h, s));
        assert_eq!(nd::output([h]), dfa_accept(s));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::SYMBOLS;

    fn all_sequences(max_len: usize) -> Vec<Vec<(i64, i64)>> {
        let mut out = vec![vec![]];
        let mut frontier = vec![vec![]];
        for _ in 0..max_len {
            let mut next = vec![];
            for seq in &frontier {
                for &sym in &SYMBOLS {
                    let mut s: Vec<(i64, i64)> = seq.clone();
                    s.push(sym);
                    next.push(s);
                }
            }
            out.extend(next.iter().cloned());
            frontier = next;
        }
        out
    }

    /// English-spec oracle written independently of everything above.
    fn oracle(seq: &[(i64, i64)]) -> usize {
        let bits: Vec<bool> = seq.iter().filter(|s| s.0 + s.1 > 0).map(|s| s.1 == 1).collect();
        bits.windows(2).any(|w| w[0] && w[1]) as usize
    }

    #[test]
    fn circuit_dfa_and_oracle_agree_and_invariant_holds() {
        for seq in all_sequences(9) {
            let arr: Vec<[i64; 2]> = seq.iter().map(|s| [s.0, s.1]).collect();
            let mut h = [0i64];
            let mut s = DFA_INIT;
            assert!(inv(h[0], s));
            for &(x0, x1) in &seq {
                h = nd::step(h, x0, x1);
                s = dfa_step(s, x0, x1);
                assert!(inv(h[0], s), "invariant broken on {:?}", seq);
            }
            let want = oracle(&seq);
            assert_eq!(nd::decompiled(&arr, arr.len()), want, "{:?}", seq);
            assert_eq!(nd::output(h), want);
            assert_eq!(dfa_accept(s), want);
            assert_eq!(spec_contains_11(&arr, arr.len()), want);
        }
    }

    /// The literal i64 circuit overflows on the 63rd consecutive 1-bit (debug build panics);
    /// 62 ones are fine. This is the boundary the `literal_*` proofs stop at.
    #[test]
    fn literal_circuit_survives_62_ones() {
        let mut h = [0i64];
        for _ in 0..62 {
            h = nd::step(h, 0, 1);
        }
        assert_eq!(nd::output(h), 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "overflow")]
    fn literal_circuit_overflows_on_63rd_one() {
        let mut h = [0i64];
        for _ in 0..63 {
            h = nd::step(h, 0, 1);
        }
    }

    #[test]
    fn valid_symbols_are_exactly_the_three() {
        for x0 in -1..=2 {
            for x1 in -1..=2 {
                assert_eq!(valid_symbol(x0, x1), SYMBOLS.contains(&(x0, x1)));
            }
        }
    }
}
