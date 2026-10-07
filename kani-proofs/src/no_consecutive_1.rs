//! no_consecutive_1 — returns 1 iff the bit stream has NO two consecutive 1-bits.
//!
//! Circuit: `crate::generated::no_consecutive_1` (what `nd decompile
//! examples/regex/no_consecutive_1.json --format rust-kani` prints). Same hidden dynamics as
//! `contains_11`, output logits swapped: class 1 iff `h < 2`.
//!
//! Proof structure and overflow scope are exactly those of `contains_11` (read its module doc):
//! the same relation `contains_11::inv` between `h` and the spec DFA `(seen, prev1)` is shown to be
//! inductive for this circuit, now with accept = `!seen`.

use crate::common::CAP;
use crate::contains_11::{dfa_step, inv, step_sat, Dfa, DFA_INIT};
use crate::generated::no_consecutive_1 as nd;

pub fn dfa_accept(s: Dfa) -> usize {
    (!s.seen) as usize
}

/// Independent reference: 1 iff no two consecutive real bits are both 1 (padding skipped).
pub fn spec_no_consecutive_1(seq: &[[i64; 2]], len: usize) -> usize {
    let mut prev_is_one = false;
    let mut i = 0;
    while i < len {
        let is_one = seq[i][0] == 0 && seq[i][1] == 1;
        if is_one && prev_is_one {
            return 0;
        }
        let is_bit = seq[i][0] + seq[i][1] > 0;
        if is_bit {
            prev_is_one = is_one;
        }
        i += 1;
    }
    1
}

#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::common::any_symbol;

    const MAX_LEN: usize = 5;

    /// BOUNDED (exhaustive): every sequence of length <= 5 over {bit0, bit1, pad}.
    #[kani::proof]
    #[kani::unwind(7)]
    fn verify_no_consecutive_1() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);
        let mut seq = [[0i64; 2]; MAX_LEN];
        let mut i = 0;
        while i < MAX_LEN {
            let (x0, x1) = any_symbol();
            seq[i] = [x0, x1];
            i += 1;
        }
        assert_eq!(nd::decompiled(&seq, len), spec_no_consecutive_1(&seq, len));
    }

    /// UNBOUNDED step: the *same* `inv` is preserved by this circuit's emitted step
    /// (step expression is textually that of contains_11; proven here on this circuit's own code).
    #[kani::proof]
    fn unbounded_step() {
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= CAP);
        kani::assume(inv(h, s));
        let (x0, x1) = any_symbol();

        let h2 = nd::step([h], x0, x1)[0].min(CAP);
        assert!(0 <= h2 && h2 <= CAP);
        assert!(inv(h2, dfa_step(s, x0, x1)));
        // and it is the very same transition function as contains_11's
        assert_eq!(h2, step_sat(h, x0, x1));
    }

    /// UNBOUNDED base.
    #[kani::proof]
    fn unbounded_base() {
        assert!(inv(0, DFA_INIT));
    }

    /// UNBOUNDED output: related states give "no 11 seen yet" as the verdict.
    #[kani::proof]
    fn unbounded_output() {
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= CAP);
        kani::assume(inv(h, s));
        assert_eq!(nd::output([h]), dfa_accept(s));
    }

    /// LITERAL i64 circuit, overflow-checked, all sequences of up to 62 steps (see contains_11).
    #[kani::proof]
    fn literal_step() {
        let n: u32 = kani::any();
        kani::assume(n <= 61);
        let h: i64 = kani::any();
        let s: Dfa = kani::any();
        kani::assume(0 <= h && h <= (1i64 << n) - 1);
        kani::assume(inv(h, s));
        let (x0, x1) = any_symbol();

        let h2 = nd::step([h], x0, x1)[0];
        assert!(0 <= h2 && h2 <= (1i64 << (n + 1)) - 1);
        assert!(inv(h2, dfa_step(s, x0, x1)));
    }

    #[kani::proof]
    fn literal_output() {
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

    #[test]
    fn exhaustive_up_to_len_9() {
        let mut frontier: Vec<Vec<(i64, i64)>> = vec![vec![]];
        for _ in 0..=9 {
            let mut next = vec![];
            for seq in &frontier {
                let arr: Vec<[i64; 2]> = seq.iter().map(|s| [s.0, s.1]).collect();
                let bits: Vec<bool> =
                    seq.iter().filter(|s| s.0 + s.1 > 0).map(|s| s.1 == 1).collect();
                let want = (!bits.windows(2).any(|w| w[0] && w[1])) as usize;
                assert_eq!(nd::decompiled(&arr, arr.len()), want, "{:?}", seq);
                assert_eq!(spec_no_consecutive_1(&arr, arr.len()), want, "{:?}", seq);
                for &sym in &SYMBOLS {
                    let mut s = seq.clone();
                    s.push(sym);
                    next.push(s);
                }
            }
            frontier = next;
        }
    }
}
