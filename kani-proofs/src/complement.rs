//! Complement: `contains_11(x) + no_consecutive_1(x) == 1` for every input.
//!
//! Both circuits come straight from `nd` (`crate::generated::{contains_11, no_consecutive_1}`).
//!
//! # Unbounded argument (induction on the sequence, any length)
//!
//! * `step_functions_identical`: for every hidden value `h` and every input symbol, the two
//!   emitted `step` functions return the same value (the code is textually identical, and this is
//!   checked by Kani over h in [-2^61, 2^61]). Hence by induction on the input the two hidden
//!   states are equal after every prefix — whatever the length.
//! * `outputs_complementary`: for every hidden value `h`, `c.output(h) + n.output(h) == 1`.
//!
//! Combine: after any input both circuits sit at the same `h`, and at the same `h` their outputs
//! are complementary. No invariant on `h` is needed, so no overflow caveat beyond the |h| <= 2^61
//! range Kani explores (the identity of the two step functions is textual, so it also holds where
//! the i64 code would wrap).

use crate::generated::{contains_11 as c, no_consecutive_1 as n};

#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::common::any_symbol;

    const MAX_LEN: usize = 5;
    const BOUND: i64 = 1 << 61;

    /// BOUNDED (exhaustive): length <= 5, any mix of bit0/bit1/pad, verbatim nd functions.
    #[kani::proof]
    #[kani::unwind(7)]
    fn verify_complement() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);
        let mut seq = [[0i64; 2]; MAX_LEN];
        let mut i = 0;
        while i < MAX_LEN {
            let (x0, x1) = any_symbol();
            seq[i] = [x0, x1];
            i += 1;
        }
        assert_eq!(c::decompiled(&seq, len) + n::decompiled(&seq, len), 1, "Not complements!");
    }

    /// UNBOUNDED lemma 1: identical transition functions.
    #[kani::proof]
    fn step_functions_identical() {
        let h: i64 = kani::any();
        kani::assume(-BOUND <= h && h <= BOUND);
        let (x0, x1) = any_symbol();
        assert_eq!(c::step([h], x0, x1), n::step([h], x0, x1));
    }

    /// UNBOUNDED lemma 2: complementary outputs at every hidden value.
    #[kani::proof]
    fn outputs_complementary() {
        let h: i64 = kani::any();
        kani::assume(-BOUND <= h && h <= BOUND);
        assert_eq!(c::output([h]) + n::output([h]), 1);
    }

    /// Same two lemmas for ARBITRARY (not just valid) integer inputs in a safe range: the
    /// complement property does not depend on the input encoding.
    #[kani::proof]
    fn step_functions_identical_any_input() {
        let h: i64 = kani::any();
        let x0: i64 = kani::any();
        let x1: i64 = kani::any();
        kani::assume(-(1i64 << 40) <= h && h <= (1i64 << 40));
        kani::assume(-(1i64 << 40) <= x0 && x0 <= (1i64 << 40));
        kani::assume(-(1i64 << 40) <= x1 && x1 <= (1i64 << 40));
        assert_eq!(c::step([h], x0, x1), n::step([h], x0, x1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::SYMBOLS;

    #[test]
    fn complement_exhaustive_up_to_len_9() {
        let mut frontier: Vec<Vec<(i64, i64)>> = vec![vec![]];
        for _ in 0..=9 {
            let mut next = vec![];
            for seq in &frontier {
                let arr: Vec<[i64; 2]> = seq.iter().map(|s| [s.0, s.1]).collect();
                assert_eq!(c::decompiled(&arr, arr.len()) + n::decompiled(&arr, arr.len()), 1);
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
