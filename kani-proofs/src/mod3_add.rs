//! mod3_add — (a + b) mod 3 for digits a, b in {0,1,2}, fed as two one-hot steps.
//!
//! Circuit: `crate::generated::mod3_add` (= `nd decompile examples/mod3_add.json --format rust-kani`;
//! 97% integer weights, so this one is f64 with the non-integer coefficient 0.56).
//!
//! # Scope
//!
//! The input domain is exactly the 3 × 3 = 9 digit pairs; `verify_mod3_add` checks **all 9**.
//! That is a complete proof over the circuit's domain (fixed length 2), not an induction.

use crate::generated::mod3_add as nd;

pub fn spec_mod3_add(a: usize, b: usize) -> usize {
    (a + b) % 3
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(4)]
    fn verify_mod3_add() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a < 3 && b < 3);
        let mut seq = [[0.0f64; 3]; 2];
        seq[0][a] = 1.0;
        seq[1][b] = 1.0;
        assert_eq!(nd::decompiled(&seq, 2), spec_mod3_add(a, b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_nine() {
        for a in 0..3usize {
            for b in 0..3usize {
                let mut seq = [[0.0f64; 3]; 2];
                seq[0][a] = 1.0;
                seq[1][b] = 1.0;
                assert_eq!(nd::decompiled(&seq, 2), spec_mod3_add(a, b), "{a}+{b}");
            }
        }
    }
}
