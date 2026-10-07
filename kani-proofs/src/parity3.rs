//! parity3 — class 1 iff the number of 1-bits among exactly 3 input bits is odd.
//!
//! Circuit: `crate::generated::parity3` (= `nd decompile examples/parity3.json --format rust-kani`).
//!
//! # Scope
//!
//! The network is only meaningful at length exactly 3 (no padding symbol exists for it): its input
//! domain is the 2^3 = 8 bit triples, and `verify_parity3` checks **all 8** — a *complete* proof
//! over the circuit's domain, not a bounded approximation of a longer one. It is NOT an
//! induction-friendly parity machine: the circuit does not compute parity at other lengths
//! (`wrong_at_length_2` / `wrong_at_length_4` are Kani-checked witnesses), so no unbounded
//! proof exists or is claimed.

use crate::generated::parity3 as nd;

pub fn spec_parity3(seq: &[[i64; 2]; 3]) -> usize {
    let mut ones = 0u8;
    let mut i = 0;
    while i < 3 {
        if seq[i][1] == 1 {
            ones += 1;
        }
        i += 1;
    }
    (ones % 2) as usize
}

pub fn spec_parity(seq: &[[i64; 2]], len: usize) -> usize {
    let mut ones = 0usize;
    let mut i = 0;
    while i < len {
        if seq[i][1] == 1 {
            ones += 1;
        }
        i += 1;
    }
    ones % 2
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// COMPLETE over the 8-element input domain.
    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_parity3() {
        let mut seq = [[0i64; 2]; 3];
        let mut i = 0;
        while i < 3 {
            let bit: u8 = kani::any();
            kani::assume(bit <= 1);
            seq[i] = if bit == 0 { [1, 0] } else { [0, 1] };
            i += 1;
        }
        assert_eq!(nd::decompiled(&seq, 3), spec_parity3(&seq));
    }

    /// Witness: bits 1,1 (parity 0) are classified 1 by the circuit run for 2 steps.
    #[kani::proof]
    #[kani::unwind(5)]
    fn wrong_at_length_2() {
        let seq = [[0i64, 1], [0, 1]];
        assert_eq!(spec_parity(&seq, 2), 0);
        assert_eq!(nd::decompiled(&seq, 2), 1);
    }

    /// Witness: bits 1,1,1,1 (parity 0) at length 4 — see `wrong_at_length_4_is_a_real_witness` test
    /// for the search that produced it; here Kani re-checks that the circuit disagrees.
    #[kani::proof]
    #[kani::unwind(6)]
    fn wrong_at_length_4() {
        let seq = [[0i64, 1], [0, 1], [0, 1], [0, 1]];
        assert_ne!(nd::decompiled(&seq, 4), spec_parity(&seq, 4));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_parity3() {
        for a in 0..=1u8 {
            for b in 0..=1u8 {
                for c in 0..=1u8 {
                    let f = |x: u8| if x == 0 { [1i64, 0] } else { [0, 1] };
                    let seq = [f(a), f(b), f(c)];
                    let expected = ((a + b + c) % 2) as usize;
                    assert_eq!(nd::decompiled(&seq, 3), expected);
                    assert_eq!(spec_parity3(&seq), expected);
                }
            }
        }
    }

    #[test]
    fn wrong_at_length_4_is_a_real_witness() {
        let seq = [[0i64, 1], [0, 1], [0, 1], [0, 1]];
        assert_ne!(nd::decompiled(&seq, 4), spec_parity(&seq, 4));
    }
}
