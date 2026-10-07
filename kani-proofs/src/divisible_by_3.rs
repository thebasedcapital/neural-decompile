//! divisible_by_3 — class 1 iff the MSB-first binary number on the input is divisible by 3.
//!
//! Circuit: `crate::generated::divisible_by_3` (= `nd decompile examples/regex/divisible_by_3.json
//! --format rust-kani`, 4 live neurons). The circuit reads only `x1`; so padding `(0,0)` and bit 0
//! `(1,0)` are the same input to it. Padding is therefore only meaningful as a *suffix*
//! (appending 0-bits never changes divisibility by 3), which is how the examples and the
//! original proof use it.
//!
//! # RESULT: this circuit is NOT correct for all lengths — Kani proves a counterexample
//!
//! `verify_divisible_by_3` shows exhaustive agreement with the spec for every input of up to 7
//! bits (+ trailing padding). But the bit string `10100001` (= 161, 161 mod 3 = 2, so class 0
//! expected) is classified 1 by the circuit (`counterexample_len8`, a Kani proof of the mismatch).
//! Hence **no unbounded inductive proof of "circuit == divisibility by 3" can exist**; the weights
//! were trained on length <= 7 and only extrapolate up to there. What *is* unbounded and proved:
//!
//! * `reach9_*`: the hidden state is confined to an explicit 9-element set for every input of any
//!   length, and the output is 1 iff the hidden state is the zero vector (so the circuit is a
//!   9-state machine; closed under the emitted step, literal i64 code, no overflow possible).
//!
//! Python enumeration (not Kani) of the product of this 9-state machine with the true mod-3 DFA
//! gives 27 reachable pairs and the first disagreement at length 8 — see `RESULTS.md`.

use crate::generated::divisible_by_3 as nd;

pub const SEQ_LEN: usize = 7;

/// The 9 hidden states reachable from h = 0 under bit-0/bit-1/padding, for any length.
pub const REACH9: [[i64; 4]; 9] = [
    [0, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 2],
    [0, 2, 2, 0],
    [0, 2, 4, 0],
    [0, 4, 2, 2],
    [0, 4, 2, 4],
    [2, 0, 0, 0],
    [4, 0, 0, 0],
];

pub fn in_reach9(h: [i64; 4]) -> bool {
    h == REACH9[0]
        || h == REACH9[1]
        || h == REACH9[2]
        || h == REACH9[3]
        || h == REACH9[4]
        || h == REACH9[5]
        || h == REACH9[6]
        || h == REACH9[7]
        || h == REACH9[8]
}

/// Reference: extract real bits (padding skipped), read MSB-first, test `% 3 == 0`.
pub fn spec_divisible_by_3(seq: &[[i64; 2]], len: usize) -> usize {
    let mut val: u64 = 0;
    let mut i = 0;
    while i < len {
        let is_bit = seq[i][0] + seq[i][1] > 0;
        if is_bit {
            val = val * 2 + seq[i][1] as u64;
        }
        i += 1;
    }
    if val % 3 == 0 { 1 } else { 0 }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// BOUNDED (exhaustive): 0..=7 real bits followed by trailing padding up to length 7.
    #[kani::proof]
    #[kani::unwind(9)]
    fn verify_divisible_by_3() {
        let n_bits: usize = kani::any();
        kani::assume(n_bits <= SEQ_LEN);
        let mut seq = [[0i64; 2]; SEQ_LEN];
        let mut i = 0;
        while i < SEQ_LEN {
            if i < n_bits {
                let bit: u8 = kani::any();
                kani::assume(bit <= 1);
                seq[i] = if bit == 0 { [1, 0] } else { [0, 1] };
            }
            i += 1;
        }
        assert_eq!(nd::decompiled(&seq, SEQ_LEN), spec_divisible_by_3(&seq, SEQ_LEN));
    }

    /// FALSIFICATION (passes iff the circuit really is wrong here): 10100001 = 161 ≡ 2 (mod 3).
    /// The spec says class 0; the emitted circuit says class 1.
    #[kani::proof]
    #[kani::unwind(10)]
    fn counterexample_len8() {
        let one = [0i64, 1];
        let zero = [1i64, 0];
        let seq = [one, zero, one, zero, zero, zero, zero, one];
        assert_eq!(spec_divisible_by_3(&seq, 8), 0);
        assert_eq!(nd::decompiled(&seq, 8), 1);
    }

    /// UNBOUNDED base.
    #[kani::proof]
    fn reach9_base() {
        assert!(in_reach9([0; 4]));
    }

    /// UNBOUNDED step: from ANY state in REACH9 and ANY symbol (bit0, bit1, padding) the emitted
    /// step stays in REACH9. Hence the hidden state is in REACH9 after every prefix of any length.
    #[kani::proof]
    fn reach9_step() {
        let h: [i64; 4] = kani::any();
        kani::assume(in_reach9(h));
        let k: u8 = kani::any();
        kani::assume(k < 3);
        let (x0, x1) = [(1, 0), (0, 1), (0, 0)][k as usize];
        assert!(in_reach9(nd::step(h, x0, x1)));
    }

    /// UNBOUNDED output: on REACH9 the circuit answers 1 exactly when h is the zero vector.
    #[kani::proof]
    fn reach9_output() {
        let h: [i64; 4] = kani::any();
        kani::assume(in_reach9(h));
        assert_eq!(nd::output(h) == 1, h == [0; 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::SYMBOLS;

    fn bits_seq(bits: &[u8]) -> Vec<[i64; 2]> {
        bits.iter().map(|&b| if b == 0 { [1, 0] } else { [0, 1] }).collect()
    }

    #[test]
    fn spec_matches_integer_arithmetic() {
        for n in 0..=12u32 {
            for v in 0..(1u64 << n) {
                let bits: Vec<u8> = (0..n).map(|i| ((v >> (n - 1 - i)) & 1) as u8).collect();
                let seq = bits_seq(&bits);
                assert_eq!(spec_divisible_by_3(&seq, seq.len()), (v % 3 == 0) as usize);
            }
        }
    }

    /// Exact boundary of correctness: all inputs of <= 7 bits right, first failure at 8 bits.
    #[test]
    fn correct_up_to_7_bits_first_failure_at_8() {
        let mut fails_by_len = vec![0usize; 15];
        for n in 0..=14u32 {
            for v in 0..(1u64 << n) {
                let bits: Vec<u8> = (0..n).map(|i| ((v >> (n - 1 - i)) & 1) as u8).collect();
                let seq = bits_seq(&bits);
                let got = nd::decompiled(&seq, seq.len());
                if got != spec_divisible_by_3(&seq, seq.len()) {
                    fails_by_len[n as usize] += 1;
                }
            }
        }
        for n in 0..=7 {
            assert_eq!(fails_by_len[n], 0, "len {n}");
        }
        assert_eq!(fails_by_len[8], 1);
        assert!(fails_by_len[9] > 0);
        println!("mismatches by length 0..=14: {:?}", fails_by_len);
    }

    /// REACH9 is exactly the set of hidden states reachable (BFS), not merely closed.
    #[test]
    fn reach9_is_exactly_the_reachable_set() {
        let mut seen: Vec<[i64; 4]> = vec![[0; 4]];
        let mut i = 0;
        while i < seen.len() {
            for &(x0, x1) in &SYMBOLS {
                let n = nd::step(seen[i], x0, x1);
                if !seen.contains(&n) {
                    seen.push(n);
                }
            }
            i += 1;
        }
        assert_eq!(seen.len(), REACH9.len());
        for h in &seen {
            assert!(in_reach9(*h));
        }
    }
}
