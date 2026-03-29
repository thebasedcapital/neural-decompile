/// Decompiled RNN for "contains_11" — detects consecutive 1s in a binary sequence.
///
/// Circuit (h0 dead, only h1 active):
///   h1 = (2*h1 - x0 + 2*x1 - 1).max(0)
///   logits = [-2*h1 + 3, 2*h1 - 3]
///   output = argmax(logits) = if h1 >= 2 { 1 } else { 0 }
///
/// Input encoding: [1,0]=bit0, [0,1]=bit1, [0,0]=padding (no-op).
/// Sequences are padded to fixed length 5 with [0,0].

/// The decompiled circuit, integer-lifted.
/// Takes raw (x0, x1) pairs — each is (0,0), (1,0), or (0,1).
fn contains_11_decompiled(seq: &[[i64; 2]], len: usize) -> usize {
    let mut h1: i64 = 0;

    let mut i = 0;
    while i < len {
        let x0 = seq[i][0];
        let x1 = seq[i][1];
        h1 = (2 * h1 - x0 + 2 * x1 - 1).max(0);
        i += 1;
    }

    // argmax: h1 >= 2 => class 1 (contains "11"), else class 0
    if h1 >= 2 { 1 } else { 0 }
}

/// Reference: does the active (non-padding) portion contain consecutive 1-bits?
fn spec_contains_11(seq: &[[i64; 2]], len: usize) -> usize {
    let mut prev_is_one = false;
    let mut i = 0;
    while i < len {
        let is_one = seq[i][0] == 0 && seq[i][1] == 1;
        if is_one && prev_is_one {
            return 1;
        }
        // Only update prev if this is an actual bit (not padding)
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

    const MAX_LEN: usize = 5;

    /// Prove: for ALL valid input sequences (length 5, with one-hot bits + padding),
    /// the decompiled circuit matches the spec.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_contains_11() {
        // Number of real bits (1..=5), rest are padding [0,0]
        let n_bits: usize = kani::any();
        kani::assume(n_bits >= 1 && n_bits <= MAX_LEN);

        let mut seq: [[i64; 2]; MAX_LEN] = [[0, 0]; MAX_LEN];

        let mut i = 0;
        while i < MAX_LEN {
            if i < n_bits {
                let bit: u8 = kani::any();
                kani::assume(bit <= 1);
                if bit == 0 {
                    seq[i] = [1, 0];
                } else {
                    seq[i] = [0, 1];
                }
            }
            // else: seq[i] stays [0, 0] (padding)
            i += 1;
        }

        let result = contains_11_decompiled(&seq, MAX_LEN);
        let expected = spec_contains_11(&seq, MAX_LEN);
        assert_eq!(result, expected);
    }

    /// Prove monotonicity: once h1 >= 2 (accepted), it stays >= 2.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_monotonicity() {
        let n_bits: usize = kani::any();
        kani::assume(n_bits >= 1 && n_bits <= MAX_LEN);

        let mut seq: [[i64; 2]; MAX_LEN] = [[0, 0]; MAX_LEN];
        let mut i = 0;
        while i < MAX_LEN {
            if i < n_bits {
                let bit: u8 = kani::any();
                kani::assume(bit <= 1);
                if bit == 0 { seq[i] = [1, 0]; } else { seq[i] = [0, 1]; }
            }
            i += 1;
        }

        let mut h1: i64 = 0;
        let mut ever_accepted = false;
        let mut j = 0;
        while j < MAX_LEN {
            let x0 = seq[j][0];
            let x1 = seq[j][1];
            h1 = (2 * h1 - x0 + 2 * x1 - 1).max(0);
            if h1 >= 2 { ever_accepted = true; }
            if ever_accepted {
                assert!(h1 >= 2, "Monotonicity violated");
            }
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(bits: &[u8]) -> Vec<[i64; 2]> {
        let mut seq: Vec<[i64; 2]> = bits.iter().map(|&b| {
            if b == 0 { [1, 0] } else { [0, 1] }
        }).collect();
        // Pad to 5
        while seq.len() < 5 { seq.push([0, 0]); }
        seq
    }

    #[test]
    fn test_basic_cases() {
        let cases: Vec<(Vec<u8>, usize)> = vec![
            (vec![1, 1], 1),
            (vec![1, 0], 0),
            (vec![0, 1, 1], 1),
            (vec![0], 0),
            (vec![1, 0, 1, 0, 1], 0),
            (vec![1, 0, 0, 1, 1], 1),
        ];
        for (bits, expected) in cases {
            let seq = make_seq(&bits);
            assert_eq!(
                contains_11_decompiled(&seq, 5), expected,
                "Decompiled failed for bits={:?}", bits
            );
            assert_eq!(
                spec_contains_11(&seq, 5), expected,
                "Spec failed for bits={:?}", bits
            );
        }
    }
}
