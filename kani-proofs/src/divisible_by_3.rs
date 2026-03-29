/// Decompiled RNN for "divisible_by_3" — returns 1 if binary number is divisible by 3.
///
/// Circuit (5 hidden dims, h4 dead):
///   h0 = (-h0 + h1 - 3*h2 + 2*x1).max(0)
///   h1 = (h0 + h2 - 2*x1).max(0)
///   h2 = (h0 + h2 + h3 - 4*x1 - 2).max(0)
///   h3 = (h0 - h1 + h2 - 2*x1).max(0)
///   logits[0] = 5*h0 + 3*h1 + 3*h2 + 3*h3 - 4  (class 0 = not div by 3)
///   logits[1] = -(5*h0 + 3*h1 + 3*h2 + 3*h3) + 4  (class 1 = div by 3)
///
/// Input: [1,0]=bit0, [0,1]=bit1, [0,0]=padding. Fixed length 7.
/// Class 1 = divisible by 3.

const SEQ_LEN: usize = 7;

fn divisible_by_3_decompiled(seq: &[[i64; 2]; SEQ_LEN]) -> usize {
    let mut h: [i64; 4] = [0, 0, 0, 0];

    let mut i = 0;
    while i < SEQ_LEN {
        let x1 = seq[i][1];

        let h0 = (-h[0] + h[1] - 3 * h[2] + 2 * x1).max(0);
        let h1 = (h[0] + h[2] - 2 * x1).max(0);
        let h2 = (h[0] + h[2] + h[3] - 4 * x1 - 2).max(0);
        let h3 = (h[0] - h[1] + h[2] - 2 * x1).max(0);
        h = [h0, h1, h2, h3];
        i += 1;
    }

    // logits[1] > logits[0] iff score < 4
    let score = 5 * h[0] + 3 * h[1] + 3 * h[2] + 3 * h[3];
    if score < 4 { 1 } else { 0 }
}

/// Reference: extract bits from one-hot sequence, interpret as MSB-first binary, check % 3.
fn spec_divisible_by_3(seq: &[[i64; 2]; SEQ_LEN]) -> usize {
    let mut val: u64 = 0;
    let mut i = 0;
    while i < SEQ_LEN {
        let is_bit = seq[i][0] + seq[i][1] > 0;
        if is_bit {
            let bit = seq[i][1] as u64;
            val = val * 2 + bit;
        }
        i += 1;
    }
    if val % 3 == 0 { 1 } else { 0 }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Prove: for ALL valid padded input sequences,
    /// the decompiled circuit matches divisible-by-3 check.
    #[kani::proof]
    #[kani::unwind(8)]
    fn verify_divisible_by_3() {
        let n_bits: usize = kani::any();
        kani::assume(n_bits >= 1 && n_bits <= SEQ_LEN);

        let mut seq: [[i64; 2]; SEQ_LEN] = [[0, 0]; SEQ_LEN];
        let mut i = 0;
        while i < SEQ_LEN {
            if i < n_bits {
                let bit: u8 = kani::any();
                kani::assume(bit <= 1);
                if bit == 0 { seq[i] = [1, 0]; } else { seq[i] = [0, 1]; }
            }
            i += 1;
        }

        let result = divisible_by_3_decompiled(&seq);
        let expected = spec_divisible_by_3(&seq);
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(bits: &[u8]) -> [[i64; 2]; SEQ_LEN] {
        let mut seq = [[0i64, 0]; SEQ_LEN];
        for (i, &b) in bits.iter().enumerate() {
            if b == 0 { seq[i] = [1, 0]; } else { seq[i] = [0, 1]; }
        }
        seq
    }

    #[test]
    fn test_divisible_by_3() {
        // 0 (just a "0" bit) = 0, div by 3 => 1
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[0])), 1);
        // 1 = 1, not div => 0
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1])), 0);
        // "11" = 3, div => 1
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1, 1])), 1);
        // "100" = 4, not div => 0
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1, 0, 0])), 0);
        // "110" = 6, div => 1
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1, 1, 0])), 1);
        // "1001" = 9, div => 1
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1, 0, 0, 1])), 1);
        // "1111111" = 127, not div => 0
        assert_eq!(divisible_by_3_decompiled(&make_seq(&[1,1,1,1,1,1,1])), 0);
    }

    #[test]
    fn test_spec_matches() {
        for val in 0..128u64 {
            let bits: Vec<u8> = if val == 0 {
                vec![0]
            } else {
                let n = 64 - val.leading_zeros() as usize;
                (0..n).map(|i| ((val >> (n - 1 - i)) & 1) as u8).collect()
            };
            if bits.len() <= SEQ_LEN {
                let seq = make_seq(&bits);
                let expected = if val % 3 == 0 { 1 } else { 0 };
                assert_eq!(
                    spec_divisible_by_3(&seq), expected,
                    "Spec failed for val={} bits={:?}", val, bits
                );
            }
        }
    }
}
