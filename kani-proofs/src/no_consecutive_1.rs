/// Decompiled RNN for "no_consecutive_1" — returns 1 if NO consecutive 1s exist.
///
/// Identical hidden dynamics to contains_11, swapped output logits.
/// This is the complement language: L(no_consecutive_1) = Σ* \ L(contains_11).

fn no_consecutive_1_decompiled(seq: &[[i64; 2]], len: usize) -> usize {
    let mut h1: i64 = 0;
    let mut i = 0;
    while i < len {
        let x0 = seq[i][0];
        let x1 = seq[i][1];
        h1 = (2 * h1 - x0 + 2 * x1 - 1).max(0);
        i += 1;
    }
    // Swapped from contains_11: class 1 when h1 < 2
    if h1 >= 2 { 0 } else { 1 }
}

/// Reference: no two consecutive bits are both 1.
fn spec_no_consecutive_1(seq: &[[i64; 2]], len: usize) -> usize {
    let mut prev_is_one = false;
    let mut i = 0;
    while i < len {
        let is_one = seq[i][0] == 0 && seq[i][1] == 1;
        if is_one && prev_is_one {
            return 0; // found consecutive 1s → reject
        }
        let is_bit = seq[i][0] + seq[i][1] > 0;
        if is_bit {
            prev_is_one = is_one;
        }
        i += 1;
    }
    1 // no consecutive 1s → accept
}

#[cfg(kani)]
mod proofs {
    use super::*;

    const MAX_LEN: usize = 5;

    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_no_consecutive_1() {
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

        let result = no_consecutive_1_decompiled(&seq, MAX_LEN);
        let expected = spec_no_consecutive_1(&seq, MAX_LEN);
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(bits: &[u8]) -> Vec<[i64; 2]> {
        let mut seq: Vec<[i64; 2]> = bits.iter().map(|&b| {
            if b == 0 { [1, 0] } else { [0, 1] }
        }).collect();
        while seq.len() < 5 { seq.push([0, 0]); }
        seq
    }

    #[test]
    fn test_no_consecutive_1() {
        // Has "11" → 0
        assert_eq!(no_consecutive_1_decompiled(&make_seq(&[1, 1]), 5), 0);
        // No consecutive 1s → 1
        assert_eq!(no_consecutive_1_decompiled(&make_seq(&[1, 0]), 5), 1);
        assert_eq!(no_consecutive_1_decompiled(&make_seq(&[1, 0, 1, 0, 1]), 5), 1);
        assert_eq!(no_consecutive_1_decompiled(&make_seq(&[0]), 5), 1);
    }
}
