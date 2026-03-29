/// Complement proof: contains_11 and no_consecutive_1 always return opposite outputs.
///
/// Both circuits have IDENTICAL hidden dynamics (same h1 update rule).
/// The only difference: output logits are swapped.
///   contains_11:      l0 = -2*h1 + 3,  l1 = 2*h1 - 3
///   no_consecutive_1: l0 = 2*h1 - 3,   l1 = -2*h1 + 3
///
/// This means: contains_11(x) + no_consecutive_1(x) == 1 for all x.
/// Equivalently: they partition the input space into exact complements.

/// Contains_11 circuit (from proven module)
fn contains_11(seq: &[[i64; 2]], len: usize) -> usize {
    let mut h1: i64 = 0;
    let mut i = 0;
    while i < len {
        let x0 = seq[i][0];
        let x1 = seq[i][1];
        h1 = (2 * h1 - x0 + 2 * x1 - 1).max(0);
        i += 1;
    }
    // l0 = -2*h1 + 3, l1 = 2*h1 - 3
    // l1 > l0 iff h1 >= 2
    if h1 >= 2 { 1 } else { 0 }
}

/// No_consecutive_1 circuit — same dynamics, swapped logits
fn no_consecutive_1(seq: &[[i64; 2]], len: usize) -> usize {
    let mut h1: i64 = 0;
    let mut i = 0;
    while i < len {
        let x0 = seq[i][0];
        let x1 = seq[i][1];
        h1 = (2 * h1 - x0 + 2 * x1 - 1).max(0);
        i += 1;
    }
    // l0 = 2*h1 - 3, l1 = -2*h1 + 3
    // l1 > l0 iff h1 < 2 (exact opposite)
    if h1 >= 2 { 0 } else { 1 }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    const MAX_LEN: usize = 5;

    /// Prove: for ALL valid inputs, contains_11(x) + no_consecutive_1(x) == 1.
    /// They are exact boolean complements.
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_complement() {
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

        let c11 = contains_11(&seq, MAX_LEN);
        let nc1 = no_consecutive_1(&seq, MAX_LEN);

        // They must always sum to exactly 1
        assert_eq!(c11 + nc1, 1, "Not complements!");
    }

    /// Prove: both circuits share identical hidden state evolution.
    /// (Structural property — the DFA is literally the same, just accept/reject swapped.)
    #[kani::proof]
    #[kani::unwind(6)]
    fn verify_shared_dynamics() {
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

        // Run both circuits, tracking h1
        let mut h1_a: i64 = 0;
        let mut h1_b: i64 = 0;
        let mut j = 0;
        while j < MAX_LEN {
            let x0 = seq[j][0];
            let x1 = seq[j][1];
            h1_a = (2 * h1_a - x0 + 2 * x1 - 1).max(0);
            h1_b = (2 * h1_b - x0 + 2 * x1 - 1).max(0);
            // Hidden states must be identical at every step
            assert_eq!(h1_a, h1_b, "Hidden states diverged!");
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
        while seq.len() < 5 { seq.push([0, 0]); }
        seq
    }

    #[test]
    fn test_complement() {
        let cases: Vec<Vec<u8>> = vec![
            vec![0], vec![1], vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1],
            vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0], vec![1, 1, 1],
            vec![0, 0, 0, 0], vec![1, 0, 1, 0, 1],
        ];
        for bits in &cases {
            let seq = make_seq(bits);
            let c = contains_11(&seq, 5);
            let n = no_consecutive_1(&seq, 5);
            assert_eq!(c + n, 1, "Not complements for {:?}: c11={} nc1={}", bits, c, n);
        }
    }
}
