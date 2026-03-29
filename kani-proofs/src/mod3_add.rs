/// Decompiled RNN for "mod3_add" — computes (a + b) mod 3 for a,b in {0,1,2}.
///
/// Circuit (3 hidden dims, input_dim=3, output_dim=3):
///   h0 = (-h1 + h2 + 2*x0 - 2*x1).max(0)
///   h1 = (0.56*h0 - h2 - x0 + 2*x2).max(0)
///   h2 = (h1 - h2 - 2*x0 + 2*x1).max(0)
///   logits = [h0 - 2*h1 + h2, 2*h0 + 2*h1 + h2 - 3, -2*h0 + h1 - 2*h2 + 2]
///
/// Input: one-hot [1,0,0]=0, [0,1,0]=1, [0,0,1]=2. Exactly 2 steps (a, b).
/// Has non-integer weight 0.56, so we use f64 arithmetic.
/// But since there are only 9 possible inputs (3×3), this is highly tractable.

fn mod3_add_decompiled(seq: &[[f64; 3]; 2]) -> usize {
    let mut h: [f64; 3] = [0.0; 3];

    let mut i = 0;
    while i < 2 {
        let x0 = seq[i][0];
        let x1 = seq[i][1];
        let x2 = seq[i][2];

        let h0 = (-1.0 * h[1] + 1.0 * h[2] + 2.0 * x0 + -2.0 * x1).max(0.0);
        let h1 = (0.56 * h[0] + -1.0 * h[2] + -1.0 * x0 + 2.0 * x2).max(0.0);
        let h2 = (1.0 * h[1] + -1.0 * h[2] + -2.0 * x0 + 2.0 * x1).max(0.0);
        h = [h0, h1, h2];
        i += 1;
    }

    let logits: [f64; 3] = [
        1.0 * h[0] + -2.0 * h[1] + 1.0 * h[2],
        2.0 * h[0] + 2.0 * h[1] + 1.0 * h[2] + -3.0,
        -2.0 * h[0] + 1.0 * h[1] + -2.0 * h[2] + 2.0,
    ];

    let mut best = 0usize;
    if logits[1] > logits[best] { best = 1; }
    if logits[2] > logits[best] { best = 2; }
    best
}

fn spec_mod3_add(seq: &[[f64; 3]; 2]) -> usize {
    // Decode one-hot to digit
    let a = if seq[0][0] > 0.5 { 0usize }
            else if seq[0][1] > 0.5 { 1 }
            else { 2 };
    let b = if seq[1][0] > 0.5 { 0usize }
            else if seq[1][1] > 0.5 { 1 }
            else { 2 };
    (a + b) % 3
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Prove: for ALL 9 valid (a,b) pairs, decompiled == (a+b) mod 3.
    #[kani::proof]
    #[kani::unwind(3)]
    fn verify_mod3_add() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a < 3);
        kani::assume(b < 3);

        let mut seq: [[f64; 3]; 2] = [[0.0; 3]; 2];
        seq[0][a] = 1.0;
        seq[1][b] = 1.0;

        let result = mod3_add_decompiled(&seq);
        let expected = spec_mod3_add(&seq);
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_mod3_add() {
        for a in 0..3usize {
            for b in 0..3usize {
                let mut seq: [[f64; 3]; 2] = [[0.0; 3]; 2];
                seq[0][a] = 1.0;
                seq[1][b] = 1.0;
                let result = mod3_add_decompiled(&seq);
                let expected = (a + b) % 3;
                assert_eq!(
                    result, expected,
                    "Failed for {}+{}: got {} expected {}",
                    a, b, result, expected
                );
            }
        }
    }
}
