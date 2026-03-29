/// Decompiled RNN for "parity3" — returns 1 if sum(bits) is odd, 0 if even.
///
/// Circuit (3 hidden dims, exactly 3 steps, no padding):
///   h0 = (h1 + h2 + 3*x0 - x1).max(0)
///   h1 = (-h1 + h2 + 2*x0 - x1 + 1).max(0)
///   h2 = (h0 + h1 - h2 + x0 - x1).max(0)
///   logits = [-2*h0 + 2*h1 + 2*h2 - 1, 2*h0 - 2*h1 - 2*h2 + 1]
///
/// Input: [1,0]=bit0, [0,1]=bit1. Always exactly 3 steps.

fn parity3_decompiled(seq: &[[i64; 2]; 3]) -> usize {
    let mut h: [i64; 3] = [0, 0, 0];

    let mut i = 0;
    while i < 3 {
        let x0 = seq[i][0];
        let x1 = seq[i][1];

        let h0 = (h[1] + h[2] + 3 * x0 - x1).max(0);
        let h1 = (-h[1] + h[2] + 2 * x0 - x1 + 1).max(0);
        let h2 = (h[0] + h[1] - h[2] + x0 - x1).max(0);
        h = [h0, h1, h2];
        i += 1;
    }

    let l0 = -2 * h[0] + 2 * h[1] + 2 * h[2] - 1;
    let l1 = 2 * h[0] - 2 * h[1] - 2 * h[2] + 1;
    if l1 > l0 { 1 } else { 0 }
}

fn spec_parity3(seq: &[[i64; 2]; 3]) -> usize {
    let mut ones = 0u8;
    let mut i = 0;
    while i < 3 {
        if seq[i][1] == 1 { ones += 1; }
        i += 1;
    }
    (ones % 2) as usize
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Prove: for ALL 3-bit binary inputs, decompiled == parity spec.
    #[kani::proof]
    #[kani::unwind(4)]
    fn verify_parity3() {
        let mut seq: [[i64; 2]; 3] = [[0, 0]; 3];
        let mut i = 0;
        while i < 3 {
            let bit: u8 = kani::any();
            kani::assume(bit <= 1);
            if bit == 0 { seq[i] = [1, 0]; } else { seq[i] = [0, 1]; }
            i += 1;
        }

        let result = parity3_decompiled(&seq);
        let expected = spec_parity3(&seq);
        assert_eq!(result, expected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_parity3() {
        for a in 0..=1u8 {
            for b in 0..=1u8 {
                for c in 0..=1u8 {
                    let seq: [[i64; 2]; 3] = [
                        if a == 0 { [1, 0] } else { [0, 1] },
                        if b == 0 { [1, 0] } else { [0, 1] },
                        if c == 0 { [1, 0] } else { [0, 1] },
                    ];
                    let expected = ((a + b + c) % 2) as usize;
                    assert_eq!(
                        parity3_decompiled(&seq), expected,
                        "Decompiled failed for ({},{},{})", a, b, c
                    );
                    assert_eq!(
                        spec_parity3(&seq), expected,
                        "Spec failed for ({},{},{})", a, b, c
                    );
                }
            }
        }
    }
}
