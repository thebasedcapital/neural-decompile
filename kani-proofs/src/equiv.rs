//! The proofs use `step`/`output`, a mechanical split (done by `check_circuits.sh`) of nd's
//! verbatim `decompiled`. This module proves, with Kani, that splitting changed nothing:
//! folding `step` over the input and applying `output` equals `decompiled` — for every sequence of
//! up to 4 symbols (any mix of bit0/bit1/padding). This is a bounded check by nature
//! (the verbatim function is a loop); its job is only to certify the textual split.

#[cfg(kani)]
mod proofs {
    use crate::common::any_symbol;
    use crate::generated::*;

    const N: usize = 4;

    fn symbols() -> ([[i64; 2]; N], usize) {
        let len: usize = kani::any();
        kani::assume(len <= N);
        let mut seq = [[0i64; 2]; N];
        let mut i = 0;
        while i < N {
            let (x0, x1) = any_symbol();
            seq[i] = [x0, x1];
            i += 1;
        }
        (seq, len)
    }

    macro_rules! split_equiv {
        ($name:ident, $m:ident, $hn:expr) => {
            #[kani::proof]
            #[kani::unwind(7)]
            fn $name() {
                let (seq, len) = symbols();
                let mut h = [0i64; $hn];
                let mut i = 0;
                while i < len {
                    h = $m::step(h, seq[i][0], seq[i][1]);
                    i += 1;
                }
                assert_eq!($m::output(h), $m::decompiled(&seq, len));
            }
        };
    }

    split_equiv!(split_equiv_contains_11, contains_11, 1);
    split_equiv!(split_equiv_no_consecutive_1, no_consecutive_1, 1);
    split_equiv!(split_equiv_divisible_by_3, divisible_by_3, 4);
    split_equiv!(split_equiv_parity3, parity3, 3);

    #[kani::proof]
    #[kani::unwind(4)]
    fn split_equiv_mod3_add() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a < 3 && b < 3);
        let mut seq = [[0.0f64; 3]; 2];
        seq[0][a] = 1.0;
        seq[1][b] = 1.0;
        let mut h = [0.0f64; 3];
        let mut i = 0;
        while i < 2 {
            h = mod3_add::step(h, seq[i][0], seq[i][1], seq[i][2]);
            i += 1;
        }
        assert_eq!(mod3_add::output(h), mod3_add::decompiled(&seq, 2));
    }
}
