//! Shared helpers for the proofs.

/// The three legal input symbols of the regex-style circuits, as (x0, x1) pairs:
/// `(1,0)` = bit 0, `(0,1)` = bit 1, `(0,0)` = padding.
pub const SYMBOLS: [(i64, i64); 3] = [(1, 0), (0, 1), (0, 0)];

pub fn valid_symbol(x0: i64, x1: i64) -> bool {
    (x0 == 0 && x1 == 0) || (x0 == 1 && x1 == 0) || (x0 == 0 && x1 == 1)
}

/// A symbolic input symbol: any of bit 0, bit 1, padding.
#[cfg(kani)]
pub fn any_symbol() -> (i64, i64) {
    let k: u8 = kani::any();
    kani::assume(k < 3);
    SYMBOLS[k as usize]
}

/// Saturation bound used to model the circuits over the *integers* without machine overflow.
/// Every i64 expression in the circuits is evaluated only on states in `[0, CAP]`, where
/// `2*CAP + 2 < 2^63`, so Kani's overflow checks stay meaningful.
pub const CAP: i64 = 1 << 60;
