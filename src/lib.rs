//! Library surface for `nd`. Exists so integration tests (tests/*.rs) can
//! exercise the real decompilation/quantization/verification/emission code
//! paths directly, instead of only through the CLI binary.
pub mod weights;
pub mod quantize;
pub mod emit;
pub mod verify;
pub mod fsm;
pub mod gguf;
pub mod trace;
pub mod diagnose;
pub mod compare;
pub mod visualize;
pub mod taxonomy;
pub mod diff;
pub mod evolve;
pub mod patch;
pub mod slice;
pub mod transformer;
pub mod xray;
pub mod intmap;
