use neural_decompile::{emit, fsm, gguf, patch, quantize, transformer::Transformer, verify, weights};
use std::{fs, path::PathBuf, process::Command, sync::atomic::{AtomicUsize, Ordering}};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("nd-test-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn write(&self, name: &str, body: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, body).unwrap();
        path
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn checked(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).unwrap()
}

const SMALL: &str = r#"{"W_hh":[[0]],"W_hx":[[0.0001]],"b_h":[0],"W_y":[[0],[1]],"b_y":[0.00005,0]}"#;

#[test]
fn residual_weights_survive_python_rust_and_patch_roundtrip() {
    let tmp = Scratch::new();
    let model = weights::load_rnn_weights(&tmp.write("weights.json", SMALL)).unwrap();
    let q = quantize::quantize_rnn(&model, 0.0);
    assert_eq!(fsm::run_fsm(&q, &[vec![1.0]]), 1);
    let python = emit::emit_python(&q, "decompiled");
    let script = tmp.write("run.py", &(python.clone() + "\nprint(decompiled([[1.0]]))\n"));
    assert_eq!(checked(Command::new("python3").arg(script)).trim(), "1");
    let rust = emit::emit_rust(&q, "decompiled") + "\nfn main() { println!(\"{}\", decompiled(&[vec![1.0]])); }\n";
    let source = tmp.write("run.rs", &rust);
    let binary = tmp.0.join("run");
    checked(Command::new("rustc").args(["--edition=2021", "--crate-name=nd_emitted"]).arg(source).arg("-o").arg(&binary));
    assert_eq!(checked(&mut Command::new(binary)).trim(), "1");
    let parsed = patch::parse_program(&python).unwrap();
    let restored = weights::load_rnn_weights(&tmp.write("patched.json", &patch::program_to_json(&parsed))).unwrap();
    let restored = quantize::quantize_rnn(&restored, 0.0);
    assert_eq!(fsm::run_fsm(&restored, &[vec![1.0]]), 1);
    assert_eq!(fsm::run_fsm(&restored, &[vec![0.0]]), 0);
}

#[test]
fn emitted_rust_handles_a_dead_neuron_and_tied_logits() {
    let tmp = Scratch::new();
    let model = weights::load_rnn_weights(&tmp.write("weights.json", r#"{"W_hh":[[0]],"W_hx":[[0]],"b_h":[0],"W_y":[[0],[0]],"b_y":[0,0]}"#)).unwrap();
    let q = quantize::quantize_rnn(&model, 0.15);
    let source = tmp.write("tie.rs", &(emit::emit_rust(&q, "decompiled") + "\nfn main() { println!(\"{}\", decompiled(&[vec![1.0]])); }\n"));
    let binary = tmp.0.join("tie");
    checked(Command::new("rustc").args(["--edition=2021", "--crate-name=nd_tie"]).arg(source).arg("-o").arg(&binary));
    assert_eq!(checked(&mut Command::new(binary)).trim(), "0");
}

#[test]
fn quantization_threshold_is_strict_and_half_rounds_away_from_zero() {
    let tmp = Scratch::new();
    let mut model = weights::load_rnn_weights(&tmp.write("weights.json", SMALL)).unwrap();
    model.w_hx[[0, 0]] = 0.25;
    assert_eq!(quantize::quantize_rnn(&model, 0.25).w_hx[[0, 0]], 0.25);
    assert_eq!(quantize::quantize_rnn(&model, 0.25001).w_hx[[0, 0]], 0.0);
    model.w_hx[[0, 0]] = -0.5;
    assert_eq!(quantize::quantize_rnn(&model, 0.51).w_hx[[0, 0]], -1.0);
}

#[test]
fn transformer_near_tie_cannot_pass_by_logit_tolerance_alone() {
    let model = Transformer {
        n_layers: 0, d_model: 1, vocab_size: 2, max_seq_len: 1,
        token_emb: vec![vec![1.0], vec![0.0]], pos_emb: None, layers: vec![],
        ln_final_gamma: None, ln_final_beta: None, w_out: vec![0.0, 0.004], b_out: None,
    };
    let q = quantize::quantize_transformer(&model, 0.01);
    let result = verify::verify_decompiled_transformer(&model, &q, &[verify::TransformerTest { tokens: vec![0], expected: 1 }]);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failures[0].got, 0);
}

#[test]
fn malformed_matrix_dimensions_are_rejected_before_execution() {
    let tmp = Scratch::new();
    let malformed = SMALL.replace("\"b_h\":[0]", "\"b_h\":[0,0]");
    assert!(weights::load_rnn_weights(&tmp.write("bad.json", &malformed)).is_err());
    let ragged = r#"{"W_hh":[[0,0],[0],[0,0,0]],"W_hx":[[0],[0],[0]],"b_h":[0,0,0],"W_y":[[0,0,0]],"b_y":[0]}"#;
    assert!(weights::load_rnn_weights(&tmp.write("ragged.json", ragged)).is_err());
}

#[test]
fn cli_emits_executable_substring_detector() {
    let tmp = Scratch::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = tmp.0.join("detector.py");
    checked(Command::new(env!("CARGO_BIN_EXE_nd")).arg("decompile")
        .arg(root.join("examples/regex/contains_11.json")).arg("--output").arg(&script));
    let mut code = fs::read_to_string(&script).unwrap();
    code.push_str("\nassert decompiled([[0,1],[0,1]]) == 1\nassert decompiled([[0,1],[1,0],[0,1]]) == 0\nprint('matched')\n");
    fs::write(&script, code).unwrap();
    assert_eq!(checked(Command::new("python3").arg(&script)).trim(), "matched");
}

#[test]
fn gguf_fixture_decodes_values_instead_of_only_loading_headers() {
    let tmp = Scratch::new();
    let path = tmp.0.join("test.gguf");
    gguf::create_test_gguf(&path).unwrap();
    let model = gguf::GgufFile::open(&path).unwrap();
    let values = model.extract_f32("weight.0").unwrap();
    assert_eq!(values, (1..=12).map(|i| i as f32).collect::<Vec<_>>());
}
