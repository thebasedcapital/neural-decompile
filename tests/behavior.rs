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

#[test]
fn verify_rejects_empty_and_failing_fixtures_for_both_architectures() {
    let tmp = Scratch::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rnn = tmp.write("weights.json", SMALL);
    let transformer = root.join("examples/parity_transformer.json");
    let empty = tmp.write("empty.json", "[]");
    for (model, failing) in [
        (&rnn, r#"[{"inputs":[[1.0]],"expected":99}]"#),
        (&transformer, r#"[{"tokens":[0,1,0],"expected":99}]"#),
    ] {
        let failing = tmp.write("failing.json", failing);
        for fixture in [&empty, &failing] {
            let output = Command::new(env!("CARGO_BIN_EXE_nd")).arg("verify")
                .arg(model).arg(fixture).output().unwrap();
            assert_eq!(output.status.code(), Some(1));
            let stdout = String::from_utf8(output.stdout).unwrap();
            assert!(stdout.contains("FAIL — verification failed"), "{}", stdout);
            assert!(!stdout.contains("PERFECT") && !stdout.contains("NaN"), "{}", stdout);
            if fixture == &empty {
                assert!(String::from_utf8_lossy(&output.stderr).contains("Test set is empty"));
            }
        }
    }
    let model = weights::load_rnn_weights(&rnn).unwrap();
    assert!(!verify::run_verification(&quantize::quantize_rnn(&model, 0.15), &[]).is_success());
    let model = Transformer::from_json(&transformer).unwrap();
    assert!(!verify::verify_decompiled_transformer(&model, &quantize::quantize_transformer(&model, 0.01), &[]).is_success());
}

#[test]
fn cli_uses_the_same_epsilon_for_decompile_verify_and_xray() {
    let tmp = Scratch::new();
    let model = tmp.write("weights.json", SMALL);
    let tests = tmp.write("tests.json", r#"[{"inputs":[[1.0]],"expected":1}]"#);
    for (eps, success) in [("0", true), ("0.15", false)] {
        let output = Command::new(env!("CARGO_BIN_EXE_nd")).arg("verify")
            .arg(&model).arg(&tests).args(["--eps", eps]).output().unwrap();
        assert_eq!(output.status.success(), success);
        let code = checked(Command::new(env!("CARGO_BIN_EXE_nd")).arg("decompile")
            .arg(&model).args(["--eps", eps]));
        let script = tmp.write("epsilon.py", &(code + "\nprint(decompiled([[1.0]]))\n"));
        assert_eq!(checked(Command::new("python3").arg(script)).trim(), if success { "1" } else { "0" });
        let report = checked(Command::new(env!("CARGO_BIN_EXE_nd")).arg("xray")
            .arg(&model).arg(&tests).args(["--eps", eps]));
        assert!(report.contains(if success { "1/1" } else { "0/1" }), "{}", report);
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("examples/parity_transformer.json");
    let tests_path = root.join("examples/parity_transformer_tests.json");
    let model = Transformer::from_json(&path).unwrap();
    let tests: Vec<verify::TransformerTest> = serde_json::from_str(&fs::read_to_string(&tests_path).unwrap()).unwrap();
    for eps in [None, Some("0"), Some("0.15")] {
        let epsilon = eps.map(|v| v.parse().unwrap()).unwrap_or(0.01);
        let quantized = quantize::quantize_transformer(&model, epsilon);
        let expected = verify::verify_decompiled_transformer(&model, &quantized, &tests);
        let mut command = Command::new(env!("CARGO_BIN_EXE_nd"));
        command.arg("verify").arg(&path).arg(&tests_path);
        if let Some(eps) = eps { command.args(["--eps", eps]); }
        let output = command.output().unwrap();
        assert_eq!(output.status.success(), expected.is_success());
        assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("{}/{} passed", expected.passed, expected.total)));

        let mut command = Command::new(env!("CARGO_BIN_EXE_nd"));
        command.arg("decompile").arg(&path);
        if let Some(eps) = eps { command.args(["--eps", eps]); }
        assert_eq!(checked(&mut command), emit::emit_transformer_python(&quantized, "decompiled"));
        let report = neural_decompile::xray::run_transformer_xray(&model, "parity_transformer", Some(&tests), epsilon);
        assert_eq!(report.python_code, emit::emit_transformer_python(&quantized, "parity_transformer"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_nd"));
        command.arg("xray").arg(&path).arg(&tests_path);
        if let Some(eps) = eps { command.args(["--eps", eps]); }
        assert_eq!(checked(&mut command), neural_decompile::xray::format_transformer_xray(&report));
    }
}

#[test]
fn transformer_python_and_compiled_rust_match_internal_logits_with_optional_biases() {
    let tmp = Scratch::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let base = Transformer::from_json(&root.join("examples/parity_transformer.json")).unwrap();
    let tests: Vec<verify::TransformerTest> = serde_json::from_str(
        &fs::read_to_string(root.join("examples/parity_transformer_tests.json")).unwrap()).unwrap();
    let mut biased = base.clone();
    let layer = &mut biased.layers[0];
    layer.b_q = Some(vec![0.125, -0.25]);
    layer.b_k = Some(vec![-0.375, 0.5]);
    layer.b_v = Some(vec![0.625, -0.75]);
    layer.b_o = Some(vec![-0.875, 0.125]);
    layer.b_ff_in = Some(vec![0.25, -0.375, 0.5, -0.625]);
    layer.b_ff_out = Some(vec![0.75, -0.875]);
    layer.n_heads = 2; // head_dim=1 must emit a floating-point score divisor.
    let mut gelu = biased.clone();
    gelu.layers[0].gelu = true;
    for model in [&base, &biased, &gelu] {
        let quantized = quantize::quantize_transformer(model, 0.001);
        let mut python = emit::emit_transformer_python(&quantized, "decompiled") + "\n";
        let mut rust = emit::emit_transformer_rust(&quantized, "decompiled") + "\nfn main() {\n";
        for tc in &tests {
            let expected = verify::forward_quantized(&quantized, &tc.tokens);
            python.push_str(&format!("actual = decompiled({:?})\nexpected = {:?}\n", tc.tokens, expected));
            python.push_str("assert len(actual) == len(expected)\nfor a, e in zip(actual, expected):\n    assert len(a) == len(e)\n    assert all(abs(x-y) < 1e-9 for x, y in zip(a, e)), (a, e)\n");
            rust.push_str(&format!("let actual = decompiled(&{:?});\n", tc.tokens));
            for (i, row) in expected.iter().enumerate() {
                for (j, value) in row.iter().enumerate() {
                    rust.push_str(&format!("assert!((actual[{}][{}] - {:?}).abs() < 1e-9);\n", i, j, value));
                }
            }
        }
        python.push_str("print('matched')\n");
        let source = tmp.write("transformer.py", &python);
        assert_eq!(checked(Command::new("python3").arg(source)).trim(), "matched");
        rust.push_str("println!(\"matched\");\n}\n");
        let source = tmp.write("transformer.rs", &rust);
        let binary = tmp.0.join("transformer");
        checked(Command::new("rustc").args(["--edition=2021", "--crate-name=nd_transformer"])
            .arg(source).arg("-o").arg(&binary));
        assert_eq!(checked(&mut Command::new(binary)).trim(), "matched");
    }
}

#[test]
fn circuit_output_is_only_non_executable_heuristic_analysis() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = checked(Command::new(env!("CARGO_BIN_EXE_nd")).arg("decompile")
        .arg(root.join("examples/parity_transformer.json")).args(["--format", "circuit"]));
    assert!(output.starts_with("# Heuristic circuit analysis:"));
    assert!(output.contains("NOT EXECUTABLE"));
    assert!(output.lines().all(|line| line.is_empty() || line.starts_with('#')));
    assert!(!output.contains("Full circuit execution") && !output.contains("return logits"));
}

#[test]
fn benchmark_reports_expected_failures_and_continues() {
    let tmp = Scratch::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = tmp.0.join("benchmark.md");
    checked(Command::new("bash").arg(root.join("scripts/bench.sh")).current_dir(&root)
        .env("ND", env!("CARGO_BIN_EXE_nd")).env("OUT", &output)
        .env("RNN_EPS", "0.15").env("TRANSFORMER_EPS", "0.01"));
    let report = fs::read_to_string(output).unwrap();
    assert!(report.contains("| Expected failure |"));
    for (name, fixtures) in [("bitwise_xor", "14/16 (87%)"), ("mod5", "23/25 (92%)")] {
        let row = report.lines().find(|line| line.starts_with(&format!("| {} |", name))).unwrap();
        assert!(row.contains(fixtures) && row.contains("yes (exit 1)"), "{}", row);
    }
    let transformer = report.lines().find(|line| line.starts_with("| parity_transformer |")).unwrap();
    assert!(transformer.contains("| Transformer | 0.01 |") && transformer.contains("no (exit 0)"), "{}", transformer);
}
