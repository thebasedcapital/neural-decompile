use crate::quantize::QuantizedRnn;
use crate::fsm;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub struct TestCase {
    pub inputs: Vec<Vec<f64>>,
    pub expected: usize,
}

pub struct VerifyResults {
    pub total: usize,
    pub passed: usize,
    pub failures: Vec<Failure>,
}

pub struct Failure {
    pub input: Vec<Vec<f64>>,
    pub expected: usize,
    pub got: usize,
}

pub fn load_test_cases(path: &Path) -> Result<Vec<TestCase>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read test file {}", path.display()))?;
    serde_json::from_str(&data).context("Failed to parse test cases JSON")
}

pub fn run_verification(q: &QuantizedRnn, tests: &[TestCase]) -> VerifyResults {
    let mut passed = 0;
    let mut failures = Vec::new();

    for tc in tests {
        let got = fsm::run_fsm(q, &tc.inputs);
        if got == tc.expected {
            passed += 1;
        } else {
            failures.push(Failure {
                input: tc.inputs.clone(),
                expected: tc.expected,
                got,
            });
        }
    }

    VerifyResults {
        total: tests.len(),
        passed,
        failures,
    }
}
