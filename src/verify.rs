use crate::quantize::{QuantizedRnn, QuantizedTransformer};
use crate::transformer::Transformer;
use crate::fsm;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub struct TestCase {
    pub inputs: Vec<Vec<f64>>,
    pub expected: usize,
}

/// Transformer test case: token sequence -> expected output class
#[derive(Deserialize)]
pub struct TransformerTest {
    pub tokens: Vec<usize>,
    pub expected: usize,  // argmax of logits at last position
}

/// Transformer verification with full logits comparison
#[derive(Deserialize)]
pub struct TransformerLogitsTest {
    pub tokens: Vec<usize>,
    pub expected_logits: Vec<f64>,  // expected logits at last position
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

pub struct TransformerVerifyResults {
    pub total: usize,
    pub passed: usize,
    pub failures: Vec<TransformerFailure>,
}

pub struct TransformerFailure {
    pub tokens: Vec<usize>,
    pub expected: usize,
    pub got: usize,
    pub logits: Vec<f64>,
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

/// Verify transformer by running forward pass and comparing argmax
pub fn verify_transformer(t: &Transformer, tests: &[TransformerTest]) -> TransformerVerifyResults {
    let mut passed = 0;
    let mut failures = Vec::new();

    for tc in tests {
        let logits = t.forward(&tc.tokens);
        let last_logits = logits.last().unwrap();
        let got = argmax(last_logits);

        if got == tc.expected {
            passed += 1;
        } else {
            failures.push(TransformerFailure {
                tokens: tc.tokens.clone(),
                expected: tc.expected,
                got,
                logits: last_logits.clone(),
            });
        }
    }

    TransformerVerifyResults {
        total: tests.len(),
        passed,
        failures,
    }
}

/// Verify transformer logits directly (for numerical verification)
pub fn verify_transformer_logits(t: &Transformer, tests: &[TransformerLogitsTest], tolerance: f64) -> (usize, usize) {
    let mut passed = 0;
    let total = tests.len();

    for tc in tests {
        let logits = t.forward(&tc.tokens);
        let last_logits = logits.last().unwrap();

        // Compare each logit within tolerance
        let matches = last_logits.len() == tc.expected_logits.len()
            && last_logits.iter().zip(tc.expected_logits.iter())
                .all(|(a, b)| (a - b).abs() < tolerance);

        if matches {
            passed += 1;
        }
    }

    (passed, total)
}

/// Verify that decompiled transformer produces same logits as original
/// Uses tolerance-based comparison for numerical accuracy
pub fn verify_decompiled_transformer(
    original: &Transformer,
    quantized: &QuantizedTransformer,
    tests: &[TransformerTest],
) -> TransformerVerifyResults {
    let mut passed = 0;
    let mut failures = Vec::new();
    let tolerance = 0.01;  // 1% tolerance for numerical differences

    for tc in tests {
        // Run original
        let orig_logits = original.forward(&tc.tokens);
        let orig_last = orig_logits.last().unwrap();
        let orig_argmax = argmax(orig_last);

        // Run quantized forward pass
        let quant_logits = forward_quantized(quantized, &tc.tokens);
        let quant_last = quant_logits.last().unwrap();
        let quant_argmax = argmax(quant_last);

        // Compare logits numerically within tolerance
        let logits_match = orig_last.iter().zip(quant_last.iter())
            .all(|(a, b)| (a - b).abs() < tolerance);

        // Both must match: numerical logits AND classification output
        if logits_match && orig_argmax == tc.expected {
            passed += 1;
        } else {
            failures.push(TransformerFailure {
                tokens: tc.tokens.clone(),
                expected: tc.expected,
                got: quant_argmax,
                logits: quant_last.clone(),
            });
        }
    }

    TransformerVerifyResults {
        total: tests.len(),
        passed,
        failures,
    }
}

/// Forward pass on quantized transformer
fn forward_quantized(t: &QuantizedTransformer, tokens: &[usize]) -> Vec<Vec<f64>> {
    let seq_len = tokens.len();
    assert!(seq_len <= t.max_seq_len);

    // Token + position embeddings
    let mut hidden: Vec<Vec<f64>> = tokens.iter().enumerate().map(|(i, &tok)| {
        let mut emb = t.token_emb[tok].clone();
        if let Some(ref pos) = t.pos_emb {
            for j in 0..t.d_model {
                emb[j] += pos[i][j];
            }
        }
        emb
    }).collect();

    // Pass through layers
    for layer in &t.layers {
        hidden = forward_layer_quantized(layer, &hidden, t.d_model);
    }

    // Final layer norm
    if let (Some(g), Some(b)) = (&t.ln_final_gamma, &t.ln_final_beta) {
        for h in &mut hidden {
            *h = layer_norm(h, g, b);
        }
    }

    // Output projection
    hidden.iter().map(|h| {
        (0..t.vocab_size).map(|j| {
            let mut s = t.b_out.as_ref().map(|b| b[j]).unwrap_or(0.0);
            for i in 0..t.d_model {
                s += h[i] * t.w_out[i * t.vocab_size + j];
            }
            s
        }).collect()
    }).collect()
}

fn forward_layer_quantized(layer: &crate::quantize::QuantizedLayer, hidden: &[Vec<f64>], d_model: usize) -> Vec<Vec<f64>> {
    let seq_len = hidden.len();
    let head_dim = d_model / layer.n_heads;

    // Pre-norm attention
    let x: Vec<Vec<f64>> = hidden.iter().map(|h| layer_norm(h, &layer.ln1_gamma, &layer.ln1_beta)).collect();

    // Multi-head attention
    let q = matmul(&x, &layer.w_q, d_model, d_model, layer.b_q.as_ref());
    let k = matmul(&x, &layer.w_k, d_model, d_model, layer.b_k.as_ref());
    let v = matmul(&x, &layer.w_v, d_model, d_model, layer.b_v.as_ref());

    let mut attn_out = vec![vec![0.0; d_model]; seq_len];

    for h in 0..layer.n_heads {
        let h_offset = h * head_dim;

        // Compute attention scores
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                for d in 0..head_dim {
                    scores[i][j] += q[i][h_offset + d] * k[j][h_offset + d];
                }
                scores[i][j] /= (head_dim as f64).sqrt();
            }
        }

        // Softmax
        let attn = softmax_2d(&scores);

        // Apply attention to values
        for i in 0..seq_len {
            for d in 0..head_dim {
                for j in 0..seq_len {
                    attn_out[i][h_offset + d] += attn[i][j] * v[j][h_offset + d];
                }
            }
        }
    }

    // Output projection
    let attn_out = matmul(&attn_out, &layer.w_o, d_model, d_model, layer.b_o.as_ref());

    // Residual
    let mut hidden: Vec<Vec<f64>> = hidden.to_vec();
    for i in 0..seq_len {
        for j in 0..d_model {
            hidden[i][j] += attn_out[i][j];
        }
    }

    // Pre-norm FFN
    let x: Vec<Vec<f64>> = hidden.iter().map(|h| layer_norm(h, &layer.ln2_gamma, &layer.ln2_beta)).collect();

    // FFN
    let ffn_hidden = matmul(&x, &layer.w_ff_in, d_model, layer.d_ff, layer.b_ff_in.as_ref());
    let ffn_hidden: Vec<Vec<f64>> = ffn_hidden.iter().map(|h| {
        h.iter().map(|&v| if layer.gelu { gelu(v) } else { v.max(0.0) }).collect()
    }).collect();
    let ffn_out = matmul(&ffn_hidden, &layer.w_ff_out, layer.d_ff, d_model, layer.b_ff_out.as_ref());

    // Residual
    for i in 0..seq_len {
        for j in 0..d_model {
            hidden[i][j] += ffn_out[i][j];
        }
    }

    hidden
}

fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();
    x.iter().enumerate().map(|(i, &v)| (v - mean) / std * gamma[i] + beta[i]).collect()
}

fn softmax_2d(scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
    scores.iter().map(|row| {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.iter().map(|&v| v / sum).collect()
    }).collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
}

fn matmul(x: &[Vec<f64>], w: &[f64], in_dim: usize, out_dim: usize, bias: Option<&Vec<f64>>) -> Vec<Vec<f64>> {
    x.iter().map(|row| {
        (0..out_dim).map(|j| {
            let mut sum = bias.map(|b| b[j]).unwrap_or(0.0);
            for i in 0..in_dim {
                sum += row[i] * w[i * out_dim + j];
            }
            sum
        }).collect()
    }).collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
