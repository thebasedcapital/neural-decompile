use crate::quantize::QuantizedRnn;
use crate::trace::{trace_quantized, trace_raw};
use crate::verify::TestCase;
use crate::weights::RnnWeights;
use crate::fsm;

/// Diagnosis of a single failure case
#[derive(Debug)]
pub struct FailureDiagnosis {
    pub input: Vec<Vec<f64>>,
    pub expected: usize,
    pub raw_prediction: usize,
    pub quantized_prediction: usize,
    /// (timestep, neuron_idx, raw_val, quantized_val, delta)
    pub divergence_point: Option<DivergencePoint>,
    /// Weights that are furthest from integer (the "suspects")
    pub suspect_weights: Vec<SuspectWeight>,
    /// Output logit margin (how close was it to being correct)
    pub logit_margin: f64,
}

#[derive(Debug)]
pub struct DivergencePoint {
    pub timestep: usize,
    pub neuron: usize,
    pub raw_val: f64,
    pub quant_val: f64,
    pub delta: f64,
}

#[derive(Debug)]
pub struct SuspectWeight {
    pub matrix: &'static str,
    pub row: usize,
    pub col: usize,
    pub raw_val: f64,
    pub quantized_val: f64,
    pub distance_to_int: f64,
}

/// Full diagnosis report
#[derive(Debug)]
pub struct DiagnoseReport {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub failures: Vec<FailureDiagnosis>,
    pub global_suspects: Vec<SuspectWeight>,
}

/// Find weights furthest from integers (the ones quantization hurts most)
fn find_suspect_weights(rnn: &RnnWeights, q: &QuantizedRnn, top_n: usize) -> Vec<SuspectWeight> {
    let mut suspects = Vec::new();

    // W_hh
    for i in 0..rnn.hidden_dim {
        for j in 0..rnn.hidden_dim {
            let raw = rnn.w_hh[[i, j]];
            let quant = q.w_hh[[i, j]];
            let dist = (raw - raw.round()).abs();
            if dist > 0.01 {
                suspects.push(SuspectWeight {
                    matrix: "W_hh", row: i, col: j,
                    raw_val: raw, quantized_val: quant, distance_to_int: dist,
                });
            }
        }
    }

    // W_hx
    for i in 0..rnn.hidden_dim {
        for j in 0..rnn.input_dim {
            let raw = rnn.w_hx[[i, j]];
            let quant = q.w_hx[[i, j]];
            let dist = (raw - raw.round()).abs();
            if dist > 0.01 {
                suspects.push(SuspectWeight {
                    matrix: "W_hx", row: i, col: j,
                    raw_val: raw, quantized_val: quant, distance_to_int: dist,
                });
            }
        }
    }

    // b_h
    for i in 0..rnn.hidden_dim {
        let raw = rnn.b_h[i];
        let quant = q.b_h[i];
        let dist = (raw - raw.round()).abs();
        if dist > 0.01 {
            suspects.push(SuspectWeight {
                matrix: "b_h", row: i, col: 0,
                raw_val: raw, quantized_val: quant, distance_to_int: dist,
            });
        }
    }

    // W_y
    for i in 0..rnn.output_dim {
        for j in 0..rnn.hidden_dim {
            let raw = rnn.w_y[[i, j]];
            let quant = q.w_y[[i, j]];
            let dist = (raw - raw.round()).abs();
            if dist > 0.01 {
                suspects.push(SuspectWeight {
                    matrix: "W_y", row: i, col: j,
                    raw_val: raw, quantized_val: quant, distance_to_int: dist,
                });
            }
        }
    }

    // b_y
    for i in 0..rnn.output_dim {
        let raw = rnn.b_y[i];
        let quant = q.b_y[i];
        let dist = (raw - raw.round()).abs();
        if dist > 0.01 {
            suspects.push(SuspectWeight {
                matrix: "b_y", row: i, col: 0,
                raw_val: raw, quantized_val: quant, distance_to_int: dist,
            });
        }
    }

    suspects.sort_by(|a, b| b.distance_to_int.partial_cmp(&a.distance_to_int).unwrap());
    suspects.truncate(top_n);
    suspects
}

/// Diagnose a single failure: trace both raw and quantized, find where they diverge
fn diagnose_failure(
    rnn: &RnnWeights,
    q: &QuantizedRnn,
    test: &TestCase,
) -> FailureDiagnosis {
    let raw_trace = trace_raw(rnn, &test.inputs);
    let quant_trace = trace_quantized(q, &test.inputs);

    // Find first significant divergence
    let mut divergence: Option<DivergencePoint> = None;
    for (t, (rs, qs)) in raw_trace.steps.iter().zip(quant_trace.steps.iter()).enumerate() {
        for i in 0..rnn.hidden_dim {
            let delta = (rs.hidden[i] - qs.hidden[i]).abs();
            if delta > 0.01 {
                if divergence.is_none()
                    || delta > divergence.as_ref().unwrap().delta
                {
                    divergence = Some(DivergencePoint {
                        timestep: t,
                        neuron: i,
                        raw_val: rs.hidden[i],
                        quant_val: qs.hidden[i],
                        delta,
                    });
                }
            }
        }
    }

    // Logit margin: how close was quantized to getting it right?
    let correct_logit = quant_trace.output_logits[test.expected];
    let best_wrong = quant_trace.output_logits.iter().enumerate()
        .filter(|&(i, _)| i != test.expected)
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = correct_logit - best_wrong;

    // Find suspect weights involved in the divergent neuron
    let mut suspects = Vec::new();
    if let Some(ref div) = divergence {
        let n = div.neuron;
        // Check W_hh row n and W_hx row n
        for j in 0..rnn.hidden_dim {
            let raw = rnn.w_hh[[n, j]];
            let dist = (raw - raw.round()).abs();
            if dist > 0.05 {
                suspects.push(SuspectWeight {
                    matrix: "W_hh", row: n, col: j,
                    raw_val: raw, quantized_val: q.w_hh[[n, j]], distance_to_int: dist,
                });
            }
        }
        for j in 0..rnn.input_dim {
            let raw = rnn.w_hx[[n, j]];
            let dist = (raw - raw.round()).abs();
            if dist > 0.05 {
                suspects.push(SuspectWeight {
                    matrix: "W_hx", row: n, col: j,
                    raw_val: raw, quantized_val: q.w_hx[[n, j]], distance_to_int: dist,
                });
            }
        }
        let raw = rnn.b_h[n];
        let dist = (raw - raw.round()).abs();
        if dist > 0.05 {
            suspects.push(SuspectWeight {
                matrix: "b_h", row: n, col: 0,
                raw_val: raw, quantized_val: q.b_h[n], distance_to_int: dist,
            });
        }
        suspects.sort_by(|a, b| b.distance_to_int.partial_cmp(&a.distance_to_int).unwrap());
    }

    FailureDiagnosis {
        input: test.inputs.clone(),
        expected: test.expected,
        raw_prediction: raw_trace.prediction,
        quantized_prediction: quant_trace.prediction,
        divergence_point: divergence,
        suspect_weights: suspects,
        logit_margin: margin,
    }
}

/// Run full diagnosis on all failing test cases
pub fn run_diagnosis(
    rnn: &RnnWeights,
    q: &QuantizedRnn,
    tests: &[TestCase],
) -> DiagnoseReport {
    let mut passed = 0;
    let mut failures = Vec::new();

    for tc in tests {
        let got = fsm::run_fsm(q, &tc.inputs);
        if got == tc.expected {
            passed += 1;
        } else {
            failures.push(diagnose_failure(rnn, q, tc));
        }
    }

    let global_suspects = find_suspect_weights(rnn, q, 10);

    DiagnoseReport {
        total: tests.len(),
        passed,
        failed: failures.len(),
        failures,
        global_suspects,
    }
}

/// Format diagnosis report
pub fn format_diagnosis(report: &DiagnoseReport) -> String {
    let mut out = String::new();

    out.push_str(&format!("═══ DIAGNOSIS REPORT ═══\n"));
    out.push_str(&format!("Pass: {}/{}  Fail: {}\n\n", report.passed, report.total, report.failed));

    // Global suspect weights
    if !report.global_suspects.is_empty() {
        out.push_str("── Top Suspect Weights (furthest from integer) ──\n");
        out.push_str(&format!("{:<8} {:>4} {:>4}  {:>10} {:>10}  {:>8}\n",
            "matrix", "row", "col", "raw", "quantized", "dist"));
        for s in &report.global_suspects {
            out.push_str(&format!("{:<8} {:>4} {:>4}  {:>10.4} {:>10.4}  {:>8.4}\n",
                s.matrix, s.row, s.col, s.raw_val, s.quantized_val, s.distance_to_int));
        }
        out.push('\n');
    }

    // Per-failure analysis
    for (i, f) in report.failures.iter().enumerate() {
        // Compact input display
        let input_str: String = f.input.iter().map(|x| {
            if x.len() == 2 {
                // One-hot: show which bit
                if x[0] > x[1] { "0" } else { "1" }
            } else {
                "?"
            }
        }).collect();

        out.push_str(&format!("── Failure {} ──\n", i + 1));
        out.push_str(&format!("  input: {:?}  (compact: \"{}\")\n",
            if f.input.len() <= 8 { format!("{:?}", f.input) } else { format!("[{}×{}d]", f.input.len(), f.input[0].len()) },
            input_str));
        out.push_str(&format!("  expected: {}  raw: {}  quantized: {}\n",
            f.expected, f.raw_prediction, f.quantized_prediction));
        out.push_str(&format!("  logit margin: {:.4} (negative = wrong answer won)\n", f.logit_margin));

        if let Some(ref div) = f.divergence_point {
            out.push_str(&format!("  divergence: t={} h{} — raw={:.4} quant={:.4} Δ={:.4}\n",
                div.timestep, div.neuron, div.raw_val, div.quant_val, div.delta));
        }

        if !f.suspect_weights.is_empty() {
            out.push_str("  suspects:\n");
            for s in &f.suspect_weights {
                out.push_str(&format!("    {}[{},{}] = {:.4} → {:.4} (dist {:.4})\n",
                    s.matrix, s.row, s.col, s.raw_val, s.quantized_val, s.distance_to_int));
            }
        }
        out.push('\n');
    }

    // Summary insight
    if !report.failures.is_empty() {
        let avg_margin: f64 = report.failures.iter().map(|f| f.logit_margin).sum::<f64>()
            / report.failures.len() as f64;
        let close_calls = report.failures.iter().filter(|f| f.logit_margin.abs() < 1.0).count();

        out.push_str("── Summary ──\n");
        out.push_str(&format!("  avg logit margin on failures: {:.4}\n", avg_margin));
        out.push_str(&format!("  close calls (|margin| < 1.0): {}/{}\n", close_calls, report.failures.len()));

        if close_calls == report.failures.len() {
            out.push_str("  → ALL failures are close calls — better L1 training might save them\n");
        }

        // Check if failures share a common divergent neuron
        let mut neuron_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for f in &report.failures {
            if let Some(ref div) = f.divergence_point {
                *neuron_counts.entry(div.neuron).or_default() += 1;
            }
        }
        if let Some((&neuron, &count)) = neuron_counts.iter().max_by_key(|&(_, c)| c) {
            if count > 1 {
                out.push_str(&format!("  → h{} is the common culprit ({}/{} failures)\n",
                    neuron, count, report.failures.len()));
            }
        }
    }

    out
}
