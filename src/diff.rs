use crate::quantize::QuantizedRnn;

/// Semantic diff between two circuits
#[derive(Debug)]
pub struct CircuitDiff {
    pub hidden_dim_a: usize,
    pub hidden_dim_b: usize,
    /// Per-neuron changes in the transition function
    pub neuron_diffs: Vec<NeuronDiff>,
    /// Changes in output layer
    pub output_diffs: Vec<OutputDiff>,
    /// Overall weight delta stats
    pub total_changed_weights: usize,
    pub total_weights: usize,
    pub max_delta: f64,
    pub mean_delta: f64,
}

#[derive(Debug)]
pub struct NeuronDiff {
    pub neuron: usize,
    /// Sum of absolute weight changes for this neuron's row
    pub total_delta: f64,
    /// Individual weight changes: (matrix, col, old, new, delta)
    pub changes: Vec<WeightChange>,
}

#[derive(Debug)]
pub struct OutputDiff {
    pub output_class: usize,
    pub total_delta: f64,
    pub changes: Vec<WeightChange>,
}

#[derive(Debug)]
pub struct WeightChange {
    pub matrix: &'static str,
    pub col: usize,
    pub old_val: f64,
    pub new_val: f64,
    pub delta: f64,
}

/// Compute semantic diff between two quantized RNNs.
/// They must have the same dimensions (use slice first if needed).
pub fn diff_circuits(a: &QuantizedRnn, b: &QuantizedRnn) -> Result<CircuitDiff, String> {
    if a.hidden_dim != b.hidden_dim {
        return Err(format!(
            "Hidden dim mismatch: {} vs {}. Slice both circuits first.",
            a.hidden_dim, b.hidden_dim
        ));
    }
    if a.input_dim != b.input_dim || a.output_dim != b.output_dim {
        return Err(format!(
            "IO dim mismatch: in {}→{}, out {}→{}",
            a.input_dim, b.input_dim, a.output_dim, b.output_dim
        ));
    }

    let hd = a.hidden_dim;
    let threshold = 0.01;

    let mut neuron_diffs = Vec::new();
    let mut total_changed = 0;
    let mut total_weights = 0;
    let mut all_deltas = Vec::new();

    for i in 0..hd {
        let mut changes = Vec::new();

        // W_hh row i
        for j in 0..hd {
            total_weights += 1;
            let delta = b.w_hh[[i, j]] - a.w_hh[[i, j]];
            if delta.abs() > threshold {
                total_changed += 1;
                all_deltas.push(delta.abs());
                changes.push(WeightChange {
                    matrix: "W_hh",
                    col: j,
                    old_val: a.w_hh[[i, j]],
                    new_val: b.w_hh[[i, j]],
                    delta,
                });
            }
        }

        // W_hx row i
        for j in 0..a.input_dim {
            total_weights += 1;
            let delta = b.w_hx[[i, j]] - a.w_hx[[i, j]];
            if delta.abs() > threshold {
                total_changed += 1;
                all_deltas.push(delta.abs());
                changes.push(WeightChange {
                    matrix: "W_hx",
                    col: j,
                    old_val: a.w_hx[[i, j]],
                    new_val: b.w_hx[[i, j]],
                    delta,
                });
            }
        }

        // b_h[i]
        total_weights += 1;
        let delta = b.b_h[i] - a.b_h[i];
        if delta.abs() > threshold {
            total_changed += 1;
            all_deltas.push(delta.abs());
            changes.push(WeightChange {
                matrix: "b_h",
                col: 0,
                old_val: a.b_h[i],
                new_val: b.b_h[i],
                delta,
            });
        }

        let total_delta: f64 = changes.iter().map(|c| c.delta.abs()).sum();
        if !changes.is_empty() {
            neuron_diffs.push(NeuronDiff {
                neuron: i,
                total_delta,
                changes,
            });
        }
    }

    // Output layer
    let mut output_diffs = Vec::new();
    for o in 0..a.output_dim {
        let mut changes = Vec::new();

        for j in 0..hd {
            total_weights += 1;
            let delta = b.w_y[[o, j]] - a.w_y[[o, j]];
            if delta.abs() > threshold {
                total_changed += 1;
                all_deltas.push(delta.abs());
                changes.push(WeightChange {
                    matrix: "W_y",
                    col: j,
                    old_val: a.w_y[[o, j]],
                    new_val: b.w_y[[o, j]],
                    delta,
                });
            }
        }

        total_weights += 1;
        let delta = b.b_y[o] - a.b_y[o];
        if delta.abs() > threshold {
            total_changed += 1;
            all_deltas.push(delta.abs());
            changes.push(WeightChange {
                matrix: "b_y",
                col: 0,
                old_val: a.b_y[o],
                new_val: b.b_y[o],
                delta,
            });
        }

        let total_delta: f64 = changes.iter().map(|c| c.delta.abs()).sum();
        if !changes.is_empty() {
            output_diffs.push(OutputDiff {
                output_class: o,
                total_delta,
                changes,
            });
        }
    }

    let max_delta = all_deltas.iter().copied().fold(0.0_f64, f64::max);
    let mean_delta = if all_deltas.is_empty() {
        0.0
    } else {
        all_deltas.iter().sum::<f64>() / all_deltas.len() as f64
    };

    Ok(CircuitDiff {
        hidden_dim_a: a.hidden_dim,
        hidden_dim_b: b.hidden_dim,
        neuron_diffs,
        output_diffs,
        total_changed_weights: total_changed,
        total_weights,
        max_delta,
        mean_delta,
    })
}

fn fmt_delta(old: f64, new: f64) -> String {
    let fmt = |v: f64| -> String {
        if (v - v.round()).abs() < 0.01 {
            format!("{}", v.round() as i64)
        } else {
            format!("{:.2}", v)
        }
    };
    format!("{} → {}", fmt(old), fmt(new))
}

/// Format the diff report
pub fn format_diff(diff: &CircuitDiff) -> String {
    let mut out = String::new();

    out.push_str("═══ CIRCUIT DIFF ═══\n");
    out.push_str(&format!(
        "Hidden dim: {}  |  Changed: {}/{} weights ({:.0}%)\n",
        diff.hidden_dim_a,
        diff.total_changed_weights,
        diff.total_weights,
        diff.total_changed_weights as f64 / diff.total_weights as f64 * 100.0
    ));
    out.push_str(&format!(
        "Max Δ: {:.4}  Mean Δ: {:.4}\n\n",
        diff.max_delta, diff.mean_delta
    ));

    if diff.neuron_diffs.is_empty() && diff.output_diffs.is_empty() {
        out.push_str("No differences found — circuits are identical.\n");
        return out;
    }

    // Transition layer changes
    for nd in &diff.neuron_diffs {
        out.push_str(&format!(
            "── h{} (Σ|Δ| = {:.2}) ──\n",
            nd.neuron, nd.total_delta
        ));
        for c in &nd.changes {
            let loc = if c.matrix == "b_h" {
                format!("{}[{}]", c.matrix, nd.neuron)
            } else {
                format!("{}[{},{}]", c.matrix, nd.neuron, c.col)
            };
            out.push_str(&format!(
                "  {:<12} {}  (Δ {:+.4})\n",
                loc,
                fmt_delta(c.old_val, c.new_val),
                c.delta
            ));
        }
    }

    // Output layer changes
    for od in &diff.output_diffs {
        out.push_str(&format!(
            "── output[{}] (Σ|Δ| = {:.2}) ──\n",
            od.output_class, od.total_delta
        ));
        for c in &od.changes {
            let loc = if c.matrix == "b_y" {
                format!("{}[{}]", c.matrix, od.output_class)
            } else {
                format!("{}[{},{}]", c.matrix, od.output_class, c.col)
            };
            out.push_str(&format!(
                "  {:<12} {}  (Δ {:+.4})\n",
                loc,
                fmt_delta(c.old_val, c.new_val),
                c.delta
            ));
        }
    }

    out
}
