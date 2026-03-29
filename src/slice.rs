use crate::quantize::QuantizedRnn;
use crate::trace::{trace_quantized, Trace};
use crate::verify::TestCase;
use ndarray::Array2;

/// Result of slicing: a smaller circuit + metadata about what was removed
#[derive(Debug)]
pub struct SliceResult {
    /// The pruned RNN (only active neurons remain)
    pub circuit: QuantizedRnn,
    /// Original neuron indices that survived (maps new_idx → old_idx)
    pub kept_neurons: Vec<usize>,
    /// Original neuron indices that were sliced away
    pub removed_neurons: Vec<usize>,
    /// For each kept neuron, max activation seen across all traces
    pub max_activations: Vec<f64>,
    /// Total number of input traces used for analysis
    pub num_traces: usize,
    /// Original hidden dim
    pub original_hidden_dim: usize,
}

/// Analyze which neurons are active across a set of traces
fn find_active_neurons(traces: &[Trace], hidden_dim: usize) -> Vec<bool> {
    let mut ever_active = vec![false; hidden_dim];

    for trace in traces {
        for step in &trace.steps {
            for i in 0..hidden_dim {
                if step.hidden[i] > 0.0 {
                    ever_active[i] = true;
                }
            }
        }
    }

    ever_active
}

/// Compute max activation per neuron across all traces
fn max_activations(traces: &[Trace], hidden_dim: usize) -> Vec<f64> {
    let mut maxes = vec![0.0_f64; hidden_dim];

    for trace in traces {
        for step in &trace.steps {
            for i in 0..hidden_dim {
                maxes[i] = maxes[i].max(step.hidden[i]);
            }
        }
    }

    maxes
}

/// Also check: a neuron might fire but never influence the output.
/// If W_y[:,i] is all zero AND no other active neuron reads from it
/// (W_hh[j,i] = 0 for all active j), it's output-dead.
fn find_output_reachable(q: &QuantizedRnn, ever_active: &[bool]) -> Vec<bool> {
    let hd = q.hidden_dim;
    let mut reachable = vec![false; hd];

    // Phase 1: neurons directly connected to output
    for i in 0..hd {
        if !ever_active[i] {
            continue;
        }
        for o in 0..q.output_dim {
            if q.w_y[[o, i]].abs() > 0.005 {
                reachable[i] = true;
                break;
            }
        }
    }

    // Phase 2: backward propagation — if neuron j is reachable and reads from i,
    // then i is also reachable (iterate until stable)
    loop {
        let mut changed = false;
        for j in 0..hd {
            if !reachable[j] || !ever_active[j] {
                continue;
            }
            for i in 0..hd {
                if !ever_active[i] || reachable[i] {
                    continue;
                }
                if q.w_hh[[j, i]].abs() > 0.005 {
                    reachable[i] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    reachable
}

/// Slice a circuit using traces from test cases
pub fn slice_from_tests(q: &QuantizedRnn, tests: &[TestCase]) -> SliceResult {
    let traces: Vec<Trace> = tests
        .iter()
        .map(|tc| trace_quantized(q, &tc.inputs))
        .collect();

    slice_from_traces(q, &traces)
}

/// Slice a circuit using pre-computed traces
pub fn slice_from_traces(q: &QuantizedRnn, traces: &[Trace]) -> SliceResult {
    let hd = q.hidden_dim;
    let ever_active = find_active_neurons(traces, hd);
    let reachable = find_output_reachable(q, &ever_active);
    let maxes = max_activations(traces, hd);

    // A neuron survives only if it's both active AND output-reachable
    let kept: Vec<usize> = (0..hd)
        .filter(|&i| ever_active[i] && reachable[i])
        .collect();
    let removed: Vec<usize> = (0..hd)
        .filter(|&i| !ever_active[i] || !reachable[i])
        .collect();

    let new_hd = kept.len();

    // Build sliced weight matrices
    let mut new_w_hh = Array2::<f64>::zeros((new_hd, new_hd));
    let mut new_w_hx = Array2::<f64>::zeros((new_hd, q.input_dim));
    let mut new_b_h = vec![0.0; new_hd];
    let mut new_w_y = Array2::<f64>::zeros((q.output_dim, new_hd));
    let new_b_y = q.b_y.clone();

    for (ni, &oi) in kept.iter().enumerate() {
        // W_hh: new[ni, nj] = old[oi, oj]
        for (nj, &oj) in kept.iter().enumerate() {
            new_w_hh[[ni, nj]] = q.w_hh[[oi, oj]];
        }
        // W_hx: new[ni, j] = old[oi, j]
        for j in 0..q.input_dim {
            new_w_hx[[ni, j]] = q.w_hx[[oi, j]];
        }
        // b_h
        new_b_h[ni] = q.b_h[oi];
        // W_y: new[o, ni] = old[o, oi]
        for o in 0..q.output_dim {
            new_w_y[[o, ni]] = q.w_y[[o, oi]];
        }
    }

    let kept_maxes: Vec<f64> = kept.iter().map(|&i| maxes[i]).collect();

    SliceResult {
        circuit: QuantizedRnn {
            w_hh: new_w_hh,
            w_hx: new_w_hx,
            b_h: new_b_h,
            w_y: new_w_y,
            b_y: new_b_y,
            hidden_dim: new_hd,
            input_dim: q.input_dim,
            output_dim: q.output_dim,
        },
        kept_neurons: kept,
        removed_neurons: removed,
        max_activations: kept_maxes,
        num_traces: traces.len(),
        original_hidden_dim: hd,
    }
}

/// Format the slice report
pub fn format_slice(result: &SliceResult) -> String {
    let mut out = String::new();

    out.push_str("═══ CIRCUIT SLICE ═══\n");
    out.push_str(&format!(
        "Traced {} inputs across {} neurons\n",
        result.num_traces, result.original_hidden_dim
    ));
    out.push_str(&format!(
        "Active circuit: {} neurons (sliced {} dead)\n\n",
        result.kept_neurons.len(),
        result.removed_neurons.len()
    ));

    // Neuron mapping table
    out.push_str("── Neuron Map ──\n");
    out.push_str(&format!(
        "{:>5}  {:>5}  {:>10}  {}\n",
        "old", "new", "max_act", "status"
    ));
    let mut new_idx = 0;
    for i in 0..result.original_hidden_dim {
        if result.kept_neurons.contains(&i) {
            out.push_str(&format!(
                "  h{:<3} → h{:<3}  {:>8.2}  KEPT\n",
                i, new_idx, result.max_activations[new_idx]
            ));
            new_idx += 1;
        } else {
            out.push_str(&format!("  h{:<3}   ···   {:>8}  SLICED\n", i, "—"));
        }
    }

    // Compression ratio
    let old_params = result.original_hidden_dim * result.original_hidden_dim
        + result.original_hidden_dim * result.circuit.input_dim
        + result.original_hidden_dim
        + result.circuit.output_dim * result.original_hidden_dim
        + result.circuit.output_dim;
    let new_hd = result.kept_neurons.len();
    let new_params = new_hd * new_hd
        + new_hd * result.circuit.input_dim
        + new_hd
        + result.circuit.output_dim * new_hd
        + result.circuit.output_dim;
    let ratio = 1.0 - (new_params as f64 / old_params as f64);

    out.push_str(&format!(
        "\nParameters: {} → {} ({:.0}% reduction)\n",
        old_params,
        new_params,
        ratio * 100.0
    ));

    out
}
