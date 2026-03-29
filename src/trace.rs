use crate::quantize::QuantizedRnn;
use crate::weights::RnnWeights;

/// A single timestep's state during RNN execution
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub t: usize,
    pub input: Vec<f64>,
    /// Pre-activation values for each hidden neuron
    pub pre_relu: Vec<f64>,
    /// Post-activation (after ReLU)
    pub hidden: Vec<f64>,
}

/// Full trace of an RNN execution
#[derive(Debug)]
pub struct Trace {
    pub steps: Vec<TraceStep>,
    pub output_logits: Vec<f64>,
    pub prediction: usize,
    pub hidden_dim: usize,
    pub input_dim: usize,
}

/// Run an RNN (quantized or raw) and record every hidden state
pub fn trace_quantized(q: &QuantizedRnn, input_sequence: &[Vec<f64>]) -> Trace {
    let mut h = vec![0.0_f64; q.hidden_dim];
    let mut steps = Vec::with_capacity(input_sequence.len());

    for (t, x) in input_sequence.iter().enumerate() {
        let mut pre_relu = vec![0.0; q.hidden_dim];
        let mut h_new = vec![0.0; q.hidden_dim];

        for i in 0..q.hidden_dim {
            let mut val = q.b_h[i];
            for j in 0..q.hidden_dim {
                val += q.w_hh[[i, j]] * h[j];
            }
            for j in 0..q.input_dim {
                val += q.w_hx[[i, j]] * x[j];
            }
            pre_relu[i] = val;
            h_new[i] = val.max(0.0);
        }

        steps.push(TraceStep {
            t,
            input: x.clone(),
            pre_relu,
            hidden: h_new.clone(),
        });
        h = h_new;
    }

    // Output logits
    let mut logits = vec![0.0; q.output_dim];
    for i in 0..q.output_dim {
        logits[i] = q.b_y[i];
        for j in 0..q.hidden_dim {
            logits[i] += q.w_y[[i, j]] * h[j];
        }
    }

    let prediction = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    Trace {
        steps,
        output_logits: logits,
        prediction,
        hidden_dim: q.hidden_dim,
        input_dim: q.input_dim,
    }
}

/// Run raw (unquantized) RNN and trace it
pub fn trace_raw(rnn: &RnnWeights, input_sequence: &[Vec<f64>]) -> Trace {
    let mut h = vec![0.0_f64; rnn.hidden_dim];
    let mut steps = Vec::with_capacity(input_sequence.len());

    for (t, x) in input_sequence.iter().enumerate() {
        let mut pre_relu = vec![0.0; rnn.hidden_dim];
        let mut h_new = vec![0.0; rnn.hidden_dim];

        for i in 0..rnn.hidden_dim {
            let mut val = rnn.b_h[i];
            for j in 0..rnn.hidden_dim {
                val += rnn.w_hh[[i, j]] * h[j];
            }
            for j in 0..rnn.input_dim {
                val += rnn.w_hx[[i, j]] * x[j];
            }
            pre_relu[i] = val;
            h_new[i] = val.max(0.0);
        }

        steps.push(TraceStep {
            t,
            input: x.clone(),
            pre_relu,
            hidden: h_new.clone(),
        });
        h = h_new;
    }

    let mut logits = vec![0.0; rnn.output_dim];
    for i in 0..rnn.output_dim {
        logits[i] = rnn.b_y[i];
        for j in 0..rnn.hidden_dim {
            logits[i] += rnn.w_y[[i, j]] * h[j];
        }
    }

    let prediction = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    Trace {
        steps,
        output_logits: logits,
        prediction,
        hidden_dim: rnn.hidden_dim,
        input_dim: rnn.input_dim,
    }
}

/// Format trace as a readable table
pub fn format_trace(trace: &Trace) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!("  t │ input"));
    for i in 0..trace.hidden_dim {
        out.push_str(&format!(" │ h{:<3}", i));
    }
    out.push('\n');

    let col_width = 6 + trace.hidden_dim * 7;
    out.push_str(&"─".repeat(col_width.min(120)));
    out.push('\n');

    // Initial state
    out.push_str(&format!("  - │     "));
    for _ in 0..trace.hidden_dim {
        out.push_str(&format!(" │ {:>4.1}", 0.0));
    }
    out.push_str("  (initial)\n");

    // Each timestep
    for step in &trace.steps {
        // Format input compactly
        let input_str = if step.input.len() <= 3 {
            step.input.iter().map(|v| format!("{:.0}", v)).collect::<Vec<_>>().join(",")
        } else {
            format!("[{}d]", step.input.len())
        };

        out.push_str(&format!("{:>3} │ {:>5}", step.t, input_str));

        for i in 0..trace.hidden_dim {
            let h = step.hidden[i];
            if h == 0.0 {
                out.push_str(" │    · ");
            } else if (h - h.round()).abs() < 0.01 {
                out.push_str(&format!(" │ {:>4.0}  ", h));
            } else {
                out.push_str(&format!(" │ {:>5.2}", h));
            }
        }

        // Show killed neurons (pre-relu negative, post-relu zero)
        let killed: Vec<usize> = (0..trace.hidden_dim)
            .filter(|&i| step.pre_relu[i] < 0.0 && step.hidden[i] == 0.0)
            .collect();
        if !killed.is_empty() {
            out.push_str(&format!("  ✗{}", killed.iter().map(|i| format!("h{}", i)).collect::<Vec<_>>().join(",")));
        }

        out.push('\n');
    }

    // Output
    out.push_str(&"─".repeat(col_width.min(120)));
    out.push('\n');
    out.push_str(&format!("  logits: [{}]",
        trace.output_logits.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>().join(", ")));
    out.push_str(&format!("  → class {}\n", trace.prediction));

    out
}
