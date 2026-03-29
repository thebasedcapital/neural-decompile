use crate::quantize::QuantizedRnn;

/// Run the quantized RNN as a finite state machine on a sequence of inputs.
/// Each input is a vector (e.g., one-hot encoded).
/// Returns the predicted class (argmax of output logits).
pub fn run_fsm(q: &QuantizedRnn, input_sequence: &[Vec<f64>]) -> usize {
    let mut h = vec![0.0_f64; q.hidden_dim];

    for x in input_sequence {
        let mut h_new = vec![0.0; q.hidden_dim];
        for i in 0..q.hidden_dim {
            let mut val = q.b_h[i];
            for j in 0..q.hidden_dim {
                val += q.w_hh[[i, j]] * h[j];
            }
            for j in 0..q.input_dim {
                val += q.w_hx[[i, j]] * x[j];
            }
            h_new[i] = val.max(0.0); // ReLU
        }
        h = h_new;
    }

    // Output: argmax(W_y @ h + b_y)
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..q.output_dim {
        let mut logit = q.b_y[i];
        for j in 0..q.hidden_dim {
            logit += q.w_y[[i, j]] * h[j];
        }
        if logit > best_val {
            best_val = logit;
            best_idx = i;
        }
    }
    best_idx
}
