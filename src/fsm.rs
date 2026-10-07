use crate::quantize::QuantizedRnn;

/// Run the quantized RNN as a finite state machine on a sequence of inputs.
/// Each input is a vector (e.g., one-hot encoded).
/// Returns the predicted class (argmax of output logits).
pub fn run_fsm(q: &QuantizedRnn, input_sequence: &[Vec<f64>]) -> usize {
    let mut h = vec![0.0_f64; q.hidden_dim];

    for x in input_sequence {
        let mut h_new = vec![0.0; q.hidden_dim];
        for (i, h_i) in h_new.iter_mut().enumerate() {
            let mut val = q.b_h[i];
            for (j, &h_j) in h.iter().enumerate() {
                val += q.w_hh[[i, j]] * h_j;
            }
            for (j, &x_j) in x[..q.input_dim].iter().enumerate() {
                val += q.w_hx[[i, j]] * x_j;
            }
            *h_i = val.max(0.0); // ReLU
        }
        h = h_new;
    }

    // Output: argmax(W_y @ h + b_y)
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..q.output_dim {
        let mut logit = q.b_y[i];
        for (j, &h_j) in h.iter().enumerate() {
            logit += q.w_y[[i, j]] * h_j;
        }
        if logit > best_val {
            best_val = logit;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// 1-neuron "have I seen a 1?" detector: h = ReLU(h + x), class = (h > 0).
    fn seen_one() -> QuantizedRnn {
        QuantizedRnn {
            w_hh: arr2(&[[1.0]]),
            w_hx: arr2(&[[1.0]]),
            b_h: vec![0.0],
            w_y: arr2(&[[0.0], [1.0]]),
            b_y: vec![0.5, 0.0],
            hidden_dim: 1,
            input_dim: 1,
            output_dim: 2,
        }
    }

    fn seq(bits: &[f64]) -> Vec<Vec<f64>> {
        bits.iter().map(|&b| vec![b]).collect()
    }

    #[test]
    fn empty_sequence_uses_bias_only() {
        // h = 0 -> logits = b_y = [0.5, 0.0] -> class 0
        assert_eq!(run_fsm(&seen_one(), &[]), 0);
    }

    #[test]
    fn state_persists_across_steps() {
        let q = seen_one();
        assert_eq!(run_fsm(&q, &seq(&[0.0, 0.0, 0.0])), 0);
        assert_eq!(run_fsm(&q, &seq(&[0.0, 1.0, 0.0])), 1);
        assert_eq!(run_fsm(&q, &seq(&[1.0])), 1);
    }

    #[test]
    fn relu_clamps_negative_preactivations() {
        let mut q = seen_one();
        q.w_hx = arr2(&[[-1.0]]);
        // h would go to -3 without ReLU, flipping the logits; with ReLU h stays 0
        assert_eq!(run_fsm(&q, &seq(&[1.0, 1.0, 1.0])), 0);
    }

    #[test]
    fn ties_resolve_to_lowest_class() {
        let mut q = seen_one();
        q.b_y = vec![0.0, 0.0];
        assert_eq!(run_fsm(&q, &[]), 0);
    }
}
