use crate::weights::RnnWeights;
use ndarray::Array2;

/// Quantized RNN — weights snapped to nearest integer where within eps
#[derive(Debug, Clone)]
pub struct QuantizedRnn {
    pub w_hh: Array2<f64>,
    pub w_hx: Array2<f64>,
    pub b_h: Vec<f64>,
    pub w_y: Array2<f64>,
    pub b_y: Vec<f64>,
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub struct WeightStats {
    pub total_weights: usize,
    pub integer_count: usize,
    pub zero_count: usize,
    pub pct_integer: f64,
    pub dead_neurons: Vec<usize>,
    pub max_abs: f64,
    pub linf_to_int: f64,
}

fn quantize_val(v: f64, eps: f64) -> f64 {
    let rounded = v.round();
    if (v - rounded).abs() < eps { rounded } else { v }
}

fn quantize_array(arr: &Array2<f64>, eps: f64) -> Array2<f64> {
    arr.mapv(|v| quantize_val(v, eps))
}

fn quantize_vec(v: &[f64], eps: f64) -> Vec<f64> {
    v.iter().map(|&x| quantize_val(x, eps)).collect()
}

pub fn quantize_rnn(rnn: &RnnWeights, eps: f64) -> QuantizedRnn {
    QuantizedRnn {
        w_hh: quantize_array(&rnn.w_hh, eps),
        w_hx: quantize_array(&rnn.w_hx, eps),
        b_h: quantize_vec(&rnn.b_h, eps),
        w_y: quantize_array(&rnn.w_y, eps),
        b_y: quantize_vec(&rnn.b_y, eps),
        hidden_dim: rnn.hidden_dim,
        input_dim: rnn.input_dim,
        output_dim: rnn.output_dim,
    }
}

pub fn weight_stats(q: &QuantizedRnn) -> WeightStats {
    let all: Vec<f64> = q.w_hh.iter().copied()
        .chain(q.w_hx.iter().copied())
        .chain(q.b_h.iter().copied())
        .chain(q.w_y.iter().copied())
        .chain(q.b_y.iter().copied())
        .collect();

    let total = all.len();
    let integer_count = all.iter().filter(|&&v| (v - v.round()).abs() < 0.01).count();
    let zero_count = all.iter().filter(|&&v| v.abs() < 0.01).count();
    let max_abs = all.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let linf_to_int = all.iter().map(|&v| (v - v.round()).abs()).fold(0.0_f64, f64::max);

    // Dead neurons: hidden dims where all incoming AND outgoing weights are zero
    let mut dead_neurons = Vec::new();
    for i in 0..q.hidden_dim {
        let incoming_zero = (0..q.hidden_dim).all(|j| q.w_hh[[j, i]].abs() < 0.01)
            && (0..q.input_dim).all(|j| q.w_hx[[i, j]].abs() < 0.01)
            && q.b_h[i].abs() < 0.01;
        let outgoing_zero = (0..q.output_dim).all(|j| q.w_y[[j, i]].abs() < 0.01)
            && (0..q.hidden_dim).all(|j| q.w_hh[[i, j]].abs() < 0.01);
        if incoming_zero || outgoing_zero {
            dead_neurons.push(i);
        }
    }

    WeightStats {
        total_weights: total,
        integer_count,
        zero_count,
        pct_integer: integer_count as f64 / total as f64,
        dead_neurons,
        max_abs,
        linf_to_int,
    }
}
