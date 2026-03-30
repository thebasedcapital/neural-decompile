use crate::weights::RnnWeights;
use crate::transformer::Transformer;
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

/// Quantized transformer layer
#[derive(Debug, Clone)]
pub struct QuantizedLayer {
    pub w_q: Vec<f64>,
    pub w_k: Vec<f64>,
    pub w_v: Vec<f64>,
    pub w_o: Vec<f64>,
    pub b_q: Option<Vec<f64>>,
    pub b_k: Option<Vec<f64>>,
    pub b_v: Option<Vec<f64>>,
    pub b_o: Option<Vec<f64>>,
    pub w_ff_in: Vec<f64>,
    pub b_ff_in: Option<Vec<f64>>,
    pub w_ff_out: Vec<f64>,
    pub b_ff_out: Option<Vec<f64>>,
    pub ln1_gamma: Vec<f64>,
    pub ln1_beta: Vec<f64>,
    pub ln2_gamma: Vec<f64>,
    pub ln2_beta: Vec<f64>,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub gelu: bool,
}

/// Quantized transformer — all weights snapped where within eps
#[derive(Debug, Clone)]
pub struct QuantizedTransformer {
    pub n_layers: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub token_emb: Vec<Vec<f64>>,
    pub pos_emb: Option<Vec<Vec<f64>>>,
    pub layers: Vec<QuantizedLayer>,
    pub ln_final_gamma: Option<Vec<f64>>,
    pub ln_final_beta: Option<Vec<f64>>,
    pub w_out: Vec<f64>,
    pub b_out: Option<Vec<f64>>,
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

/// Quantize optional vec
fn quantize_opt_vec(v: Option<&Vec<f64>>, eps: f64) -> Option<Vec<f64>> {
    v.map(|vec| quantize_vec(vec, eps))
}

/// Quantize 2D vec
fn quantize_vec2d(v: &[Vec<f64>], eps: f64) -> Vec<Vec<f64>> {
    v.iter().map(|row| quantize_vec(row, eps)).collect()
}

/// Quantize a transformer
pub fn quantize_transformer(t: &Transformer, eps: f64) -> QuantizedTransformer {
    let layers: Vec<QuantizedLayer> = t.layers.iter().map(|l| {
        QuantizedLayer {
            w_q: quantize_vec(&l.w_q, eps),
            w_k: quantize_vec(&l.w_k, eps),
            w_v: quantize_vec(&l.w_v, eps),
            w_o: quantize_vec(&l.w_o, eps),
            b_q: quantize_opt_vec(l.b_q.as_ref(), eps),
            b_k: quantize_opt_vec(l.b_k.as_ref(), eps),
            b_v: quantize_opt_vec(l.b_v.as_ref(), eps),
            b_o: quantize_opt_vec(l.b_o.as_ref(), eps),
            w_ff_in: quantize_vec(&l.w_ff_in, eps),
            b_ff_in: quantize_opt_vec(l.b_ff_in.as_ref(), eps),
            w_ff_out: quantize_vec(&l.w_ff_out, eps),
            b_ff_out: quantize_opt_vec(l.b_ff_out.as_ref(), eps),
            ln1_gamma: quantize_vec(&l.ln1_gamma, eps),
            ln1_beta: quantize_vec(&l.ln1_beta, eps),
            ln2_gamma: quantize_vec(&l.ln2_gamma, eps),
            ln2_beta: quantize_vec(&l.ln2_beta, eps),
            d_model: l.d_model,
            n_heads: l.n_heads,
            d_ff: l.d_ff,
            gelu: l.gelu,
        }
    }).collect();

    QuantizedTransformer {
        n_layers: t.n_layers,
        d_model: t.d_model,
        vocab_size: t.vocab_size,
        max_seq_len: t.max_seq_len,
        token_emb: quantize_vec2d(&t.token_emb, eps),
        pos_emb: t.pos_emb.as_ref().map(|p| quantize_vec2d(p, eps)),
        layers,
        ln_final_gamma: quantize_opt_vec(t.ln_final_gamma.as_ref(), eps),
        ln_final_beta: quantize_opt_vec(t.ln_final_beta.as_ref(), eps),
        w_out: quantize_vec(&t.w_out, eps),
        b_out: quantize_opt_vec(t.b_out.as_ref(), eps),
    }
}

/// Stats for transformer weights
pub struct TransformerStats {
    pub total_params: usize,
    pub integer_count: usize,
    pub pct_integer: f64,
    pub layer_stats: Vec<LayerStats>,
}

pub struct LayerStats {
    pub attn_int: usize,
    pub attn_total: usize,
    pub ffn_int: usize,
    pub ffn_total: usize,
}

pub fn transformer_stats(t: &QuantizedTransformer) -> TransformerStats {
    let mut all_weights: Vec<f64> = Vec::new();
    let mut layer_stats = Vec::new();

    // Embeddings
    for row in &t.token_emb {
        all_weights.extend(row.iter().copied());
    }
    if let Some(ref pos) = t.pos_emb {
        for row in pos {
            all_weights.extend(row.iter().copied());
        }
    }

    // Layers
    for layer in &t.layers {
        let mut attn_weights: Vec<f64> = Vec::new();
        attn_weights.extend(layer.w_q.iter().copied());
        attn_weights.extend(layer.w_k.iter().copied());
        attn_weights.extend(layer.w_v.iter().copied());
        attn_weights.extend(layer.w_o.iter().copied());

        let mut ffn_weights: Vec<f64> = Vec::new();
        ffn_weights.extend(layer.w_ff_in.iter().copied());
        ffn_weights.extend(layer.w_ff_out.iter().copied());

        let attn_int = attn_weights.iter().filter(|&&v| (v - v.round()).abs() < 0.01).count();
        let ffn_int = ffn_weights.iter().filter(|&&v| (v - v.round()).abs() < 0.01).count();

        layer_stats.push(LayerStats {
            attn_int,
            attn_total: attn_weights.len(),
            ffn_int,
            ffn_total: ffn_weights.len(),
        });

        all_weights.extend(attn_weights);
        all_weights.extend(ffn_weights);
        all_weights.extend(layer.ln1_gamma.iter().copied());
        all_weights.extend(layer.ln1_beta.iter().copied());
        all_weights.extend(layer.ln2_gamma.iter().copied());
        all_weights.extend(layer.ln2_beta.iter().copied());
    }

    // Output
    all_weights.extend(t.w_out.iter().copied());
    if let Some(ref b) = t.b_out {
        all_weights.extend(b.iter().copied());
    }
    if let Some(ref g) = t.ln_final_gamma {
        all_weights.extend(g.iter().copied());
    }
    if let Some(ref b) = t.ln_final_beta {
        all_weights.extend(b.iter().copied());
    }

    let total = all_weights.len();
    let integer_count = all_weights.iter().filter(|&&v| (v - v.round()).abs() < 0.01).count();

    TransformerStats {
        total_params: total,
        integer_count,
        pct_integer: integer_count as f64 / total as f64,
        layer_stats,
    }
}
