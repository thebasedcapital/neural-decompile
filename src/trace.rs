use crate::quantize::{QuantizedRnn, QuantizedTransformer, QuantizedLayer};
use crate::weights::RnnWeights;
use crate::transformer::Transformer;

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

// ============================================================================
// Transformer tracing — trace through layers instead of timesteps
// ============================================================================

/// Attention pattern for a single head at a specific layer
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub head_idx: usize,
    /// Attention weights [query_pos][key_pos]
    pub weights: Vec<Vec<f64>>,
}

/// State after processing one transformer layer
#[derive(Debug, Clone)]
pub struct LayerTrace {
    pub layer_idx: usize,
    /// Hidden states after this layer [seq_len][d_model]
    pub hidden: Vec<Vec<f64>>,
    /// Attention patterns for each head
    pub attention_patterns: Vec<AttentionPattern>,
    /// Per-token FFN activations (post-activation)
    pub ffn_activations: Vec<Vec<f64>>,
}

/// Full trace of a transformer forward pass
#[derive(Debug)]
pub struct TransformerTrace {
    pub layers: Vec<LayerTrace>,
    pub initial_emb: Vec<Vec<f64>>,
    pub output_logits: Vec<Vec<f64>>,
    pub predictions: Vec<usize>,
    pub n_layers: usize,
    pub d_model: usize,
    pub seq_len: usize,
}

/// Trace a transformer forward pass through all layers
pub fn trace_transformer(t: &Transformer, tokens: &[usize]) -> TransformerTrace {
    let seq_len = tokens.len();
    let d_model = t.d_model;
    let mut layers = Vec::with_capacity(t.n_layers);

    // Initial embeddings
    let mut hidden: Vec<Vec<f64>> = tokens.iter().enumerate().map(|(i, &tok)| {
        let mut emb = t.token_emb[tok].clone();
        if let Some(ref pos) = t.pos_emb {
            for j in 0..d_model {
                emb[j] += pos[i][j];
            }
        }
        emb
    }).collect();

    let initial_emb = hidden.clone();

    // Pass through each layer
    for (li, layer) in t.layers.iter().enumerate() {
        let (new_hidden, attn_patterns, ffn_act) = trace_transformer_layer(layer, &hidden, d_model, seq_len);
        layers.push(LayerTrace {
            layer_idx: li,
            hidden: new_hidden.clone(),
            attention_patterns: attn_patterns,
            ffn_activations: ffn_act,
        });
        hidden = new_hidden;
    }

    // Final layer norm
    if let (Some(g), Some(b)) = (&t.ln_final_gamma, &t.ln_final_beta) {
        for h in &mut hidden {
            *h = layer_norm(h, g, b);
        }
    }

    // Output projection
    let output_logits: Vec<Vec<f64>> = hidden.iter().map(|h| {
        let mut out = vec![0.0; t.vocab_size];
        for (j, v) in out.iter_mut().enumerate() {
            for (i, x) in h.iter().enumerate() {
                *v += x * t.w_out[i * t.vocab_size + j];
            }
            if let Some(ref b) = t.b_out {
                *v += b[j];
            }
        }
        out
    }).collect();

    let predictions: Vec<usize> = output_logits.iter().map(|logits| {
        logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0
    }).collect();

    TransformerTrace {
        layers,
        initial_emb,
        output_logits,
        predictions,
        n_layers: t.n_layers,
        d_model,
        seq_len,
    }
}

/// Trace through a single transformer layer, capturing attention and FFN
fn trace_transformer_layer(
    layer: &crate::transformer::TransformerBlock,
    hidden: &[Vec<f64>],
    d_model: usize,
    seq_len: usize,
) -> (Vec<Vec<f64>>, Vec<AttentionPattern>, Vec<Vec<f64>>) {
    let head_dim = d_model / layer.n_heads;

    // Pre-norm
    let x_norm: Vec<Vec<f64>> = hidden.iter().map(|h| {
        layer_norm(h, &layer.ln1_gamma, &layer.ln1_beta)
    }).collect();

    // Q, K, V projections
    let q = matmul_seq(&x_norm, &layer.w_q, d_model, d_model, layer.b_q.as_ref());
    let k = matmul_seq(&x_norm, &layer.w_k, d_model, d_model, layer.b_k.as_ref());
    let v = matmul_seq(&x_norm, &layer.w_v, d_model, d_model, layer.b_v.as_ref());

    // Multi-head attention with pattern capture
    let mut attn_out = vec![vec![0.0; d_model]; seq_len];
    let mut attn_patterns = Vec::with_capacity(layer.n_heads);

    for h_idx in 0..layer.n_heads {
        let h_off = h_idx * head_dim;

        // Compute attention scores for this head
        let mut scores = vec![vec![0.0; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                for d in 0..head_dim {
                    scores[i][j] += q[i][h_off + d] * k[j][h_off + d];
                }
                scores[i][j] /= (head_dim as f64).sqrt();
            }
        }

        // Softmax
        let attn_weights = softmax_2d(&scores);
        attn_patterns.push(AttentionPattern {
            head_idx: h_idx,
            weights: attn_weights.clone(),
        });

        // Apply attention to values
        for i in 0..seq_len {
            for d in 0..head_dim {
                for j in 0..seq_len {
                    attn_out[i][h_off + d] += attn_weights[i][j] * v[j][h_off + d];
                }
            }
        }
    }

    // Output projection
    let attn_proj = matmul_seq(&attn_out, &layer.w_o, d_model, d_model, layer.b_o.as_ref());

    // Residual
    let mut after_attn: Vec<Vec<f64>> = hidden.iter().zip(attn_proj.iter())
        .map(|(h, a)| h.iter().zip(a.iter()).map(|(x, y)| x + y).collect())
        .collect();

    // Pre-norm FFN
    let x_ffn: Vec<Vec<f64>> = after_attn.iter().map(|h| {
        layer_norm(h, &layer.ln2_gamma, &layer.ln2_beta)
    }).collect();

    // FFN layer 1
    let mut ffn_hidden = vec![vec![0.0; layer.d_ff]; seq_len];
    for (i, h) in x_ffn.iter().enumerate() {
        for j in 0..layer.d_ff {
            let mut sum = layer.b_ff_in.as_ref().map(|b| b[j]).unwrap_or(0.0);
            for k in 0..d_model {
                sum += h[k] * layer.w_ff_in[k * layer.d_ff + j];
            }
            ffn_hidden[i][j] = sum;
        }
    }

    // Activation
    for h in &mut ffn_hidden {
        for v in h {
            *v = if layer.gelu { gelu(*v) } else { v.max(0.0) };
        }
    }

    let ffn_activations = ffn_hidden.clone();

    // FFN layer 2
    let mut ffn_out = vec![vec![0.0; d_model]; seq_len];
    for (i, h) in ffn_hidden.iter().enumerate() {
        for j in 0..d_model {
            let mut sum = layer.b_ff_out.as_ref().map(|b| b[j]).unwrap_or(0.0);
            for k in 0..layer.d_ff {
                sum += h[k] * layer.w_ff_out[k * d_model + j];
            }
            ffn_out[i][j] = sum;
        }
    }

    // Residual
    for i in 0..seq_len {
        for j in 0..d_model {
            after_attn[i][j] += ffn_out[i][j];
        }
    }

    (after_attn, attn_patterns, ffn_activations)
}

fn matmul_seq(x: &[Vec<f64>], w: &[f64], in_dim: usize, out_dim: usize, bias: Option<&Vec<f64>>) -> Vec<Vec<f64>> {
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

fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + 1e-5).sqrt();

    x.iter().enumerate().map(|(i, &v)| {
        (v - mean) / std * gamma[i] + beta[i]
    }).collect()
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

/// Format transformer trace for display
pub fn format_transformer_trace(trace: &TransformerTrace) -> String {
    let mut out = String::new();

    out.push_str(&format!("=== Transformer Trace ===\n"));
    out.push_str(&format!("Layers: {}, d_model: {}, seq_len: {}\n\n", trace.n_layers, trace.d_model, trace.seq_len));

    // Initial embeddings
    out.push_str("Initial Embeddings:\n");
    for (i, emb) in trace.initial_emb.iter().enumerate() {
        out.push_str(&format!("  tok[{}]: mean={:.3} std={:.3}\n", i,
            emb.iter().sum::<f64>() / emb.len() as f64,
            (emb.iter().map(|&x| x * x).sum::<f64>() / emb.len() as f64).sqrt()));
    }
    out.push('\n');

    // Per-layer info
    for layer in &trace.layers {
        out.push_str(&format!("--- Layer {} ---\n", layer.layer_idx));

        // Attention patterns
        for attn in &layer.attention_patterns {
            out.push_str(&format!("  Head {}: ", attn.head_idx));
            for (i, row) in attn.weights.iter().enumerate() {
                if i > 0 { out.push_str("         "); }
                out.push_str(&format!("[{}] ", i));
                for (_j, &w) in row.iter().enumerate() {
                    let symbol = if w > 0.5 { "██" }
                        else if w > 0.25 { "▓▓" }
                        else if w > 0.1 { "░░" }
                        else { "  " };
                    out.push_str(symbol);
                }
                out.push('\n');
            }
        }

        // FFN stats
        let total_ffn_neurons: usize = layer.ffn_activations.iter().map(|v| v.len()).sum();
        let active_ffn: usize = layer.ffn_activations.iter()
            .flat_map(|v| v.iter().filter(|&&x| x > 0.0))
            .count();
        out.push_str(&format!("  FFN: {}/{} neurons active ({:.1}%)\n",
            active_ffn, total_ffn_neurons,
            100.0 * active_ffn as f64 / total_ffn_neurons as f64));

        // Hidden stats
        let norms: Vec<f64> = layer.hidden.iter().map(|h| {
            (h.iter().map(|&x| x * x).sum::<f64>()).sqrt()
        }).collect();
        out.push_str(&format!("  Hidden L2 norms: [{}]\n",
            norms.iter().map(|&x| format!("{:.2}", x)).collect::<Vec<_>>().join(", ")));
        out.push('\n');
    }

    // Output
    out.push_str("Output:\n");
    for (i, logits) in trace.output_logits.iter().enumerate() {
        out.push_str(&format!("  pos[{}]: pred={} logits=[{}]\n", i, trace.predictions[i],
            logits.iter().take(5).map(|&x| format!("{:.2}", x)).collect::<Vec<_>>().join(", ")));
    }

    out
}
