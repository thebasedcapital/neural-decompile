use anyhow::Result;
use serde::Deserialize;

/// Transformer block weights for a single layer.
/// Supports both pre-norm and post-norm variants.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Hidden dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// FFN intermediate dimension (usually 4*d_model)
    pub d_ff: usize,
    /// Whether to use GELU (vs ReLU) in FFN
    pub gelu: bool,
    
    /// Attention: Q, K, V, O projections
    /// W_Q: [d_model, d_model] (or [n_heads, head_dim, d_model] if reshaped)
    pub w_q: Vec<f64>,
    pub w_k: Vec<f64>,
    pub w_v: Vec<f64>,
    pub w_o: Vec<f64>,
    
    /// Attention biases (optional, often zero)
    pub b_q: Option<Vec<f64>>,
    pub b_k: Option<Vec<f64>>,
    pub b_v: Option<Vec<f64>>,
    pub b_o: Option<Vec<f64>>,
    
    /// FFN: two linear layers
    pub w_ff_in: Vec<f64>,  // [d_model, d_ff]
    pub b_ff_in: Option<Vec<f64>>,
    pub w_ff_out: Vec<f64>, // [d_ff, d_model]
    pub b_ff_out: Option<Vec<f64>>,
    
    /// Layer normalization: scale (gamma) and shift (beta)
    /// Pre-norm: LN before attention and FFN
    /// Post-norm: LN after residual
    pub ln1_gamma: Vec<f64>,
    pub ln1_beta: Vec<f64>,
    pub ln2_gamma: Vec<f64>,
    pub ln2_beta: Vec<f64>,
}

/// Full transformer: embedding + layers + output head
#[derive(Debug, Clone)]
pub struct Transformer {
    pub n_layers: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    
    /// Token + position embeddings
    pub token_emb: Vec<Vec<f64>>, // [vocab_size, d_model]
    pub pos_emb: Option<Vec<Vec<f64>>>, // [max_seq_len, d_model]
    
    /// Transformer blocks
    pub layers: Vec<TransformerBlock>,
    
    /// Final layer norm (for GPT-style)
    pub ln_final_gamma: Option<Vec<f64>>,
    pub ln_final_beta: Option<Vec<f64>>,
    
    /// Output projection (to vocab). Often tied with token_emb.
    pub w_out: Vec<f64>, // [d_model, vocab_size] or [d_model, d_model] + separate unembed
    pub b_out: Option<Vec<f64>>,
}

/// JSON format for loading
#[derive(Deserialize)]
struct TransformerJson {
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_layers: usize,
    max_seq_len: usize,
    gelu: Option<bool>,
    #[serde(default)]
    vocab_size: usize,
    
    token_emb: Vec<Vec<f64>>,
    pos_emb: Option<Vec<Vec<f64>>>,
    
    layers: Vec<LayerJson>,
    
    ln_final_gamma: Option<Vec<f64>>,
    ln_final_beta: Option<Vec<f64>>,
    
    w_out: Vec<Vec<f64>>,
    b_out: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct LayerJson {
    w_q: Vec<Vec<f64>>,
    w_k: Vec<Vec<f64>>,
    w_v: Vec<Vec<f64>>,
    w_o: Vec<Vec<f64>>,
    
    b_q: Option<Vec<f64>>,
    b_k: Option<Vec<f64>>,
    b_v: Option<Vec<f64>>,
    b_o: Option<Vec<f64>>,
    
    w_ff_in: Vec<Vec<f64>>,
    b_ff_in: Option<Vec<f64>>,
    w_ff_out: Vec<Vec<f64>>,
    b_ff_out: Option<Vec<f64>>,
    
    ln1_gamma: Vec<f64>,
    ln1_beta: Vec<f64>,
    ln2_gamma: Vec<f64>,
    ln2_beta: Vec<f64>,
}

impl Transformer {
    pub fn from_json(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let json: TransformerJson = serde_json::from_str(&data)?;
        
        let vocab_size = json.vocab_size.max(json.token_emb.len());
        let d_model = json.d_model;
        
        let layers: Vec<TransformerBlock> = json.layers.into_iter().map(|l| {
            TransformerBlock {
                d_model,
                n_heads: json.n_heads,
                d_ff: json.d_ff,
                gelu: json.gelu.unwrap_or(false),
                w_q: l.w_q.into_iter().flatten().collect(),
                w_k: l.w_k.into_iter().flatten().collect(),
                w_v: l.w_v.into_iter().flatten().collect(),
                w_o: l.w_o.into_iter().flatten().collect(),
                b_q: l.b_q,
                b_k: l.b_k,
                b_v: l.b_v,
                b_o: l.b_o,
                w_ff_in: l.w_ff_in.into_iter().flatten().collect(),
                b_ff_in: l.b_ff_in,
                w_ff_out: l.w_ff_out.into_iter().flatten().collect(),
                b_ff_out: l.b_ff_out,
                ln1_gamma: l.ln1_gamma,
                ln1_beta: l.ln1_beta,
                ln2_gamma: l.ln2_gamma,
                ln2_beta: l.ln2_beta,
            }
        }).collect();
        
        Ok(Transformer {
            n_layers: json.n_layers,
            d_model,
            vocab_size,
            max_seq_len: json.max_seq_len,
            token_emb: json.token_emb,
            pos_emb: json.pos_emb,
            layers,
            ln_final_gamma: json.ln_final_gamma,
            ln_final_beta: json.ln_final_beta,
            w_out: json.w_out.into_iter().flatten().collect(),
            b_out: json.b_out,
        })
    }
    
    /// Run forward pass on a sequence of token IDs
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f64>> {
        assert!(!tokens.is_empty() && tokens.len() <= self.max_seq_len);

        // Token + positional embeddings
        let mut hidden: Vec<Vec<f64>> = tokens.iter().enumerate().map(|(i, &tok)| {
            let mut emb = self.token_emb[tok].clone();
            if let Some(ref pos) = self.pos_emb {
                // Add position embedding
                for j in 0..self.d_model {
                    emb[j] += pos[i][j];
                }
            }
            emb
        }).collect();

        // Pass through layers
        for layer in &self.layers {
            hidden = self.forward_layer(layer, &hidden);
        }

        // Final layer norm
        if let (Some(g), Some(b)) = (&self.ln_final_gamma, &self.ln_final_beta) {
            for h in &mut hidden {
                *h = layer_norm(h, g, b);
            }
        }

        // Project to vocab
        let logits: Vec<Vec<f64>> = hidden.iter().map(|h| {
            // h: [d_model], w_out: [d_model, vocab_size]
            let mut out = vec![0.0; self.vocab_size];
            for (j, v) in out.iter_mut().enumerate() {
                for (i, x) in h.iter().enumerate() {
                    *v += self.w_out[i * self.vocab_size + j] * x;
                }
                if let Some(ref b) = self.b_out {
                    *v += b[j];
                }
            }
            out
        }).collect();
        
        logits
    }
    
    fn forward_layer(&self, layer: &TransformerBlock, hidden: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = hidden.len();
        let _head_dim = self.d_model / layer.n_heads;
        
        // Pre-norm
        let mut x: Vec<Vec<f64>> = hidden.iter().map(|h| {
            layer_norm(h, &layer.ln1_gamma, &layer.ln1_beta)
        }).collect();
        
        // Multi-head self-attention
        let attn_out = self.attention(layer, &x);
        
        // Residual
        for i in 0..seq_len {
            for j in 0..self.d_model {
                x[i][j] = hidden[i][j] + attn_out[i][j];
            }
        }
        
        // FFN with pre-norm
        let x_norm: Vec<Vec<f64>> = x.iter().map(|h| {
            layer_norm(h, &layer.ln2_gamma, &layer.ln2_beta)
        }).collect();
        
        let ffn_out = self.ffn(layer, &x_norm);
        
        // Residual
        for i in 0..seq_len {
            for j in 0..self.d_model {
                x[i][j] += ffn_out[i][j];
            }
        }
        
        x
    }
    
    fn attention(&self, layer: &TransformerBlock, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let seq_len = x.len();
        let head_dim = self.d_model / layer.n_heads;
        
        // Project Q, K, V
        let q = self.matmul(x, &layer.w_q, self.d_model, self.d_model, layer.b_q.as_ref());
        let k = self.matmul(x, &layer.w_k, self.d_model, self.d_model, layer.b_k.as_ref());
        let v = self.matmul(x, &layer.w_v, self.d_model, self.d_model, layer.b_v.as_ref());
        
        // Multi-head attention
        let mut out = vec![vec![0.0; self.d_model]; seq_len];
        
        for h in 0..layer.n_heads {
            let h_offset = h * head_dim;
            
            // Compute attention scores for this head
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
                        out[i][h_offset + d] += attn[i][j] * v[j][h_offset + d];
                    }
                }
            }
        }
        
        // Output projection
        self.matmul(&out, &layer.w_o, self.d_model, self.d_model, layer.b_o.as_ref())
    }
    
    fn ffn(&self, layer: &TransformerBlock, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // First linear: [d_model, d_ff]
        let mut hidden = self.matmul(x, &layer.w_ff_in, self.d_model, layer.d_ff, layer.b_ff_in.as_ref());
        
        // Activation
        for h in &mut hidden {
            if layer.gelu {
                // Approximate GELU
                for v in h {
                    *v = gelu(*v);
                }
            } else {
                for v in h {
                    *v = v.max(0.0);
                }
            }
        }
        
        // Second linear: [d_ff, d_model]
        self.matmul(&hidden, &layer.w_ff_out, layer.d_ff, self.d_model, layer.b_ff_out.as_ref())
    }
    
    fn matmul(&self, x: &[Vec<f64>], w: &[f64], in_dim: usize, out_dim: usize, bias: Option<&Vec<f64>>) -> Vec<Vec<f64>> {
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
}

fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().sum::<f64>() / n;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
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
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))
}

pub fn load_transformer(path: &std::path::Path) -> Result<Transformer> {
    Transformer::from_json(path)
}