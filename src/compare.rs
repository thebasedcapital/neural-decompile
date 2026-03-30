use crate::quantize::{QuantizedRnn, QuantizedTransformer};

/// Comparison between two decompiled networks
#[derive(Debug)]
pub struct CompareResult {
    pub hidden_match: bool,
    pub hidden_dim_a: usize,
    pub hidden_dim_b: usize,
    /// Per-matrix similarity (cosine similarity of flattened weights)
    pub w_hh_sim: f64,
    pub w_hx_sim: f64,
    pub b_h_sim: f64,
    pub w_y_sim: f64,
    pub b_y_sim: f64,
    /// Is the output layer approximately negated? (complement detection)
    pub output_negated: bool,
    pub w_y_neg_sim: f64,
    pub b_y_neg_sim: f64,
    /// Overall relationship
    pub relationship: Relationship,
}

#[derive(Debug)]
pub enum Relationship {
    /// Same circuit, same output
    Identical,
    /// Same transition function, negated output (DFA complements)
    Complement,
    /// Similar transition function, different output
    SimilarTransition,
    /// Different circuits
    Different,
}

impl std::fmt::Display for Relationship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Identical => write!(f, "IDENTICAL — same circuit"),
            Self::Complement => write!(f, "COMPLEMENT — same transition, negated output (L̄)"),
            Self::SimilarTransition => write!(f, "SIMILAR — shared transition, different readout"),
            Self::Different => write!(f, "DIFFERENT — distinct circuits"),
        }
    }
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn flatten_array(arr: &ndarray::Array2<f64>) -> Vec<f64> {
    arr.iter().copied().collect()
}

pub fn compare(a: &QuantizedRnn, b: &QuantizedRnn) -> CompareResult {
    let hidden_match = a.hidden_dim == b.hidden_dim && a.input_dim == b.input_dim;

    // Compute similarities (only meaningful if dimensions match)
    let (w_hh_sim, w_hx_sim, b_h_sim, w_y_sim, b_y_sim) = if hidden_match {
        (
            cosine_sim(&flatten_array(&a.w_hh), &flatten_array(&b.w_hh)),
            cosine_sim(&flatten_array(&a.w_hx), &flatten_array(&b.w_hx)),
            cosine_sim(&a.b_h, &b.b_h),
            cosine_sim(&flatten_array(&a.w_y), &flatten_array(&b.w_y)),
            cosine_sim(&a.b_y, &b.b_y),
        )
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    // Check if output is negated (complement detection)
    let (w_y_neg_sim, b_y_neg_sim) = if hidden_match && a.output_dim == b.output_dim {
        let neg_w_y: Vec<f64> = flatten_array(&b.w_y).iter().map(|v| -v).collect();
        let neg_b_y: Vec<f64> = b.b_y.iter().map(|v| -v).collect();
        (
            cosine_sim(&flatten_array(&a.w_y), &neg_w_y),
            cosine_sim(&a.b_y, &neg_b_y),
        )
    } else {
        (0.0, 0.0)
    };

    let output_negated = w_y_neg_sim > 0.95 && b_y_neg_sim > 0.95;

    // Determine relationship
    let transition_sim = if hidden_match {
        (w_hh_sim + w_hx_sim + b_h_sim) / 3.0
    } else {
        0.0
    };

    let relationship = if !hidden_match {
        Relationship::Different
    } else if transition_sim > 0.99 && w_y_sim > 0.99 && b_y_sim > 0.99 {
        Relationship::Identical
    } else if transition_sim > 0.95 && output_negated {
        Relationship::Complement
    } else if transition_sim > 0.85 {
        Relationship::SimilarTransition
    } else {
        Relationship::Different
    };

    CompareResult {
        hidden_match,
        hidden_dim_a: a.hidden_dim,
        hidden_dim_b: b.hidden_dim,
        w_hh_sim,
        w_hx_sim,
        b_h_sim,
        w_y_sim,
        b_y_sim,
        output_negated,
        w_y_neg_sim,
        b_y_neg_sim,
        relationship,
    }
}

pub fn format_compare(result: &CompareResult) -> String {
    let mut out = String::new();

    out.push_str(&format!("═══ CIRCUIT COMPARISON ═══\n"));
    out.push_str(&format!("Relationship: {}\n\n", result.relationship));

    out.push_str(&format!("Dimensions: A=hd{} B=hd{} {}\n\n",
        result.hidden_dim_a, result.hidden_dim_b,
        if result.hidden_match { "✓ match" } else { "✗ mismatch" }));

    if result.hidden_match {
        out.push_str("Cosine Similarity (transition function):\n");
        out.push_str(&format!("  W_hh: {:.4}\n", result.w_hh_sim));
        out.push_str(&format!("  W_hx: {:.4}\n", result.w_hx_sim));
        out.push_str(&format!("  b_h:  {:.4}\n", result.b_h_sim));
        out.push_str(&format!("  avg:  {:.4}\n\n", (result.w_hh_sim + result.w_hx_sim + result.b_h_sim) / 3.0));

        out.push_str("Cosine Similarity (output layer):\n");
        out.push_str(&format!("  W_y:  {:.4}  (negated: {:.4})\n", result.w_y_sim, result.w_y_neg_sim));
        out.push_str(&format!("  b_y:  {:.4}  (negated: {:.4})\n", result.b_y_sim, result.b_y_neg_sim));

        if result.output_negated {
            out.push_str("\n  → Output weights are NEGATED — these circuits compute complement languages.\n");
            out.push_str("    Same DFA, inverted accept/reject. L(A) = Σ* \\ L(B)\n");
        }
    }

    out
}

// ============================================================================
// Transformer comparison
// ============================================================================

/// Comparison between two transformers
#[derive(Debug)]
pub struct TransformerCompareResult {
    pub layers_match: bool,
    pub d_model_match: bool,
    pub vocab_match: bool,
    /// Per-layer cosine similarities [layer][matrix]
    pub layer_sims: Vec<TransformerLayerSims>,
    pub overall_similarity: f64,
    pub relationship: TransformerRelationship,
}

#[derive(Debug)]
pub struct TransformerLayerSims {
    pub w_q: f64, pub w_k: f64, pub w_v: f64, pub w_o: f64,
    pub w_ff_in: f64, pub w_ff_out: f64,
    pub ln1: f64, pub ln2: f64,
}

#[derive(Debug)]
pub enum TransformerRelationship {
    Identical,
    Similar,
    DifferentArchitecture,
    Different,
}

impl std::fmt::Display for TransformerRelationship {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Identical => write!(f, "IDENTICAL — same architecture, same weights"),
            Self::Similar => write!(f, "SIMILAR — compatible architecture, related weights"),
            Self::DifferentArchitecture => write!(f, "DIFFERENT ARCH — layers/d_model/vocab mismatch"),
            Self::Different => write!(f, "DIFFERENT — distinct transformers"),
        }
    }
}

fn vec_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
    dot / (norm_a * norm_b)
}

pub fn compare_transformers(a: &QuantizedTransformer, b: &QuantizedTransformer) -> TransformerCompareResult {
    let layers_match = a.n_layers == b.n_layers;
    let d_model_match = a.d_model == b.d_model;
    let vocab_match = a.vocab_size == b.vocab_size;

    let mut layer_sims = Vec::new();

    if layers_match && d_model_match {
        for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
            let ln1_sim = vec_sim(&la.ln1_gamma, &lb.ln1_gamma) * 0.5 + vec_sim(&la.ln1_beta, &lb.ln1_beta) * 0.5;
            let ln2_sim = vec_sim(&la.ln2_gamma, &lb.ln2_gamma) * 0.5 + vec_sim(&la.ln2_beta, &lb.ln2_beta) * 0.5;

            layer_sims.push(TransformerLayerSims {
                w_q: vec_sim(&la.w_q, &lb.w_q),
                w_k: vec_sim(&la.w_k, &lb.w_k),
                w_v: vec_sim(&la.w_v, &lb.w_v),
                w_o: vec_sim(&la.w_o, &lb.w_o),
                w_ff_in: vec_sim(&la.w_ff_in, &lb.w_ff_in),
                w_ff_out: vec_sim(&la.w_ff_out, &lb.w_ff_out),
                ln1: ln1_sim,
                ln2: ln2_sim,
            });
        }
    }

    // Overall similarity
    let overall = if !layer_sims.is_empty() {
        let total: f64 = layer_sims.iter().map(|ls| {
            (ls.w_q + ls.w_k + ls.w_v + ls.w_o + ls.w_ff_in + ls.w_ff_out) / 6.0
        }).sum();
        total / layer_sims.len() as f64
    } else { 0.0 };

    let relationship = if !layers_match || !d_model_match || !vocab_match {
        TransformerRelationship::DifferentArchitecture
    } else if overall > 0.99 {
        TransformerRelationship::Identical
    } else if overall > 0.8 {
        TransformerRelationship::Similar
    } else {
        TransformerRelationship::Different
    };

    TransformerCompareResult {
        layers_match, d_model_match, vocab_match,
        layer_sims, overall_similarity: overall, relationship,
    }
}

pub fn format_transformer_compare(result: &TransformerCompareResult) -> String {
    let mut out = String::new();
    out.push_str("═══ TRANSFORMER COMPARISON ═══\n");
    out.push_str(&format!("Relationship: {}\n\n", result.relationship));

    out.push_str(&format!("Architecture: layers={} d_model={} vocab={}\n",
        if result.layers_match { "✓" } else { "✗" },
        if result.d_model_match { "✓" } else { "✗" },
        if result.vocab_match { "✓" } else { "✗" }));

    if !result.layer_sims.is_empty() {
        out.push_str("\nPer-Layer Cosine Similarity:\n");
        for (i, ls) in result.layer_sims.iter().enumerate() {
            let attn = (ls.w_q + ls.w_k + ls.w_v + ls.w_o) / 4.0;
            let ffn = (ls.w_ff_in + ls.w_ff_out) / 2.0;
            out.push_str(&format!("  L{}: attn={:.3} ffn={:.3} ln={:.3}\n",
                i, attn, ffn, (ls.ln1 + ls.ln2) / 2.0));
        }
        out.push_str(&format!("\nOverall similarity: {:.3}\n", result.overall_similarity));
    }

    out
}
