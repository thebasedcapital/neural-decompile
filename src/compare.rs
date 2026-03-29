use crate::quantize::QuantizedRnn;

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
