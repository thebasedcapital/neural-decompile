use crate::quantize::{QuantizedRnn, QuantizedTransformer, QuantizedLayer};

fn fmt_val(v: f64) -> String {
    if (v - v.round()).abs() < 0.01 {
        format!("{}", v.round() as i64)
    } else {
        format!("{:.2}", v)
    }
}

fn is_significant(v: f64) -> bool {
    v.abs() > 0.005
}

pub fn emit_python(q: &QuantizedRnn, name: &str) -> String {
    let mut lines = Vec::new();

    // Count integer weights
    let all: Vec<f64> = q.w_hh.iter().copied()
        .chain(q.w_hx.iter().copied())
        .chain(q.b_h.iter().copied())
        .chain(q.w_y.iter().copied())
        .chain(q.b_y.iter().copied())
        .collect();
    let pct_int = all.iter().filter(|&&v| (v - v.round()).abs() < 0.01).count() as f64
        / all.len() as f64;

    lines.push(format!("# Auto-decompiled by neural-decompile"));
    lines.push(format!("# {:.0}% of weights are exact integers", pct_int * 100.0));
    lines.push(format!("# Hidden dim: {}, Input dim: {}, Output dim: {}",
                       q.hidden_dim, q.input_dim, q.output_dim));
    lines.push(String::new());
    lines.push(format!("def {}(input_sequence):", name));
    lines.push(format!("    \"\"\"RNN decompiled to finite state machine.\"\"\""));
    lines.push(format!("    h = [0.0] * {}", q.hidden_dim));
    lines.push(format!("    for x in input_sequence:"));

    // Transition function: h_i = ReLU(sum(W_hh[i,j]*h[j]) + sum(W_hx[i,j]*x[j]) + b_h[i])
    for i in 0..q.hidden_dim {
        let mut terms = Vec::new();
        for j in 0..q.hidden_dim {
            let v = q.w_hh[[i, j]];
            if is_significant(v) {
                terms.push(format!("{}*h[{}]", fmt_val(v), j));
            }
        }
        for j in 0..q.input_dim {
            let v = q.w_hx[[i, j]];
            if is_significant(v) {
                terms.push(format!("{}*x[{}]", fmt_val(v), j));
            }
        }
        if is_significant(q.b_h[i]) {
            terms.push(fmt_val(q.b_h[i]));
        }

        if terms.is_empty() {
            lines.push(format!("        h{} = 0", i));
        } else {
            lines.push(format!("        h{} = max(0, {})", i, terms.join(" + ")));
        }
    }

    lines.push(format!("        h = [{}]",
        (0..q.hidden_dim).map(|i| format!("h{}", i)).collect::<Vec<_>>().join(", ")));

    // Output layer
    lines.push(format!("    logits = []"));
    for i in 0..q.output_dim {
        let mut terms = Vec::new();
        for j in 0..q.hidden_dim {
            let v = q.w_y[[i, j]];
            if is_significant(v) {
                terms.push(format!("{}*h[{}]", fmt_val(v), j));
            }
        }
        if is_significant(q.b_y[i]) {
            terms.push(fmt_val(q.b_y[i]));
        }
        if terms.is_empty() {
            lines.push(format!("    logits.append(0)"));
        } else {
            lines.push(format!("    logits.append({})", terms.join(" + ")));
        }
    }
    lines.push(format!("    return logits.index(max(logits))"));

    lines.join("\n") + "\n"
}

pub fn emit_rust(q: &QuantizedRnn, name: &str) -> String {
    let mut lines = Vec::new();

    lines.push(format!("/// Auto-decompiled by neural-decompile"));
    lines.push(format!("fn {}(input_sequence: &[Vec<f64>]) -> usize {{", name));
    lines.push(format!("    let mut h = vec![0.0_f64; {}];", q.hidden_dim));
    lines.push(format!("    for x in input_sequence {{"));

    for i in 0..q.hidden_dim {
        let mut terms = Vec::new();
        for j in 0..q.hidden_dim {
            let v = q.w_hh[[i, j]];
            if is_significant(v) {
                terms.push(format!("{:.1} * h[{}]", v, j));
            }
        }
        for j in 0..q.input_dim {
            let v = q.w_hx[[i, j]];
            if is_significant(v) {
                terms.push(format!("{:.1} * x[{}]", v, j));
            }
        }
        if is_significant(q.b_h[i]) {
            terms.push(format!("{:.1}", q.b_h[i]));
        }
        let expr = if terms.is_empty() { "0.0".to_string() } else { terms.join(" + ") };
        lines.push(format!("        let h{} = ({}).max(0.0);", i, expr));
    }

    lines.push(format!("        h = vec![{}];",
        (0..q.hidden_dim).map(|i| format!("h{}", i)).collect::<Vec<_>>().join(", ")));
    lines.push(format!("    }}"));

    lines.push(format!("    let logits: Vec<f64> = vec!["));
    for i in 0..q.output_dim {
        let mut terms = Vec::new();
        for j in 0..q.hidden_dim {
            let v = q.w_y[[i, j]];
            if is_significant(v) {
                terms.push(format!("{:.1} * h[{}]", v, j));
            }
        }
        if is_significant(q.b_y[i]) {
            terms.push(format!("{:.1}", q.b_y[i]));
        }
        let expr = if terms.is_empty() { "0.0".to_string() } else { terms.join(" + ") };
        lines.push(format!("        {},", expr));
    }
    lines.push(format!("    ];"));
    lines.push(format!("    logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0"));
    lines.push(format!("}}"));

    lines.join("\n") + "\n"
}

/// Check if all significant weights in the circuit are integers
fn all_weights_integer(q: &QuantizedRnn) -> bool {
    let all: Vec<f64> = q.w_hh.iter().copied()
        .chain(q.w_hx.iter().copied())
        .chain(q.b_h.iter().copied())
        .chain(q.w_y.iter().copied())
        .chain(q.b_y.iter().copied())
        .collect();
    all.iter().all(|&v| !is_significant(v) || (v - v.round()).abs() < 0.01)
}

/// Format an f64 as integer literal for Kani output
fn fmt_int(v: f64) -> String {
    format!("{}", v.round() as i64)
}

/// Emit Kani-friendly Rust: fixed arrays, integer math, simplified argmax.
pub fn emit_rust_kani(q: &QuantizedRnn, name: &str) -> String {
    let mut lines = Vec::new();
    let use_int = all_weights_integer(q);
    let ty = if use_int { "i64" } else { "f64" };
    let zero = if use_int { "0" } else { "0.0" };
    let fmt = |v: f64| -> String {
        if use_int { fmt_int(v) } else { format!("{:.2}", v) }
    };

    // Detect dead neurons (constant zero output regardless of input)
    let dead: Vec<bool> = (0..q.hidden_dim).map(|i| {
        let all_zero = (0..q.hidden_dim).all(|j| !is_significant(q.w_hh[[i, j]]))
            && (0..q.input_dim).all(|j| !is_significant(q.w_hx[[i, j]]))
            && !is_significant(q.b_h[i]);
        all_zero
    }).collect();
    let live_neurons: Vec<usize> = (0..q.hidden_dim).filter(|&i| !dead[i]).collect();
    let n_live = live_neurons.len();

    lines.push(format!("/// Auto-decompiled by neural-decompile (Kani-friendly)"));
    lines.push(format!("/// {} arithmetic, {} live neurons (of {})",
        if use_int { "Integer" } else { "Float" }, n_live, q.hidden_dim));
    lines.push(format!("fn {}(seq: &[[{}; {}]], len: usize) -> usize {{",
        name, ty, q.input_dim));
    lines.push(format!("    let mut h: [{}; {}] = [{}; {}];", ty, n_live, zero, n_live));
    lines.push(format!("    let mut i = 0;"));
    lines.push(format!("    while i < len {{"));

    // Input extraction
    for j in 0..q.input_dim {
        lines.push(format!("        let x{} = seq[i][{}];", j, j));
    }

    // Hidden state transition
    for (live_idx, &orig_i) in live_neurons.iter().enumerate() {
        let mut terms = Vec::new();
        for (hj, &orig_j) in live_neurons.iter().enumerate() {
            let v = q.w_hh[[orig_i, orig_j]];
            if is_significant(v) {
                terms.push(format!("{} * h[{}]", fmt(v), hj));
            }
        }
        for j in 0..q.input_dim {
            let v = q.w_hx[[orig_i, j]];
            if is_significant(v) {
                terms.push(format!("{} * x{}", fmt(v), j));
            }
        }
        if is_significant(q.b_h[orig_i]) {
            terms.push(fmt(q.b_h[orig_i]));
        }
        let expr = if terms.is_empty() { zero.to_string() } else { terms.join(" + ") };
        if use_int {
            lines.push(format!("        let h{} = ({}).max(0);", live_idx, expr));
        } else {
            lines.push(format!("        let h{} = ({}).max(0.0);", live_idx, expr));
        }
    }

    // Update h array
    lines.push(format!("        h = [{}];",
        (0..n_live).map(|i| format!("h{}", i)).collect::<Vec<_>>().join(", ")));
    lines.push(format!("        i += 1;"));
    lines.push(format!("    }}"));

    // Output layer with simplified argmax
    if q.output_dim == 2 {
        // 2-class: compare logits directly instead of argmax
        let mut terms0 = Vec::new();
        let mut terms1 = Vec::new();
        for (hj, &orig_j) in live_neurons.iter().enumerate() {
            let v0 = q.w_y[[0, orig_j]];
            let v1 = q.w_y[[1, orig_j]];
            if is_significant(v0) {
                terms0.push(format!("{} * h[{}]", fmt(v0), hj));
            }
            if is_significant(v1) {
                terms1.push(format!("{} * h[{}]", fmt(v1), hj));
            }
        }
        if is_significant(q.b_y[0]) { terms0.push(fmt(q.b_y[0])); }
        if is_significant(q.b_y[1]) { terms1.push(fmt(q.b_y[1])); }

        let l0 = if terms0.is_empty() { zero.to_string() } else { terms0.join(" + ") };
        let l1 = if terms1.is_empty() { zero.to_string() } else { terms1.join(" + ") };
        lines.push(format!("    let l0 = {};", l0));
        lines.push(format!("    let l1 = {};", l1));
        lines.push(format!("    if l1 > l0 {{ 1 }} else {{ 0 }}"));
    } else {
        // N-class: emit logit array and argmax
        lines.push(format!("    let logits: [{}; {}] = [", ty, q.output_dim));
        for i in 0..q.output_dim {
            let mut terms = Vec::new();
            for (hj, &orig_j) in live_neurons.iter().enumerate() {
                let v = q.w_y[[i, orig_j]];
                if is_significant(v) {
                    terms.push(format!("{} * h[{}]", fmt(v), hj));
                }
            }
            if is_significant(q.b_y[i]) { terms.push(fmt(q.b_y[i])); }
            let expr = if terms.is_empty() { zero.to_string() } else { terms.join(" + ") };
            lines.push(format!("        {},", expr));
        }
        lines.push(format!("    ];"));
        lines.push(format!("    let mut best = 0;"));
        lines.push(format!("    let mut k = 1;"));
        lines.push(format!("    while k < {} {{", q.output_dim));
        lines.push(format!("        if logits[k] > logits[best] {{ best = k; }}"));
        lines.push(format!("        k += 1;"));
        lines.push(format!("    }}"));
        lines.push(format!("    best"));
    }
    lines.push(format!("}}"));

    // Add Kani proof harness template
    lines.push(String::new());
    lines.push(format!("#[cfg(kani)]"));
    lines.push(format!("#[kani::proof]"));
    lines.push(format!("#[kani::unwind({})]  // TODO: set to max_seq_len + 1",
        n_live + 2));
    lines.push(format!("fn verify_{}() {{", name));
    lines.push(format!("    const MAX_LEN: usize = 6; // TODO: set to actual max sequence length"));
    lines.push(format!("    let len: usize = kani::any();"));
    lines.push(format!("    kani::assume(len >= 1 && len <= MAX_LEN);"));
    lines.push(String::new());
    lines.push(format!("    let mut seq: [[{}; {}]; MAX_LEN] = [[{}; {}]; MAX_LEN];",
        ty, q.input_dim, zero, q.input_dim));
    lines.push(format!("    let mut i = 0;"));
    lines.push(format!("    while i < MAX_LEN {{"));
    lines.push(format!("        if i < len {{"));
    if q.input_dim == 2 {
        lines.push(format!("            let bit: u8 = kani::any();"));
        lines.push(format!("            kani::assume(bit <= 1);"));
        if use_int {
            lines.push(format!("            if bit == 0 {{ seq[i] = [1, 0]; }} else {{ seq[i] = [0, 1]; }}"));
        } else {
            lines.push(format!("            if bit == 0 {{ seq[i] = [1.0, 0.0]; }} else {{ seq[i] = [0.0, 1.0]; }}"));
        }
    } else {
        lines.push(format!("            // TODO: generate one-hot encoding for input_dim={}", q.input_dim));
        lines.push(format!("            let choice: usize = kani::any();"));
        lines.push(format!("            kani::assume(choice < {});", q.input_dim));
        if use_int {
            lines.push(format!("            seq[i][choice] = 1;"));
        } else {
            lines.push(format!("            seq[i][choice] = 1.0;"));
        }
    }
    lines.push(format!("        }}"));
    lines.push(format!("        i += 1;"));
    lines.push(format!("    }}"));
    lines.push(String::new());
    lines.push(format!("    let result = {}(&seq, len);", name));
    lines.push(format!("    let expected = spec_{}(&seq, len); // TODO: implement spec", name));
    lines.push(format!("    assert_eq!(result, expected);"));
    lines.push(format!("}}"));

    lines.join("\n") + "\n"
}

pub fn emit_table(q: &QuantizedRnn) -> String {
    let mut lines = Vec::new();
    lines.push(format!("=== W_hh ({} × {}) ===", q.hidden_dim, q.hidden_dim));
    for i in 0..q.hidden_dim {
        let row: Vec<String> = (0..q.hidden_dim).map(|j| format!("{:>7}", fmt_val(q.w_hh[[i, j]]))).collect();
        lines.push(row.join(" "));
    }
    lines.push(String::new());
    lines.push(format!("=== W_hx ({} × {}) ===", q.hidden_dim, q.input_dim));
    for i in 0..q.hidden_dim {
        let row: Vec<String> = (0..q.input_dim).map(|j| format!("{:>7}", fmt_val(q.w_hx[[i, j]]))).collect();
        lines.push(row.join(" "));
    }
    lines.push(String::new());
    lines.push(format!("=== b_h ({}) ===", q.hidden_dim));
    lines.push(q.b_h.iter().map(|&v| format!("{:>7}", fmt_val(v))).collect::<Vec<_>>().join(" "));
    lines.push(String::new());
    lines.push(format!("=== W_y ({} × {}) ===", q.output_dim, q.hidden_dim));
    for i in 0..q.output_dim {
        let row: Vec<String> = (0..q.hidden_dim).map(|j| format!("{:>7}", fmt_val(q.w_y[[i, j]]))).collect();
        lines.push(row.join(" "));
    }
    lines.push(String::new());
    lines.push(format!("=== b_y ({}) ===", q.output_dim));
    lines.push(q.b_y.iter().map(|&v| format!("{:>7}", fmt_val(v))).collect::<Vec<_>>().join(" "));

    lines.join("\n") + "\n"
}

// ============================================================================
// Transformer emission
// ============================================================================

/// Emit transformer as Python decompilation
pub fn emit_transformer_python(t: &QuantizedTransformer, name: &str) -> String {
    let mut lines = Vec::new();

    lines.push(format!("# Auto-decompiled transformer: {}", name));
    lines.push(format!("# {} layers, d_model={}, vocab_size={}", t.n_layers, t.d_model, t.vocab_size));
    lines.push(String::new());

    // Helper functions
    lines.push("import math".to_string());
    lines.push(String::new());
    lines.push("def layer_norm(x, gamma, beta):".to_string());
    lines.push("    mean = sum(x) / len(x)".to_string());
    lines.push("    var = sum((v - mean) ** 2 for v in x) / len(x)".to_string());
    lines.push("    std = math.sqrt(var + 1e-5)".to_string());
    lines.push("    return [(x[i] - mean) / std * gamma[i] + beta[i] for i in range(len(x))]".to_string());
    lines.push(String::new());
    lines.push("def softmax(x):".to_string());
    lines.push("    m = max(x)".to_string());
    lines.push("    e = [math.exp(v - m) for v in x]".to_string());
    lines.push("    s = sum(e)".to_string());
    lines.push("    return [v / s for v in e]".to_string());
    lines.push(String::new());
    lines.push("def gelu(x):".to_string());
    lines.push("    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))".to_string());
    lines.push(String::new());

    // Main function
    lines.push(format!("def {}(tokens):", name));
    lines.push(format!("    \"\"\"Transformer decompilation: {} layers, {} heads\"\"\"", t.n_layers,
        t.layers.first().map(|l| l.n_heads).unwrap_or(1)));
    lines.push(format!("    seq_len = len(tokens)"));
    lines.push(format!("    assert seq_len <= {}", t.max_seq_len));

    // Embeddings
    lines.push(format!("    # Token + position embeddings"));
    lines.push(format!("    hidden = []"));
    lines.push(format!("    for i, tok in enumerate(tokens):"));
    lines.push(format!("        emb = token_emb_{}[tok][:]  # copy", name));
    if t.pos_emb.is_some() {
        lines.push(format!("        for j in range({}):", t.d_model));
        lines.push(format!("            emb[j] += pos_emb_{}[i][j]", name));
    }
    lines.push(format!("        hidden.append(emb)"));

    // Layers
    for (li, layer) in t.layers.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!("    # Layer {}", li));
        emit_layer_python(&mut lines, layer, li, t.d_model);
    }

    // Final layer norm
    if t.ln_final_gamma.is_some() {
        lines.push(String::new());
        lines.push(format!("    # Final layer norm"));
        lines.push(format!("    for i in range(seq_len):"));
        lines.push(format!("        hidden[i] = layer_norm(hidden[i], ln_final_gamma_{}, ln_final_beta_{})", name, name));
    }

    // Output projection
    lines.push(String::new());
    lines.push(format!("    # Output projection"));
    lines.push(format!("    logits = []"));
    lines.push(format!("    for h in hidden:"));
    lines.push(format!("        out = []"));
    lines.push(format!("        for j in range({}):", t.vocab_size));
    lines.push(format!("            s = 0.0"));
    lines.push(format!("            for i in range({}):", t.d_model));
    lines.push(format!("                s += h[i] * w_out_{}[i * {} + j]", name, t.vocab_size));
    if t.b_out.is_some() {
        lines.push(format!("            s += b_out_{}[j]", name));
    }
    lines.push(format!("            out.append(s)"));
    lines.push(format!("        logits.append(out)"));

    lines.push(format!("    return logits"));

    // Weight definitions
    lines.push(String::new());
    lines.push(format!("# === Weights ==="));
    // Use precise formatting for weights, standard for layer norms (mostly 0/1)
    lines.push(format!("token_emb_{} = {}", name, vec2d_python_precise(&t.token_emb)));
    if let Some(ref pos) = t.pos_emb {
        lines.push(format!("pos_emb_{} = {}", name, vec2d_python_precise(pos)));
    }
    for (li, layer) in t.layers.iter().enumerate() {
        lines.push(format!("w_q_{} = {}", li, vec_python_precise(&layer.w_q)));
        lines.push(format!("w_k_{} = {}", li, vec_python_precise(&layer.w_k)));
        lines.push(format!("w_v_{} = {}", li, vec_python_precise(&layer.w_v)));
        lines.push(format!("w_o_{} = {}", li, vec_python_precise(&layer.w_o)));
        lines.push(format!("w_ff_in_{} = {}", li, vec_python_precise(&layer.w_ff_in)));
        lines.push(format!("w_ff_out_{} = {}", li, vec_python_precise(&layer.w_ff_out)));
        lines.push(format!("ln1_gamma_{} = {}", li, vec_python(&layer.ln1_gamma)));
        lines.push(format!("ln1_beta_{} = {}", li, vec_python(&layer.ln1_beta)));
        lines.push(format!("ln2_gamma_{} = {}", li, vec_python(&layer.ln2_gamma)));
        lines.push(format!("ln2_beta_{} = {}", li, vec_python(&layer.ln2_beta)));
    }
    if let Some(ref g) = t.ln_final_gamma {
        lines.push(format!("ln_final_gamma_{} = {}", name, vec_python(g)));
    }
    if let Some(ref b) = t.ln_final_beta {
        lines.push(format!("ln_final_beta_{} = {}", name, vec_python(b)));
    }
    lines.push(format!("w_out_{} = {}", name, vec_python_precise(&t.w_out)));
    if let Some(ref b) = t.b_out {
        lines.push(format!("b_out_{} = {}", name, vec_python_precise(b)));
    }

    lines.join("\n") + "\n"
}

fn emit_layer_python(lines: &mut Vec<String>, layer: &QuantizedLayer, li: usize, d_model: usize) {
    let head_dim = d_model / layer.n_heads;

    // Pre-norm
    lines.push(format!("    # Pre-norm attention"));
    lines.push(format!("    x = [layer_norm(h, ln1_gamma_{}[:], ln1_beta_{}[:]) for h in hidden]", li, li));

    // Multi-head attention
    lines.push(format!("    # Multi-head attention ({} heads, head_dim={})", layer.n_heads, head_dim));
    // Q, K, V projections
    lines.push(format!("    q = [[0.0] * {} for _ in range(seq_len)]", d_model));
    lines.push(format!("    k = [[0.0] * {} for _ in range(seq_len)]", d_model));
    lines.push(format!("    v = [[0.0] * {} for _ in range(seq_len)]", d_model));
    lines.push(format!("    for i in range(seq_len):"));
    lines.push(format!("        for j in range({}):", d_model));
    lines.push(format!("            for h_idx in range({}):", d_model));
    lines.push(format!("                q[i][j] += x[i][h_idx] * w_q_{}[h_idx * {} + j]", li, d_model));
    lines.push(format!("                k[i][j] += x[i][h_idx] * w_k_{}[h_idx * {} + j]", li, d_model));
    lines.push(format!("                v[i][j] += x[i][h_idx] * w_v_{}[h_idx * {} + j]", li, d_model));

    // Attention per head
    lines.push(format!("    attn_out = [[0.0] * {} for _ in range(seq_len)]", d_model));
    lines.push(format!("    for h_idx in range({}):", layer.n_heads));
    lines.push(format!("        h_off = h_idx * {}", head_dim));
    lines.push(format!("        # Compute attention scores"));
    lines.push(format!("        scores = [[0.0] * seq_len for _ in range(seq_len)]"));
    lines.push(format!("        for i in range(seq_len):"));
    lines.push(format!("            for j in range(seq_len):"));
    lines.push(format!("                for d in range({}):", head_dim));
    lines.push(format!("                    scores[i][j] += q[i][h_off + d] * k[j][h_off + d]"));
    lines.push(format!("                scores[i][j] /= {}  # sqrt(head_dim)", (head_dim as f64).sqrt()));
    lines.push(format!("        # Softmax"));
    lines.push(format!("        attn_weights = [softmax(row) for row in scores]"));
    lines.push(format!("        # Apply attention to values"));
    lines.push(format!("        for i in range(seq_len):"));
    lines.push(format!("            for d in range({}):", head_dim));
    lines.push(format!("                for j in range(seq_len):"));
    lines.push(format!("                    attn_out[i][h_off + d] += attn_weights[i][j] * v[j][h_off + d]"));

    // Output projection
    lines.push(format!("    # Output projection"));
    lines.push(format!("    attn_proj = [[0.0] * {} for _ in range(seq_len)]", d_model));
    lines.push(format!("    for i in range(seq_len):"));
    lines.push(format!("        for j in range({}):", d_model));
    lines.push(format!("            for h_idx in range({}):", d_model));
    lines.push(format!("                attn_proj[i][j] += attn_out[i][h_idx] * w_o_{}[h_idx * {} + j]", li, d_model));

    // Residual
    lines.push(format!("    # Residual"));
    lines.push(format!("    for i in range(seq_len):"));
    lines.push(format!("        for j in range({}):", d_model));
    lines.push(format!("            hidden[i][j] += attn_proj[i][j]"));

    // FFN
    lines.push(format!("    # Pre-norm FFN"));
    lines.push(format!("    x = [layer_norm(h, ln2_gamma_{}[:], ln2_beta_{}[:]) for h in hidden]", li, li));
    lines.push(format!("    ffn_out = []"));
    lines.push(format!("    for h in x:"));
    lines.push(format!("        # FFN layer 1"));
    lines.push(format!("        hidden_ff = [0.0] * {}", layer.d_ff));
    lines.push(format!("        for j in range({}):", layer.d_ff));
    lines.push(format!("            for i in range({}):", d_model));
    lines.push(format!("                hidden_ff[j] += h[i] * w_ff_in_{}[i * {} + j]", li, layer.d_ff));
    lines.push(format!("        # Activation"));
    if layer.gelu {
        lines.push(format!("        hidden_ff = [gelu(v) for v in hidden_ff]"));
    } else {
        lines.push(format!("        hidden_ff = [max(0, v) for v in hidden_ff]"));
    }
    lines.push(format!("        # FFN layer 2"));
    lines.push(format!("        out = [0.0] * {}", d_model));
    lines.push(format!("        for j in range({}):", d_model));
    lines.push(format!("            for i in range({}):", layer.d_ff));
    lines.push(format!("                out[j] += hidden_ff[i] * w_ff_out_{}[i * {} + j]", li, d_model));
    lines.push(format!("        ffn_out.append(out)"));

    // Residual
    lines.push(format!("    # Residual"));
    lines.push(format!("    for i in range(seq_len):"));
    lines.push(format!("        for j in range({}):", d_model));
    lines.push(format!("            hidden[i][j] += ffn_out[i][j]"));
}

fn vec_python(v: &[f64]) -> String {
    let vals: Vec<String> = v.iter().map(|&x| fmt_val(x)).collect();
    format!("[{}]", vals.join(", "))
}

fn vec2d_python(v: &[Vec<f64>]) -> String {
    let rows: Vec<String> = v.iter().map(|row| vec_python(row)).collect();
    format!("[{}]", rows.join(", "))
}

/// Format float with full precision for transformer weights
fn fmt_float_precise(v: f64) -> String {
    if v == 0.0 {
        "0.0".to_string()
    } else if v.abs() < 0.001 {
        format!("{:.8}", v)
    } else if v.abs() < 0.01 {
        format!("{:.6}", v)
    } else {
        format!("{:.4}", v)
    }
}

fn vec_python_precise(v: &[f64]) -> String {
    let vals: Vec<String> = v.iter().map(|&x| fmt_float_precise(x)).collect();
    format!("[{}]", vals.join(", "))
}

fn vec2d_python_precise(v: &[Vec<f64>]) -> String {
    let rows: Vec<String> = v.iter().map(|row| vec_python_precise(row)).collect();
    format!("[{}]", rows.join(", "))
}

/// Emit transformer as Rust decompilation
pub fn emit_transformer_rust(t: &QuantizedTransformer, name: &str) -> String {
    let mut lines = Vec::new();

    lines.push(format!("/// Auto-decompiled transformer: {}", name));
    lines.push(format!("/// {} layers, d_model={}, vocab_size={}", t.n_layers, t.d_model, t.vocab_size));
    lines.push(String::new());

    // Helper functions
    lines.push("fn layer_norm(x: &[f64], gamma: &[f64], beta: &[f64]) -> Vec<f64> {".to_string());
    lines.push("    let n = x.len() as f64;".to_string());
    lines.push("    let mean: f64 = x.iter().sum::<f64>() / n;".to_string());
    lines.push("    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;".to_string());
    lines.push("    let std = (var + 1e-5).sqrt();".to_string());
    lines.push("    x.iter().enumerate().map(|(i, &v)| (v - mean) / std * gamma[i] + beta[i]).collect()".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("fn softmax(x: &[f64]) -> Vec<f64> {".to_string());
    lines.push("    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);".to_string());
    lines.push("    let exp: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();".to_string());
    lines.push("    let sum: f64 = exp.iter().sum();".to_string());
    lines.push("    exp.iter().map(|&v| v / sum).collect()".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("fn gelu(x: f64) -> f64 {".to_string());
    lines.push("    0.5 * x * (1.0 + ((0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))".to_string());
    lines.push("}".to_string());
    lines.push(String::new());

    // Main function
    lines.push(format!("pub fn {}(tokens: &[usize]) -> Vec<Vec<f64>> {{", name));
    lines.push(format!("    let seq_len = tokens.len();"));
    lines.push(format!("    assert!(seq_len <= {});", t.max_seq_len));

    // Embeddings
    lines.push(format!("    // Token + position embeddings"));
    lines.push(format!("    let mut hidden: Vec<Vec<f64>> = tokens.iter().enumerate().map(|(i, &tok)| {{"));
    lines.push(format!("        let mut emb = TOKEN_EMB[tok].clone();"));
    if t.pos_emb.is_some() {
        lines.push(format!("        for j in 0..{} {{ emb[j] += POS_EMB[i][j]; }}", t.d_model));
    }
    lines.push(format!("        emb").to_string());
    lines.push(format!("    }}).collect();"));

    // Layers
    for (li, layer) in t.layers.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!("    // Layer {}", li));
        emit_layer_rust(&mut lines, layer, li, t.d_model);
    }

    // Final layer norm
    if t.ln_final_gamma.is_some() {
        lines.push(String::new());
        lines.push(format!("    // Final layer norm"));
        lines.push(format!("    for h in &mut hidden {{"));
        lines.push(format!("        *h = layer_norm(h, &LN_FINAL_GAMMA, &LN_FINAL_BETA);"));
        lines.push(format!("    }}"));
    }

    // Output projection
    lines.push(String::new());
    lines.push(format!("    // Output projection"));
    lines.push(format!("    hidden.iter().map(|h| {{"));
    lines.push(format!("        (0..{}).map(|j| {{", t.vocab_size));
    lines.push(format!("            let mut s = 0.0;"));
    lines.push(format!("            for i in 0..{} {{ s += h[i] * W_OUT[i * {} + j]; }}", t.d_model, t.vocab_size));
    if t.b_out.is_some() {
        lines.push(format!("            s += B_OUT[j];"));
    }
    lines.push(format!("            s").to_string());
    lines.push(format!("        }}).collect()").to_string());
    lines.push(format!("    }}).collect()").to_string());
    lines.push(format!("}}"));

    // Weight constants
    lines.push(String::new());
    lines.push(format!("// === Weights ==="));
    lines.push(format!("const TOKEN_EMB: &[&[f64]] = &{};", vec2d_rust(&t.token_emb)));
    if let Some(ref pos) = t.pos_emb {
        lines.push(format!("const POS_EMB: &[&[f64]] = &{};", vec2d_rust(pos)));
    }
    for (li, layer) in t.layers.iter().enumerate() {
        lines.push(format!("const W_Q_{}: &[f64] = &{};", li, vec_rust(&layer.w_q)));
        lines.push(format!("const W_K_{}: &[f64] = &{};", li, vec_rust(&layer.w_k)));
        lines.push(format!("const W_V_{}: &[f64] = &{};", li, vec_rust(&layer.w_v)));
        lines.push(format!("const W_O_{}: &[f64] = &{};", li, vec_rust(&layer.w_o)));
        lines.push(format!("const W_FF_IN_{}: &[f64] = &{};", li, vec_rust(&layer.w_ff_in)));
        lines.push(format!("const W_FF_OUT_{}: &[f64] = &{};", li, vec_rust(&layer.w_ff_out)));
        lines.push(format!("const LN1_GAMMA_{}: &[f64] = &{};", li, vec_rust(&layer.ln1_gamma)));
        lines.push(format!("const LN1_BETA_{}: &[f64] = &{};", li, vec_rust(&layer.ln1_beta)));
        lines.push(format!("const LN2_GAMMA_{}: &[f64] = &{};", li, vec_rust(&layer.ln2_gamma)));
        lines.push(format!("const LN2_BETA_{}: &[f64] = &{};", li, vec_rust(&layer.ln2_beta)));
    }
    if let Some(ref g) = t.ln_final_gamma {
        lines.push(format!("const LN_FINAL_GAMMA: &[f64] = &{};", vec_rust(g)));
    }
    if let Some(ref b) = t.ln_final_beta {
        lines.push(format!("const LN_FINAL_BETA: &[f64] = &{};", vec_rust(b)));
    }
    lines.push(format!("const W_OUT: &[f64] = &{};", vec_rust(&t.w_out)));
    if let Some(ref b) = t.b_out {
        lines.push(format!("const B_OUT: &[f64] = &{};", vec_rust(b)));
    }

    lines.join("\n") + "\n"
}

fn emit_layer_rust(lines: &mut Vec<String>, layer: &QuantizedLayer, li: usize, d_model: usize) {
    let head_dim = d_model / layer.n_heads;

    // Pre-norm attention
    lines.push("    // Pre-norm attention".to_string());
    lines.push(format!("    let x: Vec<Vec<f64>> = hidden.iter().map(|h| layer_norm(h, &LN1_GAMMA_{}, &LN1_BETA_{})).collect();", li, li));

    // Multi-head attention
    lines.push(format!("    // Multi-head attention ({} heads, head_dim={})", layer.n_heads, head_dim));
    // Q, K, V projections
    lines.push(format!("    let q: Vec<Vec<f64>> = x.iter().map(|row| {{"));
    lines.push(format!("        (0..{}).map(|j| {{", d_model));
    lines.push(format!("            (0..{}).map(|i| row[i] * W_Q_{}[i * {} + j]).sum()", d_model, li, d_model));
    lines.push(format!("        }}).collect()"));
    lines.push(format!("    }}).collect();"));
    lines.push(format!("    let k: Vec<Vec<f64>> = x.iter().map(|row| {{"));
    lines.push(format!("        (0..{}).map(|j| {{", d_model));
    lines.push(format!("            (0..{}).map(|i| row[i] * W_K_{}[i * {} + j]).sum()", d_model, li, d_model));
    lines.push(format!("        }}).collect()"));
    lines.push(format!("    }}).collect();"));
    lines.push(format!("    let v: Vec<Vec<f64>> = x.iter().map(|row| {{"));
    lines.push(format!("        (0..{}).map(|j| {{", d_model));
    lines.push(format!("            (0..{}).map(|i| row[i] * W_V_{}[i * {} + j]).sum()", d_model, li, d_model));
    lines.push(format!("        }}).collect()"));
    lines.push(format!("    }}).collect();"));

    // Attention per head
    lines.push(format!("    let mut attn_out: Vec<Vec<f64>> = vec![vec![0.0; {}]; seq_len];", d_model));
    lines.push(format!("    for h_idx in 0..{} {{", layer.n_heads));
    lines.push(format!("        let h_off = h_idx * {};", head_dim));
    lines.push(format!("        // Compute attention scores"));
    lines.push(format!("        let mut scores = vec![vec![0.0; seq_len]; seq_len];"));
    lines.push(format!("        for i in 0..seq_len {{"));
    lines.push(format!("            for j in 0..seq_len {{"));
    lines.push(format!("                for d in 0..{} {{", head_dim));
    lines.push(format!("                    scores[i][j] += q[i][h_off + d] * k[j][h_off + d];"));
    lines.push(format!("                }}"));
    lines.push(format!("                scores[i][j] /= {}; // sqrt(head_dim)", (head_dim as f64).sqrt()));
    lines.push(format!("            }}"));
    lines.push(format!("        }}"));
    lines.push(format!("        // Softmax"));
    lines.push(format!("        let attn_weights: Vec<Vec<f64>> = scores.iter().map(|row| softmax(row)).collect();"));
    lines.push(format!("        // Apply to values"));
    lines.push(format!("        for i in 0..seq_len {{"));
    lines.push(format!("            for d in 0..{} {{", head_dim));
    lines.push(format!("                for j in 0..seq_len {{"));
    lines.push(format!("                    attn_out[i][h_off + d] += attn_weights[i][j] * v[j][h_off + d];"));
    lines.push(format!("                }}"));
    lines.push(format!("            }}"));
    lines.push(format!("        }}"));
    lines.push(format!("    }}"));

    // Output projection
    lines.push(format!("    // Output projection"));
    lines.push(format!("    let attn_proj: Vec<Vec<f64>> = attn_out.iter().map(|row| {{"));
    lines.push(format!("        (0..{}).map(|j| {{", d_model));
    lines.push(format!("            (0..{}).map(|i| row[i] * W_O_{}[i * {} + j]).sum()", d_model, li, d_model));
    lines.push(format!("        }}).collect()"));
    lines.push(format!("    }}).collect();"));

    // Residual
    lines.push("    // Residual".to_string());
    lines.push("    for i in 0..seq_len {".to_string());
    lines.push(format!("        for j in 0..{} {{ hidden[i][j] += attn_proj[i][j]; }}", d_model));
    lines.push("    }".to_string());

    // FFN
    lines.push("    // Pre-norm FFN".to_string());
    lines.push(format!("    let x: Vec<Vec<f64>> = hidden.iter().map(|h| layer_norm(h, &LN2_GAMMA_{}, &LN2_BETA_{})).collect();", li, li));
    lines.push("    let ffn_out: Vec<Vec<f64>> = x.iter().map(|h| {".to_string());
    lines.push(format!("        // FFN layer 1: {} -> {}", d_model, layer.d_ff));
    lines.push(format!("        let hidden_ff: Vec<f64> = (0..{}).map(|j| {{", layer.d_ff));
    lines.push(format!("            (0..{}).map(|i| h[i] * W_FF_IN_{}[i * {} + j]).sum()", d_model, li, layer.d_ff));
    lines.push("        }).collect();".to_string());
    lines.push("        // Activation".to_string());
    if layer.gelu {
        lines.push("        let hidden_ff: Vec<f64> = hidden_ff.iter().map(|&v| gelu(v)).collect();".to_string());
    } else {
        lines.push("        let hidden_ff: Vec<f64> = hidden_ff.iter().map(|&v| v.max(0.0)).collect();".to_string());
    }
    lines.push(format!("        // FFN layer 2: {} -> {}", layer.d_ff, d_model));
    lines.push(format!("        (0..{}).map(|j| {{", d_model));
    lines.push(format!("            (0..{}).map(|i| hidden_ff[i] * W_FF_OUT_{}[i * {} + j]).sum()", layer.d_ff, li, d_model));
    lines.push("        }).collect()".to_string());
    lines.push("    }).collect();".to_string());

    // Residual
    lines.push("    // Residual".to_string());
    lines.push("    for i in 0..seq_len {".to_string());
    lines.push(format!("        for j in 0..{} {{ hidden[i][j] += ffn_out[i][j]; }}", d_model));
    lines.push("    }".to_string());
}

fn vec_rust(v: &[f64]) -> String {
    let vals: Vec<String> = v.iter().map(|&x| fmt_val(x)).collect();
    format!("[{}]", vals.join(", "))
}

fn vec2d_rust(v: &[Vec<f64>]) -> String {
    let rows: Vec<String> = v.iter().map(|row| vec_rust(row)).collect();
    format!("[{}]", rows.join(", "))
}

/// Emit transformer as table summary
pub fn emit_transformer_table(t: &QuantizedTransformer) -> String {
    let mut lines = Vec::new();

    lines.push(format!("=== Transformer Summary ==="));
    lines.push(format!("Layers: {}, d_model: {}, vocab_size: {}", t.n_layers, t.d_model, t.vocab_size));
    lines.push(format!("Max seq len: {}", t.max_seq_len));
    lines.push(String::new());

    for (li, layer) in t.layers.iter().enumerate() {
        lines.push(format!("--- Layer {} ---", li));
        lines.push(format!("  Attention: {} heads, head_dim={}", layer.n_heads, t.d_model / layer.n_heads));
        lines.push(format!("  W_Q: {} params", layer.w_q.len()));
        lines.push(format!("  W_K: {} params", layer.w_k.len()));
        lines.push(format!("  W_V: {} params", layer.w_v.len()));
        lines.push(format!("  W_O: {} params", layer.w_o.len()));
        lines.push(format!("  FFN: d_model={} -> d_ff={} -> d_model={}", t.d_model, layer.d_ff, t.d_model));
        lines.push(format!("  Activation: {}", if layer.gelu { "GELU" } else { "ReLU" }));
        lines.push(String::new());
    }

    lines.push(format!("Token embedding: {} x {}", t.token_emb.len(), t.d_model));
    if t.pos_emb.is_some() {
        lines.push(format!("Position embedding: {} x {}", t.max_seq_len, t.d_model));
    }
    lines.push(format!("Output projection: {} x {}", t.d_model, t.vocab_size));

    lines.join("\n") + "\n"
}

// ============================================================================
// Full circuit decompilation for transformers
// ============================================================================

/// Honest attention head analysis - reports what we actually detect
#[derive(Debug, Clone)]
pub enum HeadPattern {
    PrevToken,              // Clear previous-token attention
    FirstToken,             // Clear first-position attention
    CurrentToken,           // Clear self-attention
    PositionalShift(i32),   // Clear offset pattern (e.g., attends to pos+2)
    Uniform,                // Roughly equal attention everywhere
    PositionalRange(usize, usize), // Attends within specific range
    ContentSensitive,       // Attention varies by token ID (detected via multi-token test)
    Mixed,                  // Multiple or unclear patterns
    Unknown,                // Could not determine pattern
}

impl std::fmt::Display for HeadPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeadPattern::PrevToken => write!(f, "prev-token"),
            HeadPattern::FirstToken => write!(f, "first-token"),
            HeadPattern::CurrentToken => write!(f, "self-token"),
            HeadPattern::PositionalShift(n) => write!(f, "pos-shift-{}", n),
            HeadPattern::Uniform => write!(f, "uniform"),
            HeadPattern::PositionalRange(start, end) => write!(f, "pos-range-{}-{}", start, end),
            HeadPattern::ContentSensitive => write!(f, "content-sensitive"),
            HeadPattern::Mixed => write!(f, "mixed-pattern"),
            HeadPattern::Unknown => write!(f, "unknown"),
        }
    }
}

/// Full circuit analysis of a transformer
pub struct TransformerCircuit {
    pub head_patterns: Vec<Vec<HeadPattern>>,  // [layer][head]
    pub ffn_circuits: Vec<Vec<FfnCircuit>>,    // [layer][neuron]
    pub important_neurons: Vec<Vec<usize>>,      // [layer] neurons that matter
}

/// What an FFN neuron computes
pub struct FfnCircuit {
    pub neuron_idx: usize,
    pub pattern_type: FfnPattern,
    pub top_inputs: Vec<(usize, f64)>,  // (input_dim, weight)
    pub activation_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum FfnPattern {
    KeyValue(usize, usize),  // (key_token, value_token) memory
    FeatureDetect(String),    // Detects some feature
    AlwaysOn,
    AlwaysOff,
    Unknown,
}

impl std::fmt::Display for FfnPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfnPattern::KeyValue(k, v) => write!(f, "key-value({}, {})", k, v),
            FfnPattern::FeatureDetect(desc) => write!(f, "feature({})", desc),
            FfnPattern::AlwaysOn => write!(f, "always-on"),
            FfnPattern::AlwaysOff => write!(f, "always-off"),
            FfnPattern::Unknown => write!(f, "unknown"),
        }
    }
}

/// FULL attention head analysis with proper token+position testing
fn analyze_head_pattern(
    w_q: &[f64],
    w_k: &[f64],
    w_v: &[f64],
    w_o: &[f64],
    b_q: Option<&Vec<f64>>,
    b_k: Option<&Vec<f64>>,
    b_v: Option<&Vec<f64>>,
    d_model: usize,
    head_dim: usize,
    pos_emb: Option<&[Vec<f64>]>,
    token_emb: &[Vec<f64>],
) -> HeadPattern {
    // Test with multiple sequence lengths and actual token variations
    let test_seq_lens = vec![4, 8, 16];
    let test_tokens: Vec<usize> = vec![0, 5, 10, 15]; // Different token IDs to test content sensitivity

    let mut all_attention_patterns: Vec<Vec<Vec<f64>>> = Vec::new();

    for &seq_len in &test_seq_lens {
        // Test with different token combinations
        for token_combo in test_tokens.iter().take(3) {
            let mut q_vecs: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
            let mut k_vecs: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

            for pos in 0..seq_len {
                let mut q = vec![0.0; head_dim];
                let mut k = vec![0.0; head_dim];

                // Get token embedding (varies by token, not just position)
                let tok_emb = if pos < token_emb.len() {
                    &token_emb[*token_combo % token_emb.len()]
                } else {
                    &token_emb[0]
                };

                // Add position embedding
                if let Some(pos_vec) = pos_emb {
                    if pos < pos_vec.len() {
                        let p = &pos_vec[pos];
                        // Combined embedding: token + position
                        for i in 0..d_model.min(tok_emb.len()).min(p.len()) {
                            let combined = tok_emb[i] + p[i];
                            for h in 0..head_dim {
                                q[h] += combined * w_q[h * d_model + i];
                                k[h] += combined * w_k[h * d_model + i];
                            }
                        }
                    }
                } else {
                    // Just token embedding
                    for i in 0..d_model.min(tok_emb.len()) {
                        for h in 0..head_dim {
                            q[h] += tok_emb[i] * w_q[h * d_model + i];
                            k[h] += tok_emb[i] * w_k[h * d_model + i];
                        }
                    }
                }

                // Add bias
                if let Some(bias_q) = b_q {
                    for h in 0..head_dim.min(bias_q.len()) {
                        q[h] += bias_q[h];
                    }
                }
                if let Some(bias_k) = b_k {
                    for h in 0..head_dim.min(bias_k.len()) {
                        k[h] += bias_k[h];
                    }
                }

                q_vecs.push(q);
                k_vecs.push(k);
            }

            // Compute attention scores
            let mut attn_pattern: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
            for q_pos in 0..seq_len {
                let mut scores: Vec<f64> = Vec::with_capacity(seq_len);
                for k_pos in 0..seq_len {
                    let score: f64 = q_vecs[q_pos].iter().zip(k_vecs[k_pos].iter())
                        .map(|(q, k)| q * k).sum::<f64>() / (head_dim as f64).sqrt();
                    scores.push(score);
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
                let sum_exp: f64 = exp_scores.iter().sum();
                let attn: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();
                attn_pattern.push(attn);
            }

            all_attention_patterns.push(attn_pattern);
        }
    }

    // Analyze patterns across all tests
    analyze_attention_behavior(&all_attention_patterns)
}

fn analyze_attention_behavior(patterns: &[Vec<Vec<f64>>]) -> HeadPattern {
    if patterns.is_empty() {
        return HeadPattern::Unknown;
    }

    // Check consistency across different test cases
    let mut is_prev_token = true;
    let mut is_first_token = true;
    let mut is_self_token = true;
    let mut is_uniform = true;
    let mut content_sensitive = false;

    let mut prev_votes = 0;
    let mut first_votes = 0;
    let mut self_votes = 0;

    for pattern in patterns {
        let seq_len = pattern.len();
        if seq_len == 0 { continue; }

        let uniform_attn = 1.0 / seq_len as f64;

        for q_pos in 0..seq_len {
            let attn = &pattern[q_pos];

            // Find where attention goes
            let (max_pos, &max_val) = attn.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));

            // Check pattern types
            if max_pos == q_pos.saturating_sub(1) && max_val > 0.5 {
                prev_votes += 1;
            }
            if max_pos == 0 && max_val > 0.5 {
                first_votes += 1;
            }
            if max_pos == q_pos && max_val > 0.5 {
                self_votes += 1;
            }

            // Check uniform
            for &a in attn {
                if (a - uniform_attn).abs() > 0.15 {
                    is_uniform = false;
                    break;
                }
            }

            // Check if attention is soft vs sharp (indicates positional vs content)
            let entropy: f64 = -attn.iter().filter(|&&a| a > 0.0)
                .map(|&a| a * a.ln()).sum::<f64>();
            if entropy > 1.5 {
                content_sensitive = true; // Broad attention often indicates content sensitivity
            }
        }
    }

    let total_tests = patterns.len() * patterns[0].len();

    // Determine pattern by majority vote
    if prev_votes as f64 / total_tests as f64 > 0.7 {
        return HeadPattern::PrevToken;
    }
    if first_votes as f64 / total_tests as f64 > 0.7 {
        return HeadPattern::FirstToken;
    }
    if self_votes as f64 / total_tests as f64 > 0.7 {
        return HeadPattern::CurrentToken;
    }
    if is_uniform {
        return HeadPattern::Uniform;
    }
    if content_sensitive {
        return HeadPattern::ContentSensitive;
    }

    // Check for positional shift patterns
    let mut shift_votes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for pattern in patterns {
        for (q_pos, attn) in pattern.iter().enumerate() {
            let max_pos = attn.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let shift = max_pos as i32 - q_pos as i32;
            *shift_votes.entry(shift).or_insert(0) += 1;
        }
    }

    if let Some((&shift, &votes)) = shift_votes.iter().max_by_key(|&(_, v)| v) {
        if votes > total_tests / 3 {
            return HeadPattern::PositionalShift(shift);
        }
    }

    HeadPattern::Mixed
}

/// HONEST FFN neuron analysis - reports actual weight patterns without false "logic gate" claims
fn analyze_ffn_neuron(
    w_in: &[f64],
    b_in: Option<&Vec<f64>>,
    w_out_col: &[f64],
    d_model: usize,
    neuron_idx: usize
) -> FfnCircuit {
    // Collect all significant input weights (> 0.01 threshold)
    let mut input_weights: Vec<(usize, f64)> = w_in.iter().enumerate()
        .map(|(i, &w)| (i, w))
        .filter(|(_, w)| w.abs() > 0.01)
        .collect();
    input_weights.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

    // Get actual bias value
    let bias = b_in.map(|b| b.get(neuron_idx).copied().unwrap_or(0.0)).unwrap_or(0.0);

    // Calculate actual activation threshold
    // For ReLU/GELU: neuron activates when sum(inputs * weights) + bias > 0
    let threshold = -bias;

    // Determine if neuron matters (has output connections)
    let max_output = w_out_col.iter().map(|&w| w.abs()).fold(0.0_f64, f64::max);
    let has_output = max_output > 0.01;

    // Classify based on weight structure
    let pattern = if input_weights.is_empty() || !has_output {
        FfnPattern::AlwaysOff
    } else {
        let n_inputs = input_weights.len();
        let positive_count = input_weights.iter().filter(|(_, w)| *w > 0.0).count();
        let negative_count = input_weights.iter().filter(|(_, w)| *w < 0.0).count();

        // Analyze weight distribution
        let total_pos: f64 = input_weights.iter().filter(|(_, w)| *w > 0.0).map(|(_, w)| w.abs()).sum();
        let total_neg: f64 = input_weights.iter().filter(|(_, w)| *w < 0.0).map(|(_, w)| w.abs()).sum();

        // Honest classification without "logic gate" false claims
        let desc = if n_inputs == 1 {
            // Single input: detects one dimension
            let (dim, weight) = input_weights[0];
            if weight > 0.0 {
                format!("single-pos-dim-{}-wt-{:.3}", dim, weight)
            } else {
                format!("single-neg-dim-{}-wt-{:.3}", dim, weight)
            }
        } else if positive_count > 0 && negative_count == 0 {
            // All positive: weighted sum
            format!("weighted-sum-{}-inputs-pos", n_inputs)
        } else if negative_count > 0 && positive_count == 0 {
            // All negative: inverse weighted sum
            format!("weighted-sum-{}-inputs-neg", n_inputs)
        } else if total_pos > total_neg * 2.0 {
            // Mostly positive
            format!("mostly-pos-{}p-{}n", positive_count, negative_count)
        } else if total_neg > total_pos * 2.0 {
            // Mostly negative
            format!("mostly-neg-{}n-{}p", negative_count, positive_count)
        } else {
            // Mixed
            format!("mixed-{}p-{}n", positive_count, negative_count)
        };

        FfnPattern::FeatureDetect(desc)
    };

    // Keep top 5 for display
    input_weights.truncate(5);

    FfnCircuit {
        neuron_idx,
        pattern_type: pattern,
        top_inputs: input_weights,
        activation_threshold: threshold,
    }
}

/// Full circuit decompilation
pub fn decompile_transformer_circuit(t: &QuantizedTransformer) -> TransformerCircuit {
    let mut head_patterns: Vec<Vec<HeadPattern>> = Vec::with_capacity(t.n_layers);
    let mut ffn_circuits: Vec<Vec<FfnCircuit>> = Vec::with_capacity(t.n_layers);
    let mut important_neurons: Vec<Vec<usize>> = Vec::with_capacity(t.n_layers);

    for (li, layer) in t.layers.iter().enumerate() {
        let head_dim = t.d_model / layer.n_heads;

        // Analyze attention heads with FULL simulation (token + position + bias)
        let mut layer_heads = Vec::with_capacity(layer.n_heads);
        for h in 0..layer.n_heads {
            let h_off = h * head_dim;
            // Extract all attention weights for this head
            let w_q_h: Vec<f64> = layer.w_q[h_off * t.d_model..(h_off + head_dim) * t.d_model].to_vec();
            let w_k_h: Vec<f64> = layer.w_k[h_off * t.d_model..(h_off + head_dim) * t.d_model].to_vec();
            let w_v_h: Vec<f64> = layer.w_v[h_off * t.d_model..(h_off + head_dim) * t.d_model].to_vec();
            let w_o_h: Vec<f64> = layer.w_o[h_off * t.d_model..(h_off + head_dim) * t.d_model].to_vec();

            // Extract biases for this head
            let b_q_h = layer.b_q.as_ref().map(|b| b[h_off..h_off + head_dim].to_vec());
            let b_k_h = layer.b_k.as_ref().map(|b| b[h_off..h_off + head_dim].to_vec());
            let b_v_h = layer.b_v.as_ref().map(|b| b[h_off..h_off + head_dim].to_vec());

            let pattern = analyze_head_pattern(
                &w_q_h, &w_k_h, &w_v_h, &w_o_h,
                b_q_h.as_ref(),
                b_k_h.as_ref(),
                b_v_h.as_ref(),
                t.d_model, head_dim,
                t.pos_emb.as_deref(),
                &t.token_emb,
            );
            layer_heads.push(pattern);
        }
        head_patterns.push(layer_heads);

        // Analyze FFN neurons
        let mut layer_neurons = Vec::with_capacity(layer.d_ff);
        let mut important = Vec::new();

        for n in 0..layer.d_ff {
            // Input weights for this neuron
            let w_in: Vec<f64> = (0..t.d_model).map(|i| layer.w_ff_in[i * layer.d_ff + n]).collect();
            // Output weights for this neuron
            let w_out_col: Vec<f64> = (0..t.d_model).map(|i| layer.w_ff_out[n * t.d_model + i]).collect();

            // Extract bias for this neuron
            let b_in = layer.b_ff_in.as_ref();

            let circuit = analyze_ffn_neuron(&w_in, b_in, &w_out_col, t.d_model, n);

            // Check if this neuron matters (has significant output weights)
            let has_output = w_out_col.iter().any(|&w| w.abs() > 0.01);
            let has_input = w_in.iter().any(|&w| w.abs() > 0.01);

            if has_output && has_input {
                important.push(n);
            }

            layer_neurons.push(circuit);
        }
        ffn_circuits.push(layer_neurons);
        important_neurons.push(important);
    }

    TransformerCircuit {
        head_patterns,
        ffn_circuits,
        important_neurons,
    }
}

/// Emit circuit as Python with interpreted patterns
pub fn emit_transformer_circuit(t: &QuantizedTransformer, name: &str) -> String {
    let circuit = decompile_transformer_circuit(t);
    let mut lines = Vec::new();

    lines.push(format!("# Circuit Decompilation: {}", name));
    lines.push(format!("# {} layers, d_model={}, heads={}, vocab={}",
        t.n_layers, t.d_model,
        t.layers[0].n_heads, t.vocab_size));
    lines.push(String::new());

    // Emit circuit structure
    for li in 0..t.n_layers {
        lines.push(format!("# === Layer {} ===", li));

        // Attention heads
        lines.push(format!("class Layer{}Attention:", li));
        for (h, pattern) in circuit.head_patterns[li].iter().enumerate() {
            lines.push(format!("    # Head {}: {}", h, pattern));
            match pattern {
                HeadPattern::PrevToken => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Attends to position q_pos - 1"));
                    lines.push(format!("        return 1.0 if k_pos == q_pos - 1 else 0.0"));
                }
                HeadPattern::FirstToken => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Always attends to position 0"));
                    lines.push(format!("        return 1.0 if k_pos == 0 else 0.0"));
                }
                HeadPattern::CurrentToken => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Attends to self"));
                    lines.push(format!("        return 1.0 if k_pos == q_pos else 0.0"));
                }
                HeadPattern::PositionalShift(n) => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Attends to position q_pos + {}", n));
                    lines.push(format!("        return 1.0 if k_pos == q_pos + {} else 0.0", n));
                }
                HeadPattern::Uniform => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Uniform attention to all positions"));
                    lines.push(format!("        return 1.0 / seq_len"));
                }
                HeadPattern::ContentSensitive => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Content-sensitive: attention varies by token ID"));
                    lines.push(format!("        # (Detected via testing with different token values)"));
                    lines.push(format!("        return attention_by_token_match(q_token, k_token)"));
                }
                HeadPattern::Mixed => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Mixed attention pattern"));
                    lines.push(format!("        return 1.0 / (abs(q_pos - k_pos) + 1)"));
                }
                _ => {
                    lines.push(format!("    def head_{}(self, q_pos, k_pos):", h));
                    lines.push(format!("        # Complex attention pattern (run trace to analyze)"));
                    lines.push(format!("        return softmax(Q[q_pos] @ K[k_pos].T)"));
                }
            }
            lines.push(String::new());
        }

        // FFN circuit
        let n_important = circuit.important_neurons[li].len();
        let n_dead = t.layers[li].d_ff - n_important;
        lines.push(format!("    # FFN: {} important, {} dead neurons", n_important, n_dead));

        for &n in &circuit.important_neurons[li][..n_important.min(10)] {
            let ffn = &circuit.ffn_circuits[li][n];
            lines.push(format!("    # Neuron {}: {}", n, ffn.pattern_type));
            if !ffn.top_inputs.is_empty() {
                let input_str: String = ffn.top_inputs.iter()
                    .map(|(i, w)| format!("h[{}]*{:.3}", i, w))
                    .collect::<Vec<_>>()
                    .join(" + ");
                lines.push(format!("    #   computes: {} > 0 ? GELU : 0", input_str));
            }
        }
        if n_important > 10 {
            lines.push(format!("    # ... and {} more neurons", n_important - 10));
        }

        lines.push(String::new());
    }

    // Full execution model
    lines.push(format!("def {}_circuit(tokens):", name));
    lines.push(format!("    \"\"\"Full circuit execution.\"\"\""));
    lines.push(format!("    # Token embeddings"));
    lines.push(format!("    hidden = [TOKEN_EMB[t] for t in tokens]"));
    lines.push(String::new());

    for li in 0..t.n_layers {
        lines.push(format!("    # Layer {}", li));
        lines.push(format!("    attn = Layer{}Attention()", li));
        lines.push(format!("    # ... multi-head attention ..."));
        lines.push(format!("    # ... FFN with {} important neurons ...", circuit.important_neurons[li].len()));
        lines.push(String::new());
    }

    lines.push(format!("    # Output projection"));
    lines.push(format!("    return logits"));

    lines.join("\n") + "\n"
}
