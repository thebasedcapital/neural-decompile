use crate::quantize::QuantizedRnn;

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
