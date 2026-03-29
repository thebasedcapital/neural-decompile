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
