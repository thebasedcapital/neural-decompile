use anyhow::{bail, Context, Result};
use std::path::Path;

/// Parsed decompiled program — ready to convert back to weight matrices
#[derive(Debug)]
pub struct ParsedProgram {
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// Transition: h_new[i] = ReLU(sum W_hh[i,j]*h[j] + sum W_hx[i,k]*x[k] + b_h[i])
    pub w_hh: Vec<Vec<f64>>,  // [hidden_dim][hidden_dim]
    pub w_hx: Vec<Vec<f64>>,  // [hidden_dim][input_dim]
    pub b_h: Vec<f64>,        // [hidden_dim]
    /// Output: logit[i] = sum W_y[i,j]*h[j] + b_y[i]
    pub w_y: Vec<Vec<f64>>,   // [output_dim][hidden_dim]
    pub b_y: Vec<f64>,        // [output_dim]
}

/// Parse a term like "2*h[1]" or "-1*x[0]" or "3" or "-1"
/// Returns (coefficient, variable_type, index)
/// variable_type: 'h', 'x', or 'c' (constant/bias)
fn parse_term(term: &str) -> Result<(f64, char, usize)> {
    let term = term.trim();

    // Pattern: COEFF*h[IDX] or COEFF*x[IDX]
    if let Some(star_pos) = term.find('*') {
        let coeff_str = &term[..star_pos];
        let var_part = &term[star_pos + 1..];

        let coeff: f64 = coeff_str.parse()
            .with_context(|| format!("Bad coefficient: '{}'", coeff_str))?;

        if var_part.starts_with("h[") && var_part.ends_with(']') {
            let idx: usize = var_part[2..var_part.len() - 1].parse()
                .with_context(|| format!("Bad h index: '{}'", var_part))?;
            return Ok((coeff, 'h', idx));
        } else if var_part.starts_with("x[") && var_part.ends_with(']') {
            let idx: usize = var_part[2..var_part.len() - 1].parse()
                .with_context(|| format!("Bad x index: '{}'", var_part))?;
            return Ok((coeff, 'x', idx));
        } else {
            bail!("Unknown variable: '{}'", var_part);
        }
    }

    // Bare constant (bias term)
    let coeff: f64 = term.parse()
        .with_context(|| format!("Bad constant: '{}'", term))?;
    Ok((coeff, 'c', 0))
}

/// Split "2*h[1] + -1*x[0] + 2*x[1] + -1" into terms
fn split_terms(expr: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in expr.chars() {
        match ch {
            '[' => { depth += 1; current.push(ch); }
            ']' => { depth -= 1; current.push(ch); }
            '+' if depth == 0 => {
                let t = current.trim().to_string();
                if !t.is_empty() {
                    terms.push(t);
                }
                current.clear();
            }
            _ => { current.push(ch); }
        }
    }
    let t = current.trim().to_string();
    if !t.is_empty() {
        terms.push(t);
    }
    terms
}

/// Parse "max(0, EXPR)" or just "0" for dead neurons
fn parse_transition_line(line: &str, hidden_dim: usize, input_dim: usize)
    -> Result<(Vec<f64>, Vec<f64>, f64)>
{
    // Dead neuron: "h0 = 0"
    let rhs = line.trim();
    if rhs == "0" {
        return Ok((vec![0.0; hidden_dim], vec![0.0; input_dim], 0.0));
    }

    // Strip "max(0, " and trailing ")"
    let inner = if rhs.starts_with("max(0,") {
        let s = &rhs[6..]; // skip "max(0,"
        let s = s.trim();
        if s.ends_with(')') {
            &s[..s.len() - 1]
        } else {
            s
        }
    } else {
        bail!("Expected 'max(0, ...)' or '0', got: '{}'", rhs);
    };

    let mut w_hh_row = vec![0.0; hidden_dim];
    let mut w_hx_row = vec![0.0; input_dim];
    let mut bias = 0.0;

    for term_str in split_terms(inner) {
        let (coeff, var_type, idx) = parse_term(&term_str)?;
        match var_type {
            'h' => {
                if idx >= hidden_dim {
                    bail!("h[{}] out of range (hidden_dim={})", idx, hidden_dim);
                }
                w_hh_row[idx] = coeff;
            }
            'x' => {
                if idx >= input_dim {
                    bail!("x[{}] out of range (input_dim={})", idx, input_dim);
                }
                w_hx_row[idx] = coeff;
            }
            'c' => {
                bias = coeff;
            }
            _ => unreachable!(),
        }
    }

    Ok((w_hh_row, w_hx_row, bias))
}

/// Parse "COEFF*h[J] + COEFF*h[K] + BIAS" for output logit
fn parse_logit_line(line: &str, hidden_dim: usize)
    -> Result<(Vec<f64>, f64)>
{
    let mut w_y_row = vec![0.0; hidden_dim];
    let mut bias = 0.0;

    for term_str in split_terms(line.trim()) {
        let (coeff, var_type, idx) = parse_term(&term_str)?;
        match var_type {
            'h' => {
                if idx >= hidden_dim {
                    bail!("h[{}] out of range (hidden_dim={})", idx, hidden_dim);
                }
                w_y_row[idx] = coeff;
            }
            'c' => {
                bias = coeff;
            }
            'x' => bail!("Unexpected x[] in output logit"),
            _ => unreachable!(),
        }
    }

    Ok((w_y_row, bias))
}

/// Parse a decompiled Python program back into weight matrices
pub fn parse_program(source: &str) -> Result<ParsedProgram> {
    let lines: Vec<&str> = source.lines().collect();

    // Extract dimensions from header comment
    let mut hidden_dim = 0;
    let mut input_dim = 0;
    let mut output_dim = 0;

    for line in &lines {
        if line.contains("Hidden dim:") {
            // Parse "# Hidden dim: 3, Input dim: 2, Output dim: 2"
            for part in line.split(',') {
                let part = part.trim().trim_start_matches('#').trim();
                if part.starts_with("Hidden dim:") {
                    hidden_dim = part.split(':').nth(1).unwrap().trim().parse()?;
                } else if part.starts_with("Input dim:") {
                    input_dim = part.split(':').nth(1).unwrap().trim().parse()?;
                } else if part.starts_with("Output dim:") {
                    output_dim = part.split(':').nth(1).unwrap().trim().parse()?;
                }
            }
        }
    }

    if hidden_dim == 0 || input_dim == 0 || output_dim == 0 {
        // Try to infer from code
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with("h = [0.0] *") {
                hidden_dim = trimmed.rsplit("* ").next().unwrap_or("0").trim().parse().unwrap_or(0);
            }
        }
        if hidden_dim == 0 {
            bail!("Cannot determine dimensions — add '# Hidden dim: N, Input dim: M, Output dim: K' comment");
        }
    }

    let mut w_hh = vec![vec![0.0; hidden_dim]; hidden_dim];
    let mut w_hx = vec![vec![0.0; input_dim]; hidden_dim];
    let mut b_h = vec![0.0; hidden_dim];
    let mut w_y = Vec::new();
    let mut b_y = Vec::new();

    let mut in_transition = false;
    let mut in_logits = false;

    for line in &lines {
        let trimmed = line.trim();

        // Detect transition block
        if trimmed == "for x in input_sequence:" {
            in_transition = true;
            continue;
        }

        // Transition lines: "h0 = max(0, ...)" or "h0 = 0"
        if in_transition && !in_logits {
            // Check for "hN = ..."
            if let Some(eq_pos) = trimmed.find(" = ") {
                let lhs = &trimmed[..eq_pos];
                if lhs.starts_with('h') && lhs.len() > 1 && lhs[1..].chars().all(|c| c.is_ascii_digit()) {
                    let neuron_idx: usize = lhs[1..].parse()?;
                    if neuron_idx < hidden_dim {
                        let rhs = &trimmed[eq_pos + 3..];
                        let (hh_row, hx_row, bias) = parse_transition_line(rhs, hidden_dim, input_dim)?;
                        w_hh[neuron_idx] = hh_row;
                        w_hx[neuron_idx] = hx_row;
                        b_h[neuron_idx] = bias;
                    }
                    continue;
                }
            }

            // "h = [h0, h1, ...]" marks end of transition
            if trimmed.starts_with("h = [") {
                continue;
            }
        }

        // Logit lines: "logits.append(...)"
        if trimmed.starts_with("logits.append(") && trimmed.ends_with(')') {
            in_logits = true;
            let inner = &trimmed[14..trimmed.len() - 1];
            let (y_row, bias) = parse_logit_line(inner, hidden_dim)?;
            w_y.push(y_row);
            b_y.push(bias);
            continue;
        }
    }

    if output_dim == 0 {
        output_dim = w_y.len();
    }

    if w_y.len() != output_dim {
        bail!("Expected {} output logits, found {}", output_dim, w_y.len());
    }

    Ok(ParsedProgram {
        hidden_dim, input_dim, output_dim,
        w_hh, w_hx, b_h, w_y, b_y,
    })
}

/// Convert parsed program back to JSON weight format
pub fn program_to_json(prog: &ParsedProgram) -> String {
    let mut out = String::new();
    out.push_str("{\n");

    // W_hh
    out.push_str("  \"W_hh\": [\n");
    for (i, row) in prog.w_hh.iter().enumerate() {
        out.push_str(&format!("    [{}]", row.iter().map(|v| format_weight(*v)).collect::<Vec<_>>().join(", ")));
        if i < prog.hidden_dim - 1 { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ],\n");

    // W_hx
    out.push_str("  \"W_hx\": [\n");
    for (i, row) in prog.w_hx.iter().enumerate() {
        out.push_str(&format!("    [{}]", row.iter().map(|v| format_weight(*v)).collect::<Vec<_>>().join(", ")));
        if i < prog.hidden_dim - 1 { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ],\n");

    // b_h
    out.push_str(&format!("  \"b_h\": [{}],\n",
        prog.b_h.iter().map(|v| format_weight(*v)).collect::<Vec<_>>().join(", ")));

    // W_y
    out.push_str("  \"W_y\": [\n");
    for (i, row) in prog.w_y.iter().enumerate() {
        out.push_str(&format!("    [{}]", row.iter().map(|v| format_weight(*v)).collect::<Vec<_>>().join(", ")));
        if i < prog.output_dim - 1 { out.push(','); }
        out.push('\n');
    }
    out.push_str("  ],\n");

    // b_y
    out.push_str(&format!("  \"b_y\": [{}]\n",
        prog.b_y.iter().map(|v| format_weight(*v)).collect::<Vec<_>>().join(", ")));

    out.push_str("}\n");
    out
}

fn format_weight(v: f64) -> String {
    if (v - v.round()).abs() < 0.001 {
        format!("{:.1}", v.round())
    } else {
        format!("{:.4}", v)
    }
}

/// Load a decompiled program from file and convert to weight JSON
pub fn patch_file(source_path: &Path, output_path: Option<&Path>) -> Result<()> {
    let source = std::fs::read_to_string(source_path)
        .with_context(|| format!("Failed to read {}", source_path.display()))?;

    let prog = parse_program(&source)?;

    eprintln!("Parsed: hidden_dim={}, input_dim={}, output_dim={}",
             prog.hidden_dim, prog.input_dim, prog.output_dim);

    // Count non-zero weights
    let total = prog.hidden_dim * prog.hidden_dim + prog.hidden_dim * prog.input_dim
        + prog.hidden_dim + prog.output_dim * prog.hidden_dim + prog.output_dim;
    let nonzero = prog.w_hh.iter().flat_map(|r| r.iter()).filter(|&&v| v.abs() > 0.001).count()
        + prog.w_hx.iter().flat_map(|r| r.iter()).filter(|&&v| v.abs() > 0.001).count()
        + prog.b_h.iter().filter(|&&v| v.abs() > 0.001).count()
        + prog.w_y.iter().flat_map(|r| r.iter()).filter(|&&v| v.abs() > 0.001).count()
        + prog.b_y.iter().filter(|&&v| v.abs() > 0.001).count();
    eprintln!("Weights: {}/{} non-zero ({:.0}% sparse)",
             nonzero, total, (1.0 - nonzero as f64 / total as f64) * 100.0);

    let json = program_to_json(&prog);

    match output_path {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("Wrote: {}", path.display());
        }
        None => print!("{}", json),
    }

    Ok(())
}
