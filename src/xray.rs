use crate::emit;
use crate::quantize::{self, QuantizedRnn};
use crate::slice;
use crate::trace::{self, Trace};
use crate::verify::{self, TestCase};
use crate::weights::RnnWeights;

/// Full X-ray analysis of a neural circuit
pub struct XrayReport {
    pub name: String,
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub pct_integer: f64,
    pub total_weights: usize,
    pub dead_neurons: Vec<usize>,
    /// Sliced circuit info (if tests provided)
    pub slice_info: Option<SliceInfo>,
    /// Hybrid decomposition per neuron
    pub neurons: Vec<NeuronXray>,
    /// Output layer decomposition
    pub outputs: Vec<OutputXray>,
    /// Verification results (if tests provided)
    pub verification: Option<VerifyInfo>,
    /// Sample traces on interesting inputs
    pub sample_traces: Vec<SampleTrace>,
    /// Decompiled Python code
    pub python_code: String,
}

pub struct SliceInfo {
    pub original_dim: usize,
    pub sliced_dim: usize,
    pub kept: Vec<usize>,
    pub removed: Vec<usize>,
    pub param_reduction_pct: f64,
}

pub struct NeuronXray {
    pub index: usize,
    /// Integer backbone terms: (source, coeff_int)
    pub integer_terms: Vec<(String, i64)>,
    /// Residual terms: (source, coeff_float)
    pub residual_terms: Vec<(String, f64)>,
    /// Total residual magnitude
    pub residual_magnitude: f64,
    /// Bias (integer part)
    pub bias_int: Option<i64>,
    /// Bias residual
    pub bias_residual: Option<f64>,
}

pub struct OutputXray {
    pub class: usize,
    pub integer_terms: Vec<(String, i64)>,
    pub residual_terms: Vec<(String, f64)>,
    pub residual_magnitude: f64,
    pub bias_int: Option<i64>,
    pub bias_residual: Option<f64>,
}

pub struct VerifyInfo {
    pub passed: usize,
    pub total: usize,
    pub perfect: bool,
}

pub struct SampleTrace {
    pub input_str: String,
    pub trace: Trace,
}

fn decompose_weight(v: f64) -> (Option<i64>, Option<f64>) {
    if v.abs() < 0.005 {
        return (None, None);
    }
    let rounded = v.round() as i64;
    let residual = v - rounded as f64;
    if residual.abs() < 0.01 {
        (Some(rounded), None)
    } else {
        // If the integer part is 0, the whole thing is residual
        if rounded == 0 {
            (None, Some(v))
        } else {
            (Some(rounded), Some(residual))
        }
    }
}

fn xray_neuron(q: &QuantizedRnn, i: usize) -> NeuronXray {
    let mut integer_terms = Vec::new();
    let mut residual_terms = Vec::new();
    let mut residual_mag = 0.0_f64;

    // W_hh row
    for j in 0..q.hidden_dim {
        let v = q.w_hh[[i, j]];
        let (int_part, res_part) = decompose_weight(v);
        if let Some(c) = int_part {
            integer_terms.push((format!("h[{}]", j), c));
        }
        if let Some(r) = res_part {
            residual_terms.push((format!("h[{}]", j), r));
            residual_mag += r.abs();
        }
    }

    // W_hx row
    for j in 0..q.input_dim {
        let v = q.w_hx[[i, j]];
        let (int_part, res_part) = decompose_weight(v);
        if let Some(c) = int_part {
            integer_terms.push((format!("x[{}]", j), c));
        }
        if let Some(r) = res_part {
            residual_terms.push((format!("x[{}]", j), r));
            residual_mag += r.abs();
        }
    }

    // Bias
    let (bias_int, bias_residual) = decompose_weight(q.b_h[i]);
    if let Some(r) = bias_residual {
        residual_mag += r.abs();
    }

    NeuronXray {
        index: i,
        integer_terms,
        residual_terms,
        residual_magnitude: residual_mag,
        bias_int,
        bias_residual,
    }
}

fn xray_output(q: &QuantizedRnn, o: usize) -> OutputXray {
    let mut integer_terms = Vec::new();
    let mut residual_terms = Vec::new();
    let mut residual_mag = 0.0_f64;

    for j in 0..q.hidden_dim {
        let v = q.w_y[[o, j]];
        let (int_part, res_part) = decompose_weight(v);
        if let Some(c) = int_part {
            integer_terms.push((format!("h[{}]", j), c));
        }
        if let Some(r) = res_part {
            residual_terms.push((format!("h[{}]", j), r));
            residual_mag += r.abs();
        }
    }

    let (bias_int, bias_residual) = decompose_weight(q.b_y[o]);
    if let Some(r) = bias_residual {
        residual_mag += r.abs();
    }

    OutputXray {
        class: o,
        integer_terms,
        residual_terms,
        residual_magnitude: residual_mag,
        bias_int,
        bias_residual,
    }
}

/// Generate interesting sample inputs for tracing
fn pick_sample_inputs(rnn: &RnnWeights, tests: Option<&[TestCase]>) -> Vec<(String, Vec<Vec<f64>>)> {
    let mut samples = Vec::new();

    if let Some(tcs) = tests {
        // Pick one per output class
        let mut seen_classes = std::collections::HashSet::new();
        for tc in tcs {
            if seen_classes.contains(&tc.expected) {
                continue;
            }
            seen_classes.insert(tc.expected);

            let input_str: String = tc.inputs.iter().map(|x| {
                if x.len() == 2 {
                    if x[0] > x[1] { "0" } else { "1" }
                } else {
                    "?"
                }
            }).collect();

            samples.push((format!("{}→class{}", input_str, tc.expected), tc.inputs.clone()));
            if samples.len() >= 4 {
                break;
            }
        }
    } else {
        // Generate some default binary sequences
        for bits in &["0,0,0", "1,1,1", "1,0,1", "0,1,0"] {
            let input_vecs: Vec<Vec<f64>> = bits.split(',').map(|b| {
                let b: u8 = b.parse().unwrap();
                if rnn.input_dim == 2 {
                    if b == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
                } else {
                    vec![b as f64]
                }
            }).collect();
            samples.push((bits.to_string(), input_vecs));
        }
    }

    samples
}

/// Run the full X-ray analysis
pub fn run_xray(
    rnn: &RnnWeights,
    q: &QuantizedRnn,
    name: &str,
    tests: Option<&[TestCase]>,
) -> XrayReport {
    let stats = quantize::weight_stats(q);

    // Slice analysis
    let slice_info = tests.map(|tcs| {
        let result = slice::slice_from_tests(q, tcs);
        let old_p = result.original_hidden_dim * result.original_hidden_dim
            + result.original_hidden_dim * q.input_dim
            + result.original_hidden_dim
            + q.output_dim * result.original_hidden_dim
            + q.output_dim;
        let new_hd = result.kept_neurons.len();
        let new_p = new_hd * new_hd + new_hd * q.input_dim + new_hd + q.output_dim * new_hd + q.output_dim;
        SliceInfo {
            original_dim: result.original_hidden_dim,
            sliced_dim: new_hd,
            kept: result.kept_neurons,
            removed: result.removed_neurons,
            param_reduction_pct: (1.0 - new_p as f64 / old_p as f64) * 100.0,
        }
    });

    // Neuron decomposition
    let neurons: Vec<NeuronXray> = (0..q.hidden_dim).map(|i| xray_neuron(q, i)).collect();
    let outputs: Vec<OutputXray> = (0..q.output_dim).map(|o| xray_output(q, o)).collect();

    // Verification
    let verification = tests.map(|tcs| {
        let results = crate::verify::run_verification(q, tcs);
        VerifyInfo {
            passed: results.passed,
            total: results.total,
            perfect: results.passed == results.total,
        }
    });

    // Sample traces
    let sample_inputs = pick_sample_inputs(rnn, tests);
    let sample_traces: Vec<SampleTrace> = sample_inputs.into_iter().map(|(label, inputs)| {
        let tr = trace::trace_quantized(q, &inputs);
        SampleTrace { input_str: label, trace: tr }
    }).collect();

    // Python code
    let python_code = emit::emit_python(q, name);

    XrayReport {
        name: name.to_string(),
        hidden_dim: q.hidden_dim,
        input_dim: q.input_dim,
        output_dim: q.output_dim,
        pct_integer: stats.pct_integer * 100.0,
        total_weights: stats.total_weights,
        dead_neurons: stats.dead_neurons,
        slice_info,
        neurons,
        outputs,
        verification,
        sample_traces,
        python_code,
    }
}

/// Render the X-ray report as an HTML page
pub fn render_html(report: &XrayReport) -> String {
    let mut html = String::new();

    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nd xray — {name}</title>
<style>
:root {{
    --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --card: #161b22; --border: #30363d; --dim: #8b949e;
    --mono: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--mono); font-size: 14px; padding: 24px; max-width: 1200px; margin: 0 auto; }}
h1 {{ color: var(--accent); font-size: 20px; margin-bottom: 4px; }}
h2 {{ color: var(--accent); font-size: 16px; margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
h3 {{ color: var(--fg); font-size: 14px; margin: 16px 0 8px; }}
.subtitle {{ color: var(--dim); font-size: 12px; margin-bottom: 20px; }}
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.stat {{ text-align: center; }}
.stat .value {{ font-size: 28px; font-weight: bold; }}
.stat .label {{ color: var(--dim); font-size: 11px; text-transform: uppercase; margin-top: 2px; }}
.green {{ color: var(--green); }}
.red {{ color: var(--red); }}
.yellow {{ color: var(--yellow); }}
.dim {{ color: var(--dim); }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
.badge-green {{ background: rgba(63,185,80,0.15); color: var(--green); }}
.badge-yellow {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
.badge-red {{ background: rgba(248,81,73,0.15); color: var(--red); }}
.neuron {{ margin-bottom: 12px; }}
.neuron-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.equation {{ background: #0d1117; padding: 8px 12px; border-radius: 4px; border-left: 3px solid var(--accent); overflow-x: auto; }}
.equation .int {{ color: var(--green); font-weight: bold; }}
.equation .res {{ color: var(--yellow); font-style: italic; }}
.equation .op {{ color: var(--dim); }}
pre {{ background: #0d1117; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
code {{ color: var(--fg); }}
.trace-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.trace-table th {{ background: var(--card); color: var(--dim); text-align: center; padding: 4px 6px; border-bottom: 1px solid var(--border); }}
.trace-table td {{ text-align: center; padding: 4px 6px; border-bottom: 1px solid var(--border); }}
.trace-table .zero {{ color: #30363d; }}
.trace-table .hot {{ background: rgba(63,185,80,0.15); color: var(--green); }}
.trace-table .warm {{ background: rgba(210,153,34,0.1); color: var(--yellow); }}
.trace-table .input-col {{ color: var(--accent); font-weight: bold; }}
.neuron-map {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
.neuron-chip {{ padding: 4px 10px; border-radius: 4px; font-size: 12px; }}
.chip-kept {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.3); }}
.chip-dead {{ background: rgba(248,81,73,0.1); color: var(--red); border: 1px solid rgba(248,81,73,0.2); text-decoration: line-through; }}
</style>
</head>
<body>
<h1>nd xray — {name}</h1>
<div class="subtitle">Neural Decompiler Circuit Analysis</div>
"#, name = report.name));

    // Stats grid
    let verify_str = match &report.verification {
        Some(v) if v.perfect => format!("<span class=\"green\">{}/{} PERFECT</span>", v.passed, v.total),
        Some(v) => format!("<span class=\"yellow\">{}/{}</span>", v.passed, v.total),
        None => "<span class=\"dim\">—</span>".to_string(),
    };
    let int_class = if report.pct_integer >= 95.0 { "green" } else if report.pct_integer >= 75.0 { "yellow" } else { "red" };

    html.push_str(&format!(r#"
<div class="grid">
    <div class="card stat"><div class="value">{hd}</div><div class="label">Hidden Neurons</div></div>
    <div class="card stat"><div class="value {int_class}">{pct:.0}%</div><div class="label">Integer Weights</div></div>
    <div class="card stat"><div class="value">{tw}</div><div class="label">Total Parameters</div></div>
    <div class="card stat"><div class="value">{verify}</div><div class="label">Verification</div></div>
</div>
"#, hd = report.hidden_dim, pct = report.pct_integer, int_class = int_class,
    tw = report.total_weights, verify = verify_str));

    // Slice info
    if let Some(ref si) = report.slice_info {
        html.push_str("<h2>Circuit Slice</h2>\n<div class=\"card\">\n");
        if si.removed.is_empty() {
            html.push_str("<p class=\"green\">All neurons active — no dead weight. This circuit is already minimal.</p>\n");
        } else {
            html.push_str(&format!("<p>{} → {} neurons ({:.0}% parameter reduction)</p>\n",
                si.original_dim, si.sliced_dim, si.param_reduction_pct));
        }
        html.push_str("<div class=\"neuron-map\">\n");
        for i in 0..si.original_dim {
            if si.kept.contains(&i) {
                html.push_str(&format!("<span class=\"neuron-chip chip-kept\">h{}</span>\n", i));
            } else {
                html.push_str(&format!("<span class=\"neuron-chip chip-dead\">h{}</span>\n", i));
            }
        }
        html.push_str("</div>\n</div>\n");
    }

    // Hybrid decomposition
    html.push_str("<h2>Transition Function (Hybrid Decomposition)</h2>\n");
    for n in &report.neurons {
        let res_badge = if n.residual_magnitude < 0.01 {
            "<span class=\"badge badge-green\">pure integer</span>".to_string()
        } else if n.residual_magnitude < 0.5 {
            format!("<span class=\"badge badge-yellow\">residual {:.2}</span>", n.residual_magnitude)
        } else {
            format!("<span class=\"badge badge-red\">residual {:.2}</span>", n.residual_magnitude)
        };

        html.push_str(&format!("<div class=\"neuron\"><div class=\"neuron-header\"><strong>h{}</strong> {}</div>\n",
            n.index, res_badge));
        html.push_str("<div class=\"equation\">h");
        html.push_str(&format!("{}", n.index));
        html.push_str(" = max(0, ");

        let mut first = true;
        // Integer backbone
        for (src, coeff) in &n.integer_terms {
            if !first {
                html.push_str(if *coeff >= 0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                let abs_c = coeff.abs();
                if abs_c == 1 {
                    html.push_str(&format!("<span class=\"int\">{}</span>", src));
                } else {
                    html.push_str(&format!("<span class=\"int\">{}{}</span>", abs_c, src));
                }
            } else {
                if *coeff == -1 {
                    html.push_str(&format!("<span class=\"int\">-{}</span>", src));
                } else if *coeff == 1 {
                    html.push_str(&format!("<span class=\"int\">{}</span>", src));
                } else {
                    html.push_str(&format!("<span class=\"int\">{}{}</span>", coeff, src));
                }
            }
            first = false;
        }
        // Residual terms
        for (src, coeff) in &n.residual_terms {
            if !first {
                html.push_str(if *coeff >= 0.0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                let abs_c = coeff.abs();
                html.push_str(&format!("<span class=\"res\">{:.2}·{}</span>", abs_c, src));
            } else {
                html.push_str(&format!("<span class=\"res\">{:.2}·{}</span>", coeff, src));
            }
            first = false;
        }
        // Bias
        if let Some(bi) = n.bias_int {
            if !first {
                html.push_str(if bi >= 0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                html.push_str(&format!("<span class=\"int\">{}</span>", bi.abs()));
            } else {
                html.push_str(&format!("<span class=\"int\">{}</span>", bi));
            }
            first = false;
        }
        if let Some(br) = n.bias_residual {
            if !first {
                html.push_str(if br >= 0.0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                html.push_str(&format!("<span class=\"res\">{:.2}</span>", br.abs()));
            } else {
                html.push_str(&format!("<span class=\"res\">{:.2}</span>", br));
            }
        }
        if first {
            html.push_str("<span class=\"dim\">0</span>");
        }
        html.push_str(")</div>\n</div>\n");
    }

    // Output layer
    html.push_str("<h2>Output Layer</h2>\n");
    for o in &report.outputs {
        html.push_str(&format!("<div class=\"neuron\"><div class=\"neuron-header\"><strong>class {}</strong></div>\n", o.class));
        html.push_str("<div class=\"equation\">");

        let mut first = true;
        for (src, coeff) in &o.integer_terms {
            if !first {
                html.push_str(if *coeff >= 0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                let abs_c = coeff.abs();
                if abs_c == 1 {
                    html.push_str(&format!("<span class=\"int\">{}</span>", src));
                } else {
                    html.push_str(&format!("<span class=\"int\">{}{}</span>", abs_c, src));
                }
            } else {
                if *coeff < 0 {
                    html.push_str(&format!("<span class=\"int\">{}{}</span>", coeff, src));
                } else if *coeff == 1 {
                    html.push_str(&format!("<span class=\"int\">{}</span>", src));
                } else {
                    html.push_str(&format!("<span class=\"int\">{}{}</span>", coeff, src));
                }
            }
            first = false;
        }
        for (src, coeff) in &o.residual_terms {
            if !first {
                html.push_str(if *coeff >= 0.0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                html.push_str(&format!("<span class=\"res\">{:.2}·{}</span>", coeff.abs(), src));
            } else {
                html.push_str(&format!("<span class=\"res\">{:.2}·{}</span>", coeff, src));
            }
            first = false;
        }
        if let Some(bi) = o.bias_int {
            if !first {
                html.push_str(if bi >= 0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                html.push_str(&format!("<span class=\"int\">{}</span>", bi.abs()));
            } else {
                html.push_str(&format!("<span class=\"int\">{}</span>", bi));
            }
            first = false;
        }
        if let Some(br) = o.bias_residual {
            if !first {
                html.push_str(if br >= 0.0 { " <span class=\"op\">+</span> " } else { " <span class=\"op\">-</span> " });
                html.push_str(&format!("<span class=\"res\">{:.2}</span>", br.abs()));
            }
        }
        if first {
            html.push_str("<span class=\"dim\">0</span>");
        }
        html.push_str("</div>\n</div>\n");
    }

    // Sample traces
    if !report.sample_traces.is_empty() {
        html.push_str("<h2>Sample Traces</h2>\n");
        for st in &report.sample_traces {
            html.push_str(&format!("<h3>Input: {} → class {}</h3>\n", st.input_str, st.trace.prediction));
            html.push_str("<table class=\"trace-table\"><tr><th>t</th><th>in</th>");
            for i in 0..st.trace.hidden_dim {
                html.push_str(&format!("<th>h{}</th>", i));
            }
            html.push_str("</tr>\n");

            for step in &st.trace.steps {
                let in_str = if step.input.len() == 2 {
                    if step.input[0] > step.input[1] { "0" } else { "1" }
                } else {
                    "?"
                };
                html.push_str(&format!("<tr><td>{}</td><td class=\"input-col\">{}</td>", step.t, in_str));
                for i in 0..st.trace.hidden_dim {
                    let v = step.hidden[i];
                    let class = if v == 0.0 { "zero" } else if v >= 2.0 { "hot" } else { "warm" };
                    let display = if v == 0.0 { "·".to_string() } else if (v - v.round()).abs() < 0.01 { format!("{}", v.round() as i64) } else { format!("{:.1}", v) };
                    html.push_str(&format!("<td class=\"{}\">{}</td>", class, display));
                }
                html.push_str("</tr>\n");
            }

            // Output row
            html.push_str("<tr><td colspan=\"2\"><strong>logits</strong></td>");
            let max_logit = st.trace.output_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for (i, &l) in st.trace.output_logits.iter().enumerate() {
                // Pad with empty cells if hidden_dim > output_dim
                if i < st.trace.hidden_dim {
                    let class = if l == max_logit { "hot" } else { "" };
                    let display = if (l - l.round()).abs() < 0.01 { format!("{}", l.round() as i64) } else { format!("{:.1}", l) };
                    html.push_str(&format!("<td class=\"{}\">{}</td>", class, display));
                }
            }
            // Fill remaining columns
            for _ in report.output_dim..st.trace.hidden_dim {
                html.push_str("<td></td>");
            }
            html.push_str("</tr>\n</table>\n");
        }
    }

    // Decompiled code
    html.push_str("<h2>Decompiled Code</h2>\n<pre><code>");
    // Escape HTML
    let escaped = report.python_code
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    html.push_str(&escaped);
    html.push_str("</code></pre>\n");

    html.push_str(r#"
<div class="subtitle" style="margin-top: 32px; text-align: center;">
    Generated by <strong>nd xray</strong> — Neural Decompiler
</div>
</body>
</html>"#);

    html
}

/// Format a plain-text version of the xray
pub fn format_xray(report: &XrayReport) -> String {
    let mut out = String::new();

    out.push_str(&format!("═══ XRAY: {} ═══\n", report.name));
    out.push_str(&format!("  hidden={} input={} output={} params={} int={:.0}%\n",
        report.hidden_dim, report.input_dim, report.output_dim,
        report.total_weights, report.pct_integer));

    if let Some(ref v) = report.verification {
        out.push_str(&format!("  verify: {}/{}{}\n", v.passed, v.total,
            if v.perfect { " PERFECT" } else { "" }));
    }

    if let Some(ref si) = report.slice_info {
        if si.removed.is_empty() {
            out.push_str("  slice: all neurons active (minimal circuit)\n");
        } else {
            out.push_str(&format!("  slice: {} → {} neurons ({:.0}% reduction)\n",
                si.original_dim, si.sliced_dim, si.param_reduction_pct));
        }
    }

    out.push_str("\n── Hybrid Decomposition ──\n");
    for n in &report.neurons {
        let int_str: String = n.integer_terms.iter().map(|(src, c)| {
            if *c == 1 { src.clone() } else if *c == -1 { format!("-{}", src) } else { format!("{}·{}", c, src) }
        }).collect::<Vec<_>>().join(" + ");

        let res_str: String = n.residual_terms.iter().map(|(src, c)| {
            format!("{:.2}·{}", c, src)
        }).collect::<Vec<_>>().join(" + ");

        let bias_str = match (n.bias_int, n.bias_residual) {
            (Some(i), Some(r)) => format!(" + {} + {:.2}", i, r),
            (Some(i), None) => format!(" + {}", i),
            (None, Some(r)) => format!(" + {:.2}", r),
            (None, None) => String::new(),
        };

        let purity = if n.residual_magnitude < 0.01 { "INT" } else { "HYB" };
        let all_empty = int_str.is_empty() && res_str.is_empty() && bias_str.is_empty();
        if all_empty {
            out.push_str(&format!("  [---] h{} = 0  (dead neuron)\n", n.index));
        } else {
            out.push_str(&format!("  [{}] h{} = max(0, {}", purity, n.index, int_str));
            if !res_str.is_empty() {
                out.push_str(&format!(" | +{}", res_str));
            }
            out.push_str(&bias_str);
            out.push_str(")\n");
        }
    }

    out.push_str("\n");
    out.push_str(&report.python_code);

    out
}
