use crate::trace::Trace;

/// Generate a standalone HTML visualization of a trace
pub fn trace_to_html(trace: &Trace, title: &str) -> String {
    let mut html = String::new();

    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nd trace — {title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; padding: 24px; }}
  h1 {{ font-size: 18px; color: #7aa2f7; margin-bottom: 4px; }}
  .subtitle {{ font-size: 12px; color: #565f89; margin-bottom: 20px; }}
  .grid {{ display: grid; gap: 2px; }}
  .header {{ background: #1a1b26; padding: 6px 10px; font-size: 11px; color: #7aa2f7; text-align: center; font-weight: 600; }}
  .cell {{ padding: 6px 10px; font-size: 12px; text-align: center; border-radius: 2px; transition: all 0.15s; position: relative; }}
  .cell:hover {{ transform: scale(1.05); z-index: 10; box-shadow: 0 0 12px rgba(122, 162, 247, 0.4); }}
  .t-col {{ background: #1a1b26; color: #565f89; font-size: 11px; }}
  .input-col {{ background: #1a1b26; color: #c0caf5; font-size: 11px; }}
  .legend {{ margin-top: 16px; display: flex; gap: 16px; font-size: 11px; color: #565f89; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 2px; }}
  .logits {{ margin-top: 16px; padding: 12px; background: #1a1b26; border-radius: 4px; font-size: 12px; }}
  .logit-bar {{ height: 20px; border-radius: 2px; margin: 4px 0; display: flex; align-items: center; padding: 0 8px; font-size: 11px; }}
  .killed {{ position: absolute; top: 1px; right: 2px; font-size: 8px; color: #f7768e; opacity: 0.7; }}
</style>
</head>
<body>
<h1>nd trace — {title}</h1>
<div class="subtitle">hidden_dim={hd} | {steps} timesteps | prediction=class {pred}</div>
"#, title = title, hd = trace.hidden_dim, steps = trace.steps.len(), pred = trace.prediction));

    // Find max hidden value for color scaling
    let max_val = trace.steps.iter()
        .flat_map(|s| s.hidden.iter())
        .fold(1.0_f64, |acc, &v| acc.max(v));

    // Grid layout: timestep | input | h0 | h1 | ... | hN
    let cols = 2 + trace.hidden_dim;
    html.push_str(&format!(r#"<div class="grid" style="grid-template-columns: 40px 60px repeat({}, 1fr);">"#, trace.hidden_dim));

    // Header row
    html.push_str(r#"<div class="header">t</div>"#);
    html.push_str(r#"<div class="header">input</div>"#);
    for i in 0..trace.hidden_dim {
        html.push_str(&format!(r#"<div class="header">h{}</div>"#, i));
    }

    // Initial state row
    html.push_str(r#"<div class="cell t-col">-</div>"#);
    html.push_str(r#"<div class="cell input-col">—</div>"#);
    for _ in 0..trace.hidden_dim {
        html.push_str(&format!(r#"<div class="cell" style="background: #1a1b26; color: #565f89;">0</div>"#));
    }

    // Each timestep
    for step in &trace.steps {
        html.push_str(&format!(r#"<div class="cell t-col">{}</div>"#, step.t));

        // Input display
        let input_str = if step.input.len() == 2 {
            if step.input[0] > step.input[1] { "0".to_string() } else { "1".to_string() }
        } else {
            step.input.iter()
                .enumerate()
                .find(|(_, &v)| v > 0.5)
                .map(|(i, _)| i.to_string())
                .unwrap_or("?".to_string())
        };
        html.push_str(&format!(r#"<div class="cell input-col">{}</div>"#, input_str));

        // Hidden state cells
        for i in 0..trace.hidden_dim {
            let h = step.hidden[i];
            let killed = step.pre_relu[i] < 0.0 && h == 0.0;

            let (bg, fg) = if h == 0.0 {
                if killed {
                    // ReLU killed — dark red hint
                    ("rgba(247, 118, 142, 0.08)".to_string(), "#565f89".to_string())
                } else {
                    ("#1a1b26".to_string(), "#565f89".to_string())
                }
            } else {
                // Scale from dim blue to bright orange based on magnitude
                let t = (h / max_val).min(1.0);
                if t < 0.5 {
                    // Low: blue shades
                    let intensity = (t * 2.0 * 255.0) as u8;
                    (format!("rgba(122, 162, 247, {:.2})", 0.1 + t * 0.6), "#c0caf5".to_string())
                } else {
                    // High: orange/gold
                    let intensity = ((t - 0.5) * 2.0);
                    (format!("rgba(224, 175, 104, {:.2})", 0.3 + intensity * 0.7), "#1a1b26".to_string())
                }
            };

            let val_str = if h == 0.0 {
                "·".to_string()
            } else if (h - h.round()).abs() < 0.01 {
                format!("{:.0}", h)
            } else {
                format!("{:.2}", h)
            };

            let killed_marker = if killed { r#"<span class="killed">✗</span>"# } else { "" };

            html.push_str(&format!(
                r#"<div class="cell" style="background: {}; color: {};" title="h{}={:.4} (pre-ReLU={:.4})">{}{}</div>"#,
                bg, fg, i, h, step.pre_relu[i], val_str, killed_marker
            ));
        }
    }

    html.push_str("</div>\n");

    // Legend
    html.push_str(r#"<div class="legend">"#);
    html.push_str(r#"<div class="legend-item"><div class="legend-swatch" style="background: #1a1b26;"></div> zero</div>"#);
    html.push_str(r#"<div class="legend-item"><div class="legend-swatch" style="background: rgba(122, 162, 247, 0.4);"></div> low</div>"#);
    html.push_str(r#"<div class="legend-item"><div class="legend-swatch" style="background: rgba(224, 175, 104, 0.7);"></div> high</div>"#);
    html.push_str(r#"<div class="legend-item"><div class="legend-swatch" style="background: rgba(247, 118, 142, 0.15);"></div><span style="color: #f7768e;">✗</span> ReLU killed</div>"#);
    html.push_str("</div>\n");

    // Output logits as bars
    html.push_str(r#"<div class="logits">"#);
    html.push_str(r#"<div style="color: #7aa2f7; font-weight: 600; margin-bottom: 8px;">Output Logits</div>"#);

    let max_logit = trace.output_logits.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);

    for (i, &logit) in trace.output_logits.iter().enumerate() {
        let width_pct = (logit.abs() / max_logit * 80.0).max(2.0);
        let (color, direction) = if logit >= 0.0 {
            if i == trace.prediction {
                ("rgba(158, 206, 106, 0.7)", "left")
            } else {
                ("rgba(122, 162, 247, 0.3)", "left")
            }
        } else {
            ("rgba(247, 118, 142, 0.3)", "right")
        };

        let winner = if i == trace.prediction { " ← winner" } else { "" };

        html.push_str(&format!(
            r#"<div class="logit-bar" style="width: {:.0}%; background: {};">class {} = {:.2}{}</div>"#,
            width_pct, color, i, logit, winner
        ));
    }

    html.push_str("</div>\n");
    html.push_str("</body>\n</html>");

    html
}
