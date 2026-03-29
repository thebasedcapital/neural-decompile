use crate::quantize::{self, QuantizedRnn};
use crate::slice;
use crate::trace;
use crate::verify::{self, TestCase};
use crate::weights::RnnWeights;
use anyhow::{Context, Result};
use std::path::Path;

/// A single training snapshot with metadata
#[derive(Debug)]
pub struct Snapshot {
    pub epoch: usize,
    pub phase: String,
    pub accuracy: f64,
    pub pct_integer: f64,
    pub rnn: RnnWeights,
}

/// Evolution analysis across all snapshots
pub struct EvolutionReport {
    pub task_name: String,
    pub total_snapshots: usize,
    pub frames: Vec<Frame>,
}

/// One frame of the evolution
pub struct Frame {
    pub epoch: usize,
    pub phase: String,
    pub accuracy: f64,
    pub pct_integer: f64,
    pub active_neurons: usize,
    pub total_neurons: usize,
    /// Per-neuron max activation (0 = dead)
    pub neuron_activity: Vec<f64>,
    /// Per-neuron residual magnitude (distance from pure integer)
    pub neuron_residual: Vec<f64>,
    /// FSM accuracy on test cases (if provided)
    pub fsm_accuracy: Option<f64>,
}

/// Load a snapshot JSON file (from train_with_snapshots.py)
fn load_snapshot(path: &Path) -> Result<Snapshot> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&data)?;

    let epoch = json["epoch"].as_u64().unwrap_or(0) as usize;
    let phase = json["phase"].as_str().unwrap_or("unknown").to_string();
    let accuracy = json["accuracy"].as_f64().unwrap_or(0.0);
    let pct_integer = json["pct_integer"].as_f64().unwrap_or(0.0);

    // Parse weight matrices
    let w_hh = parse_matrix(&json["W_hh"])?;
    let w_hx = parse_matrix(&json["W_hx"])?;
    let b_h = parse_vec(&json["b_h"])?;
    let w_y = parse_matrix(&json["W_y"])?;
    let b_y = parse_vec(&json["b_y"])?;

    let hidden_dim = w_hh.nrows();
    let input_dim = w_hx.ncols();
    let output_dim = w_y.nrows();

    Ok(Snapshot {
        epoch,
        phase,
        accuracy,
        pct_integer,
        rnn: RnnWeights {
            w_hh, w_hx, b_h, w_y, b_y,
            hidden_dim, input_dim, output_dim,
        },
    })
}

fn parse_matrix(v: &serde_json::Value) -> Result<ndarray::Array2<f64>> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone())?;
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, |r| r.len());
    let flat: Vec<f64> = rows.into_iter().flat_map(|r| r.into_iter()).collect();
    ndarray::Array2::from_shape_vec((nrows, ncols), flat)
        .context("reshape matrix")
}

fn parse_vec(v: &serde_json::Value) -> Result<Vec<f64>> {
    Ok(serde_json::from_value(v.clone())?)
}

/// Load all snapshots from a directory, sorted by epoch
pub fn load_snapshots(dir: &Path) -> Result<Vec<Snapshot>> {
    let mut snaps = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("epoch_") && name.ends_with(".json")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        match load_snapshot(&entry.path()) {
            Ok(s) => snaps.push(s),
            Err(e) => eprintln!("  skip {}: {}", entry.path().display(), e),
        }
    }

    Ok(snaps)
}

/// Analyze evolution across snapshots
pub fn analyze(
    snapshots: &[Snapshot],
    tests: Option<&[TestCase]>,
    eps: f64,
    task_name: &str,
) -> EvolutionReport {
    let mut frames = Vec::new();

    for snap in snapshots {
        let q = quantize::quantize_rnn(&snap.rnn, eps);
        let hd = q.hidden_dim;

        // Run all test inputs through to find active neurons
        let (active, activity, fsm_acc) = if let Some(tcs) = tests {
            let traces: Vec<_> = tcs.iter()
                .map(|tc| trace::trace_quantized(&q, &tc.inputs))
                .collect();

            let mut max_act = vec![0.0_f64; hd];
            let mut ever_active = vec![false; hd];
            for tr in &traces {
                for step in &tr.steps {
                    for i in 0..hd {
                        if step.hidden[i] > 0.0 {
                            ever_active[i] = true;
                            max_act[i] = max_act[i].max(step.hidden[i]);
                        }
                    }
                }
            }

            let active_count = ever_active.iter().filter(|&&a| a).count();

            // FSM accuracy
            let results = verify::run_verification(&q, tcs);
            let fsm = results.passed as f64 / results.total as f64;

            (active_count, max_act, Some(fsm))
        } else {
            (hd, vec![1.0; hd], None)
        };

        // Compute per-neuron residual (how far from pure integer)
        let mut residuals = vec![0.0_f64; hd];
        for i in 0..hd {
            let mut res = 0.0;
            for j in 0..hd {
                let v = q.w_hh[[i, j]];
                res += (v - v.round()).abs();
            }
            for j in 0..q.input_dim {
                let v = q.w_hx[[i, j]];
                res += (v - v.round()).abs();
            }
            res += (q.b_h[i] - q.b_h[i].round()).abs();
            residuals[i] = res;
        }

        frames.push(Frame {
            epoch: snap.epoch,
            phase: snap.phase.clone(),
            accuracy: snap.accuracy,
            pct_integer: snap.pct_integer,
            active_neurons: active,
            total_neurons: hd,
            neuron_activity: activity,
            neuron_residual: residuals,
            fsm_accuracy: fsm_acc,
        });
    }

    EvolutionReport {
        task_name: task_name.to_string(),
        total_snapshots: frames.len(),
        frames,
    }
}

/// Format text summary
pub fn format_evolve(report: &EvolutionReport) -> String {
    let mut out = String::new();

    out.push_str(&format!("═══ EVOLUTION: {} ({} snapshots) ═══\n\n", report.task_name, report.total_snapshots));
    out.push_str(&format!("{:>7} {:>8} {:>5} {:>5} {:>7} {:>6}",
        "epoch", "phase", "acc", "%int", "active", "fsm"));

    let hd = report.frames.first().map_or(0, |f| f.total_neurons);
    for i in 0..hd {
        out.push_str(&format!(" {:>4}", format!("h{}", i)));
    }
    out.push('\n');
    out.push_str(&"─".repeat(50 + hd * 5));
    out.push('\n');

    // Show key frames: first, phase transitions, convergence, last
    let mut prev_phase = String::new();
    let mut shown = 0;
    for (idx, f) in report.frames.iter().enumerate() {
        let is_first = idx == 0;
        let is_last = idx == report.frames.len() - 1;
        let phase_change = f.phase != prev_phase;
        let is_milestone = f.accuracy >= 1.0 && (idx == 0 || report.frames[idx - 1].accuracy < 1.0);

        if is_first || is_last || phase_change || is_milestone || shown % 8 == 0 {
            let phase_short = if f.phase.len() > 8 {
                &f.phase[..8]
            } else {
                &f.phase
            };
            let fsm_str = match f.fsm_accuracy {
                Some(a) if a >= 1.0 => "  ★".to_string(),
                Some(a) => format!("{:.0}%", a * 100.0),
                None => "  —".to_string(),
            };

            out.push_str(&format!("{:>7} {:>8} {:>4.0}% {:>4.0}% {:>3}/{:<3} {:>6}",
                f.epoch, phase_short,
                f.accuracy * 100.0, f.pct_integer * 100.0,
                f.active_neurons, f.total_neurons,
                fsm_str));

            // Neuron activity sparkline
            for i in 0..hd {
                let act = f.neuron_activity[i];
                let sym = if act == 0.0 { "  · " }
                    else if act < 1.0 { "  ░ " }
                    else if act < 3.0 { "  ▒ " }
                    else { "  █ " };
                out.push_str(sym);
            }
            out.push('\n');

            prev_phase = f.phase.clone();
        }
        shown += 1;
    }

    // Summary
    if let Some(last) = report.frames.last() {
        out.push_str(&format!("\nFinal: {} active neurons, {:.0}% integer",
            last.active_neurons, last.pct_integer * 100.0));
        if let Some(fsm) = last.fsm_accuracy {
            if fsm >= 1.0 {
                out.push_str(", FSM PERFECT");
            } else {
                out.push_str(&format!(", FSM {:.0}%", fsm * 100.0));
            }
        }
        out.push('\n');
    }

    out
}

/// Render evolution as an HTML animation
pub fn render_html(report: &EvolutionReport) -> String {
    let hd = report.frames.first().map_or(0, |f| f.total_neurons);

    // Serialize frame data for JS
    let frames_json: Vec<String> = report.frames.iter().map(|f| {
        let activity: String = f.neuron_activity.iter()
            .map(|v| format!("{:.2}", v))
            .collect::<Vec<_>>().join(",");
        let residual: String = f.neuron_residual.iter()
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>().join(",");
        let fsm = f.fsm_accuracy.map_or("null".to_string(), |v| format!("{:.4}", v));
        format!(r#"{{"epoch":{},"phase":"{}","acc":{:.4},"pct_int":{:.4},"active":{},"total":{},"fsm":{},"activity":[{}],"residual":[{}]}}"#,
            f.epoch, f.phase, f.accuracy, f.pct_integer, f.active_neurons, f.total_neurons, fsm, activity, residual)
    }).collect();

    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nd evolve — {name}</title>
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
.subtitle {{ color: var(--dim); font-size: 12px; margin-bottom: 20px; }}
.controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 12px; background: var(--card); border-radius: 8px; border: 1px solid var(--border); }}
.controls button {{ background: var(--accent); color: var(--bg); border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; font-family: var(--mono); font-weight: bold; }}
.controls button:hover {{ opacity: 0.8; }}
.controls input[type=range] {{ flex: 1; accent-color: var(--accent); }}
.controls .info {{ color: var(--dim); font-size: 12px; min-width: 200px; text-align: right; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
.metric {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }}
.metric .value {{ font-size: 24px; font-weight: bold; }}
.metric .label {{ color: var(--dim); font-size: 11px; text-transform: uppercase; margin-top: 2px; }}
.green {{ color: var(--green); }} .red {{ color: var(--red); }} .yellow {{ color: var(--yellow); }}
.neurons {{ display: flex; gap: 8px; margin-bottom: 20px; justify-content: center; flex-wrap: wrap; }}
.neuron {{ width: 80px; height: 100px; background: var(--card); border: 2px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; transition: all 0.3s; }}
.neuron .name {{ font-size: 11px; color: var(--dim); margin-bottom: 4px; }}
.neuron .bar {{ width: 40px; border-radius: 3px; transition: all 0.3s; }}
.neuron .val {{ font-size: 10px; margin-top: 4px; color: var(--dim); }}
.neuron.dead {{ opacity: 0.3; border-color: var(--red); }}
.neuron.active {{ border-color: var(--green); }}
.neuron.hybrid {{ border-color: var(--yellow); }}
canvas {{ width: 100%; height: 200px; background: var(--card); border-radius: 8px; border: 1px solid var(--border); margin-bottom: 20px; }}
.phase-bar {{ display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
.phase-seg {{ display: flex; align-items: center; justify-content: center; font-size: 10px; color: var(--bg); font-weight: bold; }}
</style>
</head>
<body>
<h1>nd evolve — {name}</h1>
<div class="subtitle">{n} snapshots · {hd} hidden neurons · watch the circuit crystallize</div>

<div class="controls">
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <input type="range" id="slider" min="0" max="{max_idx}" value="0" oninput="seek(this.value)">
    <div class="info" id="frameInfo">epoch 0</div>
</div>

<div class="metrics">
    <div class="metric"><div class="value" id="mAcc">—</div><div class="label">Accuracy</div></div>
    <div class="metric"><div class="value" id="mInt">—</div><div class="label">Integer %</div></div>
    <div class="metric"><div class="value" id="mActive">—</div><div class="label">Active Neurons</div></div>
    <div class="metric"><div class="value" id="mFsm">—</div><div class="label">FSM Verify</div></div>
</div>

<div class="neurons" id="neuronGrid"></div>

<canvas id="chart" height="200"></canvas>

<div class="subtitle" style="margin-top: 32px; text-align: center;">
    Generated by <strong>nd evolve</strong> — Neural Decompiler
</div>

<script>
const frames = [{frames_json}];
const hd = {hd};
let playing = false;
let frame = 0;
let timer = null;
let speed = 100;

// Build neuron grid
const grid = document.getElementById('neuronGrid');
for (let i = 0; i < hd; i++) {{
    const div = document.createElement('div');
    div.className = 'neuron';
    div.id = 'n' + i;
    div.innerHTML = `<div class="name">h${{i}}</div><div class="bar" id="bar${{i}}"></div><div class="val" id="val${{i}}">0</div>`;
    grid.appendChild(div);
}}

function renderFrame(idx) {{
    const f = frames[idx];
    frame = idx;
    document.getElementById('slider').value = idx;
    document.getElementById('frameInfo').textContent = `epoch ${{f.epoch}} · ${{f.phase}}`;

    // Metrics
    const accEl = document.getElementById('mAcc');
    accEl.textContent = (f.acc * 100).toFixed(0) + '%';
    accEl.className = 'value ' + (f.acc >= 1 ? 'green' : f.acc >= 0.9 ? 'yellow' : 'red');

    const intEl = document.getElementById('mInt');
    intEl.textContent = (f.pct_int * 100).toFixed(0) + '%';
    intEl.className = 'value ' + (f.pct_int >= 0.95 ? 'green' : f.pct_int >= 0.75 ? 'yellow' : 'red');

    document.getElementById('mActive').textContent = f.active + '/' + f.total;

    const fsmEl = document.getElementById('mFsm');
    if (f.fsm !== null) {{
        fsmEl.textContent = f.fsm >= 1 ? '★' : (f.fsm * 100).toFixed(0) + '%';
        fsmEl.className = 'value ' + (f.fsm >= 1 ? 'green' : 'yellow');
    }} else {{
        fsmEl.textContent = '—';
        fsmEl.className = 'value';
    }}

    // Neurons
    const maxAct = Math.max(...f.activity, 1);
    for (let i = 0; i < hd; i++) {{
        const el = document.getElementById('n' + i);
        const bar = document.getElementById('bar' + i);
        const val = document.getElementById('val' + i);
        const act = f.activity[i];
        const res = f.residual[i];

        const h = Math.round((act / maxAct) * 60);
        const isInt = res < 0.1;
        const isDead = act === 0;

        el.className = 'neuron ' + (isDead ? 'dead' : isInt ? 'active' : 'hybrid');
        bar.style.height = h + 'px';
        bar.style.background = isDead ? '#f85149' : isInt ? '#3fb950' : '#d29922';
        val.textContent = act.toFixed(1);
    }}

    drawChart(idx);
}}

function drawChart(currentIdx) {{
    const canvas = document.getElementById('chart');
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 400;
    ctx.clearRect(0, 0, w, h);

    const pad = 40;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;
    const n = frames.length;

    // Draw accuracy line
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
        const x = pad + (i / (n - 1)) * plotW;
        const y = pad + plotH - frames[i].acc * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Draw integer % line
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
        const x = pad + (i / (n - 1)) * plotW;
        const y = pad + plotH - frames[i].pct_int * plotH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Draw FSM line
    ctx.strokeStyle = '#d29922';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {{
        if (frames[i].fsm === null) continue;
        const x = pad + (i / (n - 1)) * plotW;
        const y = pad + plotH - frames[i].fsm * plotH;
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.setLineDash([]);

    // Current frame marker
    const cx = pad + (currentIdx / (n - 1)) * plotW;
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, pad);
    ctx.lineTo(cx, pad + plotH);
    ctx.stroke();

    // Legend
    ctx.font = '20px SF Mono, monospace';
    ctx.fillStyle = '#58a6ff'; ctx.fillText('— accuracy', pad, h - 8);
    ctx.fillStyle = '#3fb950'; ctx.fillText('— integer %', pad + 180, h - 8);
    ctx.fillStyle = '#d29922'; ctx.fillText('- - FSM', pad + 360, h - 8);
}}

function togglePlay() {{
    playing = !playing;
    document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
    if (playing) {{
        timer = setInterval(() => {{
            frame++;
            if (frame >= frames.length) {{ frame = 0; }}
            renderFrame(frame);
        }}, speed);
    }} else {{
        clearInterval(timer);
    }}
}}

function seek(idx) {{
    renderFrame(parseInt(idx));
}}

// Handle keyboard
document.addEventListener('keydown', (e) => {{
    if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
    if (e.key === 'ArrowRight') {{ renderFrame(Math.min(frame + 1, frames.length - 1)); }}
    if (e.key === 'ArrowLeft') {{ renderFrame(Math.max(frame - 1, 0)); }}
}});

renderFrame(0);
window.addEventListener('resize', () => drawChart(frame));
</script>
</body>
</html>"#,
        name = report.task_name,
        n = report.total_snapshots,
        hd = hd,
        max_idx = report.frames.len().saturating_sub(1),
        frames_json = frames_json.join(","),
    )
}
