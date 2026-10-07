"""Serialize compact patches and validate a dependency-free streaming Rust decoder."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from model import Decoder, apply_patch


RUNTIME = r'''
use std::io::{self, BufRead, Write};

struct Decoder {
    history: [[f64; CHANNELS]; TAPS],
    cursor: usize,
}
impl Decoder {
    fn new() -> Self { Self { history: [[0.0; CHANNELS]; TAPS], cursor: 0 } }
    fn step(&mut self, input: [f64; CHANNELS], model: usize) -> [f64; 7] {
        self.history[self.cursor] = input;
        let mut rates = [0.0; CHANNELS];
        for (lag, weight) in KERNEL.iter().enumerate() {
            let row = &self.history[(self.cursor + TAPS - lag) % TAPS];
            for (rate, count) in rates.iter_mut().zip(row) { *rate += weight * count; }
        }
        self.cursor = (self.cursor + 1) % TAPS;
        let mut output = BIAS[model];
        for (rate, row) in rates.iter().zip(&WEIGHTS[model]) {
            for (value, coefficient) in output.iter_mut().zip(row) { *value += rate * coefficient; }
        }
        output
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());
    let mut line = String::with_capacity(CHANNELS * 24);
    let mut decoder = Decoder::new();
    let mut model = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }
        if line.trim().is_empty() { continue; }
        if let Some(rest) = line.trim().strip_prefix("reset ") {
            model = rest.parse::<usize>()?;
            if model >= MODELS { return Err("model index out of range".into()); }
            decoder = Decoder::new();
            continue;
        }
        let mut fields = line.split_whitespace();
        let mut counts = [0.0; CHANNELS];
        for count in &mut counts {
            *count = fields.next().ok_or("too few neural channels")?.parse::<f64>()?;
            if !count.is_finite() || *count < 0.0 { return Err("invalid neural count".into()); }
        }
        if fields.next().is_some() { return Err("too many neural channels".into()); }
        let values = decoder.step(counts, model);
        for (index, value) in values.iter().enumerate() {
            if index > 0 { write!(writer, " ")?; }
            write!(writer, "{value:.17}")?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}
'''


def compile_and_validate(decoder, models, cases, protocol, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_path = destination.with_name("recovery-base.json")
    base_path.write_text(json.dumps({name: array.tolist() for name, array in asdict(decoder).items()},
                                    separators=(",", ":"), allow_nan=False) + "\n")
    # Compilation uses the serialized base and patch, not an in-memory fitted shortcut.
    restored = Decoder(**{name: np.asarray(value) for name, value in json.loads(base_path.read_text()).items()})
    slopes, biases = [], []
    for model in models:
        patch = json.loads(json.dumps(model["patch"], allow_nan=False))
        weights = apply_patch(restored, model["method"], patch)
        slope, bias = restored.folded(weights)
        slopes.append(slope.tolist())
        biases.append(bias.tolist())
    kernel = np.exp(-np.arange(0, protocol["filter"]["tau_ms"], protocol["bin_ms"], dtype=float)
                    / protocol["filter"]["tau_ms"])
    kernel /= kernel.sum()
    source = "// Generated from pinned human-data model and serialized compact patches.\n"
    source += "// Offline research only. Input: reset MODEL_ID, then one row of neural counts per20ms bin.\n"
    for index, model in enumerate(models):
        source += f"// Model {index}: obfuscated day {model['day']}, {model['method']}, {model['budget_seconds']}s calibration.\n"
    source += f"const CHANNELS: usize = {len(decoder.mean)};\nconst TAPS: usize = {len(kernel)};\nconst MODELS: usize = {len(models)};\n"
    source += f"const KERNEL: [f64; TAPS] = {json.dumps(kernel.tolist())};\n"
    source += f"static WEIGHTS: [[[f64; 7]; CHANNELS]; MODELS] = {json.dumps(slopes)};\n"
    source += f"static BIAS: [[f64; 7]; MODELS] = {json.dumps(biases)};\n"
    destination.write_text(source + RUNTIME)
    stream = io.StringIO()
    for model_id, counts, _ in cases:
        stream.write(f"reset {model_id}\n")
        np.savetxt(stream, counts, fmt="%.17g")
    input_text = stream.getvalue()
    expected = np.concatenate([predicted for _, _, predicted in cases])
    with tempfile.TemporaryDirectory(prefix="human-recovery-") as directory:
        executable = Path(directory) / "decoder"
        subprocess.run(["rustc", "--edition=2021", "-O", str(destination), "-o", str(executable)], check=True)
        start = perf_counter()
        run = subprocess.run([str(executable)], input=input_text, capture_output=True, text=True, check=True)
        milliseconds = (perf_counter() - start) * 1000
    actual = np.fromstring(run.stdout, sep=" ").reshape(-1, 7)
    if actual.shape != expected.shape:
        raise ValueError(f"Compiled decoder output shape {actual.shape} != {expected.shape}")
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(f"Compiled decoder differs from Python: maximum absolute error {error}")
    return {"bin_count": len(expected), "recording_count": len(cases), "max_absolute_error": error,
            "tolerance": {"rtol": 1e-10, "atol": 1e-10},
            "total_wall_ms": milliseconds,
            "wall_us_per_bin_including_io_and_startup": milliseconds * 1000 / len(expected),
            "source_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "rustc": subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True).stdout.strip(),
            "scope": "Numerical replay parity on all evaluation bins; not an all-input formal proof or clinical latency guarantee."}
