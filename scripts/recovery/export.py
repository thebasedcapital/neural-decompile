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
    model: usize,
    mean: [f64; FEATURES],
    variance: [f64; FEATURES],
    smoothed: [f64; 7],
    started: bool,
}
impl Decoder {
    fn new(model: usize) -> Self {
        Self { history: [[0.0; CHANNELS]; TAPS], cursor: 0, model, mean: ZSCORE_MEAN[model],
               variance: ZSCORE_VARIANCE[model], smoothed: [0.0; 7], started: false }
    }
    fn step(&mut self, input: [f64; CHANNELS]) -> [f64; 7] {
        self.history[self.cursor] = input;
        self.cursor = (self.cursor + 1) % TAPS;
        let mut output = BIAS[self.model];
        for (tau, kernel) in KERNELS.iter().enumerate() {
            let length = KERNEL_LENGTHS[tau];
            let mut rates = [0.0; CHANNELS];
            for lag in 0..length {
                let row = &self.history[(self.cursor + TAPS - 1 - lag) % TAPS];
                let weight = kernel[lag];
                for (rate, count) in rates.iter_mut().zip(row) { *rate += weight * count; }
            }
            let offset = tau * CHANNELS;
            for (channel, rate) in rates.iter().enumerate() {
                // Causal running z-score: label-free, updated with the current bin.
                let feature = offset + channel;
                let delta = rate - self.mean[feature];
                self.mean[feature] += ZSCORE_K * delta;
                self.variance[feature] = (1.0 - ZSCORE_K) * (self.variance[feature] + ZSCORE_K * delta * delta);
                let z = (rate - self.mean[feature]) / self.variance[feature].max(1e-8).sqrt();
                let row = &WEIGHTS[self.model][feature];
                for (value, coefficient) in output.iter_mut().zip(row) { *value += z * coefficient; }
            }
        }
        // Causal exponential smoothing; resets on decoder construction.
        let updated = if self.started {
            let mut updated = [0.0f64; 7];
            for (index, value) in output.iter().enumerate() {
                updated[index] = SMOOTH_BETA * value + (1.0 - SMOOTH_BETA) * self.smoothed[index];
            }
            updated
        } else {
            output
        };
        self.smoothed = updated;
        self.started = true;
        updated
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());
    let mut line = String::with_capacity(CHANNELS * 24);
    let mut decoder = Decoder::new(0);
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 { break; }
        if line.trim().is_empty() { continue; }
        if let Some(rest) = line.trim().strip_prefix("reset ") {
            let model = rest.parse::<usize>()?;
            if model >= MODELS { return Err("model index out of range".into()); }
            decoder = Decoder::new(model);
            continue;
        }
        let mut fields = line.split_whitespace();
        let mut counts = [0.0; CHANNELS];
        for count in &mut counts {
            *count = fields.next().ok_or("too few neural channels")?.parse::<f64>()?;
            if !count.is_finite() || *count < 0.0 { return Err("invalid neural count".into()); }
        }
        if fields.next().is_some() { return Err("too many neural channels".into()); }
        let values = decoder.step(counts);
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
    taus = protocol["filter"]["taus_ms"]
    bin_ms = protocol["bin_ms"]
    kernels, lengths = [], []
    for tau in taus:
        kernel = np.exp(-np.arange(0, tau, bin_ms, dtype=float) / tau)
        kernel /= kernel.sum()
        kernels.append(kernel.tolist())
        lengths.append(len(kernel))
    taps = max(lengths)
    padded = [row + [0.0] * (taps - len(row)) for row in kernels]
    beta = protocol["output_smoothing"]["beta"]
    k = bin_ms / 1000 / protocol["front_end"]["tau_seconds"]
    source = "// Generated from pinned human-data model and serialized compact patches.\n"
    source += "// Offline research only. Input: reset MODEL_ID, then one row of neural counts per 20 ms bin.\n"
    for index, model in enumerate(models):
        source += f"// Model {index}: obfuscated day {model['day']}, {model['method']}, {model['budget_seconds']}s calibration.\n"
    source += f"const CHANNELS: usize = {cases[0][1].shape[1]};\nconst TAPS: usize = {taps};\nconst MODELS: usize = {len(models)};\n"
    source += f"const KERNELS: [[f64; TAPS]; {len(taus)}] = {json.dumps(padded)};\n"
    source += f"const KERNEL_LENGTHS: [usize; {len(taus)}] = {json.dumps(lengths)};\n"
    source += f"const SMOOTH_BETA: f64 = {beta!r};\n"
    source += f"const ZSCORE_K: f64 = {k!r};\n"
    source += f"const FEATURES: usize = {len(decoder.mean)};\n"
    source += f"static WEIGHTS: [[[f64; 7]; FEATURES]; MODELS] = {json.dumps(slopes)};\n"
    source += f"static BIAS: [[f64; 7]; MODELS] = {json.dumps(biases)};\n"
    source += f"static ZSCORE_MEAN: [[f64; FEATURES]; MODELS] = {json.dumps([m['zscore_mean'] for m in models])};\n"
    source += f"static ZSCORE_VARIANCE: [[f64; FEATURES]; MODELS] = {json.dumps([m['zscore_variance'] for m in models])};\n"
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
