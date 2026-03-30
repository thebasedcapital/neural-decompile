mod weights;
mod quantize;
mod emit;
mod verify;
mod fsm;
mod gguf;
mod trace;
mod diagnose;
mod compare;
mod visualize;
mod taxonomy;
mod diff;
mod evolve;
mod patch;
mod slice;
mod transformer;
mod xray;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nd", about = "Neural Decompiler — auto-decompile neural nets into readable code")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Decompile weight matrices into readable code
    Decompile {
        /// Path to weight file (JSON format)
        #[arg()]
        input: PathBuf,

        /// Quantization epsilon (weights within eps of integer get snapped). Default: 0.15 for RNNs, 0.01 for transformers
        #[arg(short, long)]
        eps: Option<f64>,

        /// Output format: python, rust, rust-kani, table, circuit
        #[arg(short, long, default_value = "python")]
        format: String,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Verify decompiled FSM against test cases
    Verify {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Path to test cases (JSON: [{inputs: [[...]], expected: N}, ...])
        #[arg()]
        tests: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// Show weight statistics (% integer, dead neurons, sparsity)
    Stats {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// List all tensor names and shapes in a GGUF file
    Layers {
        /// Path to GGUF file
        #[arg()]
        input: PathBuf,
    },

    /// Extract a single tensor from a GGUF file as JSON
    Extract {
        /// Path to GGUF file
        #[arg()]
        input: PathBuf,

        /// Tensor name to extract
        #[arg(short, long)]
        tensor: String,

        /// Output file (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Trace hidden state evolution on a specific input
    Trace {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Input sequence as comma-separated bits (e.g. "1,0,1,1")
        #[arg()]
        sequence: String,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,

        /// Also show raw (unquantized) trace side-by-side
        #[arg(long)]
        raw: bool,

        /// Output as HTML heatmap (opens in browser)
        #[arg(long)]
        html: bool,
    },

    /// Compare two decompiled circuits (detect complements, shared structure)
    Compare {
        /// First weight file
        #[arg()]
        a: PathBuf,

        /// Second weight file
        #[arg()]
        b: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// Build a taxonomy of circuit families from a directory of weight files
    Taxonomy {
        /// Directory containing weight JSON files
        #[arg()]
        dir: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// Diagnose why quantization fails on specific test cases
    Diagnose {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Path to test cases
        #[arg()]
        tests: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// Isolate the active circuit — remove neurons that never fire or can't reach output
    Slice {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Path to test cases (JSON) — traces all inputs to find active neurons
        #[arg()]
        tests: Option<PathBuf>,

        /// Single input sequence (e.g. "1,0,1,1") — slice for one specific input
        #[arg(long)]
        sequence: Option<String>,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,

        /// Immediately decompile the sliced circuit (python/rust/table)
        #[arg(long)]
        emit: Option<String>,

        /// Save sliced weights to JSON
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Full circuit X-ray — stats, slice, hybrid decomposition, traces, HTML report
    Xray {
        /// Path to weight file
        #[arg()]
        input: PathBuf,

        /// Path to test cases (enables verification + slice + smart trace selection)
        #[arg()]
        tests: Option<PathBuf>,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,

        /// Output as HTML (opens in browser)
        #[arg(long)]
        html: bool,
    },

    /// Watch a circuit evolve during training — load epoch snapshots and visualize crystallization
    Evolve {
        /// Directory containing epoch_*.json snapshots
        #[arg()]
        dir: PathBuf,

        /// Path to test cases (enables FSM verification per frame)
        #[arg(long)]
        tests: Option<PathBuf>,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,

        /// Output as HTML animation (opens in browser)
        #[arg(long)]
        html: bool,
    },

    /// Semantic diff between two circuits (weight-by-weight delta)
    Diff {
        /// First weight file (before)
        #[arg()]
        a: PathBuf,

        /// Second weight file (after)
        #[arg()]
        b: PathBuf,

        /// Quantization epsilon
        #[arg(short, long, default_value = "0.15")]
        eps: f64,
    },

    /// Compile a decompiled Python program back into weight JSON
    Patch {
        /// Path to decompiled .py file
        #[arg()]
        input: PathBuf,

        /// Output JSON file (stdout if omitted)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Create a synthetic test GGUF file
    #[command(name = "make-test-gguf")]
    MakeTestGguf {
        /// Output path
        #[arg(default_value = "test.gguf")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Decompile { input, eps, format, output } => {
            let program = weights::load_neural_program(&input)?;

            // Auto-select epsilon based on model type if not specified
            let eps = eps.unwrap_or_else(|| {
                match &program {
                    weights::NeuralProgram::Rnn(_) => 0.15,
                    weights::NeuralProgram::Transformer(_) => 0.01,
                }
            });

            let code = match &program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(rnn, eps);
                    let stats = quantize::weight_stats(&quantized);

                    eprintln!("Loaded RNN: hidden_dim={}, input_dim={}, output_dim={}",
                             rnn.hidden_dim, rnn.input_dim, rnn.output_dim);
                    eprintln!("Quantized: {:.0}% integer weights (eps={})",
                             stats.pct_integer * 100.0, eps);
                    eprintln!("Dead neurons: {:?}", stats.dead_neurons);

                    match format.as_str() {
                        "python" => emit::emit_python(&quantized, "decompiled"),
                        "rust" => emit::emit_rust(&quantized, "decompiled"),
                        "rust-kani" => emit::emit_rust_kani(&quantized, "decompiled"),
                        "table" => emit::emit_table(&quantized),
                        _ => anyhow::bail!("Unknown format: {}", format),
                    }
                }
                weights::NeuralProgram::Transformer(t) => {
                    let quantized = quantize::quantize_transformer(t, eps);
                    let stats = quantize::transformer_stats(&quantized);

                    eprintln!("Loaded Transformer: {} layers, d_model={}, vocab_size={}",
                             t.n_layers, t.d_model, t.vocab_size);
                    eprintln!("Quantized: {:.0}% integer weights (eps={})",
                             stats.pct_integer * 100.0, eps);

                    match format.as_str() {
                        "python" => emit::emit_transformer_python(&quantized, "decompiled"),
                        "rust" => emit::emit_transformer_rust(&quantized, "decompiled"),
                        "table" => emit::emit_transformer_table(&quantized),
                        "circuit" => emit::emit_transformer_circuit(&quantized, "decompiled"),
                        _ => anyhow::bail!("Unknown format for transformer: {} (supported: python, rust, table, circuit)", format),
                    }
                }
            };

            match output {
                Some(path) => std::fs::write(&path, &code)?,
                None => print!("{}", code),
            }
            Ok(())
        }

        Commands::Verify { input, tests, eps } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, eps);
                    let test_cases = verify::load_test_cases(&tests)?;
                    let results = verify::run_verification(&quantized, &test_cases);

                    println!("Verification: {}/{} passed ({:.0}%)",
                            results.passed, results.total,
                            results.passed as f64 / results.total as f64 * 100.0);
                    for fail in &results.failures {
                        println!("  FAIL: input={:?} expected={} got={}",
                                fail.input, fail.expected, fail.got);
                    }

                    if results.passed == results.total {
                        println!("✓ PERFECT — decompiled FSM matches all test cases");
                    }
                }
                weights::NeuralProgram::Transformer(t) => {
                    // Load transformer tests
                    let data = std::fs::read_to_string(&tests)?;
                    let test_cases: Vec<verify::TransformerTest> = serde_json::from_str(&data)?;

                    // Verify using minimal quantization to preserve embeddings
                    // (embeddings have small values that get snapped to 0 with eps=0.15)
                    let quantized = quantize::quantize_transformer(&t, 0.001);
                    let results = verify::verify_decompiled_transformer(&t, &quantized, &test_cases);

                    println!("Transformer Verification: {}/{} passed ({:.0}%)",
                            results.passed, results.total,
                            results.passed as f64 / results.total as f64 * 100.0);
                    for fail in &results.failures {
                        println!("  FAIL: tokens={:?} expected={} got={} logits={:.3?}",
                                fail.tokens, fail.expected, fail.got,
                                fail.logits.iter().take(5).collect::<Vec<_>>());
                    }

                    if results.passed == results.total {
                        println!("✓ PERFECT — quantized transformer matches original");
                    }
                }
            }
            Ok(())
        }

        Commands::Stats { input, eps } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, eps);
                    let stats = quantize::weight_stats(&quantized);

                    println!("Neural Decompiler — Weight Statistics (RNN)");
                    println!("===========================================");
                    println!("Hidden dim:    {}", rnn.hidden_dim);
                    println!("Input dim:     {}", rnn.input_dim);
                    println!("Output dim:    {}", rnn.output_dim);
                    println!("Total weights: {}", stats.total_weights);
                    println!("Integer (eps={}): {}/{} ({:.0}%)",
                            eps, stats.integer_count, stats.total_weights,
                            stats.pct_integer * 100.0);
                    println!("Zero weights:  {}/{} ({:.0}%)",
                            stats.zero_count, stats.total_weights,
                            stats.zero_count as f64 / stats.total_weights as f64 * 100.0);
                    println!("Dead neurons:  {:?}", stats.dead_neurons);
                    println!("Max weight:    {:.4}", stats.max_abs);
                    println!("L∞ to int:     {:.4}", stats.linf_to_int);
                }
                weights::NeuralProgram::Transformer(t) => {
                    use crate::transformer::{Transformer, TransformerBlock};

                    // Calculate total parameters
                    let emb_params = t.vocab_size * t.d_model + t.max_seq_len * t.d_model;
                    let layer_params: usize = t.layers.iter().map(|l| {
                        // Attention: Q, K, V, O projections (each d_model × d_model)
                        let attn = 4 * t.d_model * t.d_model;
                        // FFN: w_in (d_model × d_ff) + w_out (d_ff × d_model)
                        let ffn = t.d_model * l.d_ff + l.d_ff * t.d_model;
                        // LayerNorm: 4 * d_model (gamma/beta for pre/post)
                        let ln = 4 * t.d_model;
                        attn + ffn + ln
                    }).sum();
                    let out_params = t.d_model * t.vocab_size;
                    let total = emb_params + layer_params + out_params;

                    println!("Neural Decompiler — Weight Statistics (Transformer)");
                    println!("=====================================================");
                    println!("Architecture:  {} layers, d_model={}, n_heads={}",
                             t.n_layers, t.d_model,
                             t.layers.first().map(|l| l.n_heads).unwrap_or(0));
                    println!("Vocab size:    {}", t.vocab_size);
                    println!("Max seq len:   {}", t.max_seq_len);
                    println!("Total params:  ~{:.2}M", total as f64 / 1e6);
                    println!("\nPer-layer breakdown:");
                    for (i, l) in t.layers.iter().enumerate() {
                        let attn = 4 * t.d_model * t.d_model;
                        let ffn = t.d_model * l.d_ff + l.d_ff * t.d_model;
                        println!("  Layer {}: attention={:.2}K params, FFN={:.2}K params, gelu={}",
                                i, attn as f64 / 1e3, ffn as f64 / 1e3, l.gelu);
                    }
                }
            }
            Ok(())
        }

        Commands::Layers { input } => {
            let gf = gguf::GgufFile::open(&input)?;
            gf.print_layers();
            Ok(())
        }

        Commands::Extract { input, tensor, output } => {
            let gf = gguf::GgufFile::open(&input)?;
            let data = gf.extract_f32(&tensor)?;

            let info = gf.find_tensor(&tensor).unwrap();
            eprintln!("Extracted '{}' {} {} — {} elements",
                     tensor, info.dtype.name(), info.shape_str(), data.len());

            let json = serde_json::to_string(&data)?;
            match output {
                Some(path) => std::fs::write(&path, &json)?,
                None => println!("{}", json),
            }
            Ok(())
        }

        Commands::Compare { a, b, eps } => {
            let prog_a = weights::load_neural_program(&a)?;
            let prog_b = weights::load_neural_program(&b)?;

            match (prog_a, prog_b) {
                (weights::NeuralProgram::Rnn(rnn_a), weights::NeuralProgram::Rnn(rnn_b)) => {
                    let qa = quantize::quantize_rnn(&rnn_a, eps);
                    let qb = quantize::quantize_rnn(&rnn_b, eps);
                    let result = compare::compare(&qa, &qb);
                    print!("{}", compare::format_compare(&result));
                }
                (weights::NeuralProgram::Transformer(t_a), weights::NeuralProgram::Transformer(t_b)) => {
                    let qa = quantize::quantize_transformer(&t_a, eps);
                    let qb = quantize::quantize_transformer(&t_b, eps);
                    let result = compare::compare_transformers(&qa, &qb);
                    print!("{}", compare::format_transformer_compare(&result));
                }
                _ => {
                    anyhow::bail!("Cannot compare RNN with Transformer — different architectures");
                }
            }
            Ok(())
        }

        Commands::Taxonomy { dir, eps } => {
            let mut circuits = Vec::new();
            let mut tf_circuits = Vec::new();
            let mut entries: Vec<_> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.ends_with(".json") && !name.contains("_tests")
                })
                .collect();
            entries.sort_by_key(|e| e.file_name());

            for entry in &entries {
                let path = entry.path();
                let name = path.file_stem().unwrap().to_string_lossy().to_string();
                match weights::load_neural_program(&path) {
                    Ok(weights::NeuralProgram::Rnn(rnn)) => {
                        let q = quantize::quantize_rnn(&rnn, eps);
                        circuits.push((name, q));
                    }
                    Ok(weights::NeuralProgram::Transformer(t)) => {
                        let q = quantize::quantize_transformer(&t, eps);
                        tf_circuits.push((name, q));
                    }
                    Err(e) => eprintln!("  skip {}: {}", path.display(), e),
                }
            }

            if !circuits.is_empty() {
                eprintln!("Loaded {} RNN circuits from {}", circuits.len(), dir.display());
                let pairs = taxonomy::build_taxonomy(&circuits);
                print!("{}", taxonomy::format_taxonomy(&circuits, &pairs));
            }
            if !tf_circuits.is_empty() {
                eprintln!("Loaded {} Transformer circuits from {}", tf_circuits.len(), dir.display());
                eprintln!("Transformer taxonomy not yet implemented");
            }
            Ok(())
        }

        Commands::Trace { input, sequence, eps, raw, html } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, eps);

                    // Parse input sequence: "1,0,1,1" → one-hot vectors
                    let bits: Vec<u8> = sequence.split(',')
                        .map(|s| s.trim().parse::<u8>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| anyhow::anyhow!("Bad sequence (expected comma-separated 0/1): {}", e))?;

                    let input_vecs: Vec<Vec<f64>> = bits.iter().map(|&b| {
                        if rnn.input_dim == 2 {
                            if b == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
                        } else if rnn.input_dim == 1 {
                            vec![b as f64]
                        } else {
                            let mut v = vec![0.0; rnn.input_dim];
                            if (b as usize) < rnn.input_dim { v[b as usize] = 1.0; }
                            v
                        }
                    }).collect();

                    if raw {
                        println!("── Raw (unquantized) ──");
                        let raw_trace = trace::trace_raw(&rnn, &input_vecs);
                        print!("{}", trace::format_trace(&raw_trace));
                        println!();
                    }

                    let quant_trace = trace::trace_quantized(&quantized, &input_vecs);

                    if html {
                        let title = input.file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "trace".to_string());
                        let html_content = visualize::trace_to_html(&quant_trace, &title);
                        let path = std::env::temp_dir().join("nd-trace.html");
                        std::fs::write(&path, &html_content)?;
                        eprintln!("Wrote: {}", path.display());
                        std::process::Command::new("open").arg(&path).spawn()?;
                    } else {
                        println!("── Quantized (eps={}) ──", eps);
                        print!("{}", trace::format_trace(&quant_trace));
                    }
                }
                weights::NeuralProgram::Transformer(t) => {
                    // Parse tokens: comma-separated integers
                    let tokens: Vec<usize> = sequence.split(',')
                        .map(|s| s.trim().parse::<usize>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| anyhow::anyhow!("Bad token sequence: {}", e))?;

                    if tokens.is_empty() || tokens.len() > t.max_seq_len {
                        anyhow::bail!("Sequence length {} out of range (1-{})", tokens.len(), t.max_seq_len);
                    }

                    let tf_trace = trace::trace_transformer(&t, &tokens);

                    if html {
                        // TODO: Add HTML visualization for transformer trace
                        eprintln!("HTML output not yet implemented for transformers, using text format");
                    }
                    print!("{}", trace::format_transformer_trace(&tf_trace));
                }
            }

            Ok(())
        }

        Commands::Diagnose { input, tests, eps } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, eps);
                    let test_cases = verify::load_test_cases(&tests)?;
                    let report = diagnose::run_diagnosis(&rnn, &quantized, &test_cases);
                    print!("{}", diagnose::format_diagnosis(&report));
                }
                weights::NeuralProgram::Transformer(t) => {
                    let test_cases: Vec<verify::TransformerTest> = {
                        let data = std::fs::read_to_string(&tests)?;
                        serde_json::from_str(&data)?
                    };
                    let failures = diagnose::diagnose_transformer(&t, &test_cases);
                    if failures.is_empty() {
                        println!("✓ All tests pass — no quantization issues detected");
                    } else {
                        print!("{}", diagnose::format_transformer_diagnosis(&failures));
                    }
                }
            }
            Ok(())
        }

        Commands::Slice { input, tests, sequence, eps, emit: emit_fmt, output } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, eps);

                    let result = if let Some(seq) = sequence {
                        // Single sequence mode
                        let bits: Vec<u8> = seq.split(',')
                            .map(|s| s.trim().parse::<u8>())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| anyhow::anyhow!("Bad sequence: {}", e))?;

                        let input_vecs: Vec<Vec<f64>> = bits.iter().map(|&b| {
                            if rnn.input_dim == 2 {
                                if b == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
                            } else if rnn.input_dim == 1 {
                                vec![b as f64]
                            } else {
                                let mut v = vec![0.0; rnn.input_dim];
                                if (b as usize) < rnn.input_dim { v[b as usize] = 1.0; }
                                v
                            }
                        }).collect();

                        let traces = vec![trace::trace_quantized(&quantized, &input_vecs)];
                        slice::slice_from_traces(&quantized, &traces)
                    } else if let Some(tests_path) = tests {
                        let test_cases = verify::load_test_cases(&tests_path)?;
                        slice::slice_from_tests(&quantized, &test_cases)
                    } else {
                        anyhow::bail!("Provide either test cases or --sequence");
                    };

                    // Print the slice report
                    print!("{}", slice::format_slice(&result));

                    // Optionally emit decompiled code from the sliced circuit
                    if let Some(fmt) = emit_fmt {
                        println!();
                        let code = match fmt.as_str() {
                            "python" => emit::emit_python(&result.circuit, "sliced"),
                            "rust" => emit::emit_rust(&result.circuit, "sliced"),
                            "rust-kani" => emit::emit_rust_kani(&result.circuit, "sliced"),
                            "table" => emit::emit_table(&result.circuit),
                            _ => anyhow::bail!("Unknown format: {}", fmt),
                        };
                        print!("{}", code);
                    }

                    // Optionally save sliced weights
                    if let Some(out_path) = output {
                        let json = serde_json::json!({
                            "W_hh": (0..result.circuit.hidden_dim).map(|i|
                                (0..result.circuit.hidden_dim).map(|j|
                                    result.circuit.w_hh[[i, j]]
                                ).collect::<Vec<_>>()
                            ).collect::<Vec<_>>(),
                            "W_hx": (0..result.circuit.hidden_dim).map(|i|
                                (0..result.circuit.input_dim).map(|j|
                                    result.circuit.w_hx[[i, j]]
                        ).collect::<Vec<_>>()
                    ).collect::<Vec<_>>(),
                    "b_h": result.circuit.b_h,
                    "W_y": (0..result.circuit.output_dim).map(|i|
                        (0..result.circuit.hidden_dim).map(|j|
                            result.circuit.w_y[[i, j]]
                        ).collect::<Vec<_>>()
                    ).collect::<Vec<_>>(),
                    "b_y": result.circuit.b_y,
                });
                std::fs::write(&out_path, serde_json::to_string_pretty(&json)?)?;
                eprintln!("Saved sliced weights: {}", out_path.display());
            }
                }
                weights::NeuralProgram::Transformer(t) => {
                    // Parse token sequences from test file or sequence argument
                    let token_sequences: Vec<Vec<usize>> = if let Some(seq) = sequence {
                        vec![seq.split(',')
                            .map(|s| s.trim().parse::<usize>())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| anyhow::anyhow!("Bad token sequence: {}", e))?]
                    } else if let Some(tests_path) = tests {
                        let data = std::fs::read_to_string(&tests_path)?;
                        let test_cases: Vec<verify::TransformerTest> = serde_json::from_str(&data)?;
                        test_cases.iter().map(|tc| tc.tokens.clone()).collect()
                    } else {
                        anyhow::bail!("Provide either test cases or --sequence");
                    };

                    let result = slice::slice_transformer(&t, &token_sequences);
                    print!("{}", slice::format_transformer_slice(&result));

                    // Transformer slice doesn't support --emit or --output yet
                    if emit_fmt.is_some() || output.is_some() {
                        eprintln!("Note: --emit and --output not yet implemented for transformer slicing");
                    }
                }
            }

            Ok(())
        }

        Commands::Xray { input, tests, eps: _, html: html_flag } => {
            let program = weights::load_neural_program(&input)?;

            match program {
                weights::NeuralProgram::Rnn(rnn) => {
                    let quantized = quantize::quantize_rnn(&rnn, 0.15);
                    let name = input.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "circuit".to_string());

                    let test_cases = match tests {
                        Some(ref path) => Some(verify::load_test_cases(path)?),
                        None => None,
                    };

                    let report = xray::run_xray(&rnn, &quantized, &name, test_cases.as_deref());

                    if html_flag {
                        let html_content = xray::render_html(&report);
                        let path = std::env::temp_dir().join(format!("nd-xray-{}.html", name));
                        std::fs::write(&path, &html_content)?;
                        eprintln!("Wrote: {}", path.display());
                        std::process::Command::new("open").arg(&path).spawn()?;
                    } else {
                        print!("{}", xray::format_xray(&report));
                    }
                }
                weights::NeuralProgram::Transformer(t) => {
                    let name = input.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "transformer".to_string());

                    let test_cases: Option<Vec<verify::TransformerTest>> = match tests {
                        Some(ref path) => {
                            let data = std::fs::read_to_string(path)?;
                            Some(serde_json::from_str(&data)?)
                        }
                        None => None,
                    };

                    let report = xray::run_transformer_xray(&t, &name, test_cases.as_deref());

                    if html_flag {
                        let html_content = xray::render_transformer_html(&report);
                        let path = std::env::temp_dir().join(format!("nd-xray-{}.html", name));
                        std::fs::write(&path, &html_content)?;
                        eprintln!("Wrote: {}", path.display());
                        std::process::Command::new("open").arg(&path).spawn()?;
                    } else {
                        print!("{}", xray::format_transformer_xray(&report));
                    }
                }
            }

            Ok(())
        }

        Commands::Evolve { dir, tests, eps, html: html_flag } => {
            let snapshots = evolve::load_snapshots(&dir)?;
            if snapshots.is_empty() {
                anyhow::bail!("No epoch_*.json snapshots found in {}", dir.display());
            }
            eprintln!("Loaded {} snapshots from {}", snapshots.len(), dir.display());

            let test_cases = match tests {
                Some(ref path) => Some(verify::load_test_cases(path)?),
                None => {
                    let auto = dir.join("tests.json");
                    if auto.exists() {
                        eprintln!("Auto-detected: {}", auto.display());
                        Some(verify::load_test_cases(&auto)?)
                    } else {
                        None
                    }
                }
            };

            let name = dir.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "circuit".to_string());

            let report = evolve::analyze(&snapshots, test_cases.as_deref(), eps, &name);

            if html_flag {
                let html_content = evolve::render_html(&report);
                let path = std::env::temp_dir().join(format!("nd-evolve-{}.html", name));
                std::fs::write(&path, &html_content)?;
                eprintln!("Wrote: {}", path.display());
                std::process::Command::new("open").arg(&path).spawn()?;
            } else {
                print!("{}", evolve::format_evolve(&report));
            }

            Ok(())
        }

        Commands::Diff { a, b, eps } => {
            let prog_a = weights::load_neural_program(&a)?;
            let prog_b = weights::load_neural_program(&b)?;

            match (prog_a, prog_b) {
                (weights::NeuralProgram::Rnn(rnn_a), weights::NeuralProgram::Rnn(rnn_b)) => {
                    let qa = quantize::quantize_rnn(&rnn_a, eps);
                    let qb = quantize::quantize_rnn(&rnn_b, eps);
                    match diff::diff_circuits(&qa, &qb) {
                        Ok(d) => print!("{}", diff::format_diff(&d)),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                }
                (weights::NeuralProgram::Transformer(_), weights::NeuralProgram::Transformer(_)) => {
                    eprintln!("diff not yet implemented for transformers — use 'compare' instead");
                }
                _ => {
                    anyhow::bail!("Cannot diff RNN with Transformer — different architectures");
                }
            }
            Ok(())
        }

        Commands::Patch { input, output } => {
            patch::patch_file(&input, output.as_deref())?;
            Ok(())
        }

        Commands::MakeTestGguf { output } => {
            gguf::create_test_gguf(&output)?;
            println!("Created test GGUF: {}", output.display());
            Ok(())
        }
    }
}
