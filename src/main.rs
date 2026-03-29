mod weights;
mod quantize;
mod emit;
mod verify;
mod fsm;
mod gguf;
mod trace;
mod diagnose;

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

        /// Quantization epsilon (weights within eps of integer get snapped)
        #[arg(short, long, default_value = "0.15")]
        eps: f64,

        /// Output format: python, rust, table
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
            let rnn = weights::load_rnn_weights(&input)?;
            let quantized = quantize::quantize_rnn(&rnn, eps);
            let stats = quantize::weight_stats(&quantized);

            eprintln!("Loaded: hidden_dim={}, input_dim={}, output_dim={}",
                     rnn.hidden_dim, rnn.input_dim, rnn.output_dim);
            eprintln!("Quantized: {:.0}% integer weights (eps={})",
                     stats.pct_integer * 100.0, eps);
            eprintln!("Dead neurons: {:?}", stats.dead_neurons);

            let code = match format.as_str() {
                "python" => emit::emit_python(&quantized, "decompiled"),
                "rust" => emit::emit_rust(&quantized, "decompiled"),
                "table" => emit::emit_table(&quantized),
                _ => anyhow::bail!("Unknown format: {}", format),
            };

            match output {
                Some(path) => std::fs::write(&path, &code)?,
                None => print!("{}", code),
            }
            Ok(())
        }

        Commands::Verify { input, tests, eps } => {
            let rnn = weights::load_rnn_weights(&input)?;
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
            Ok(())
        }

        Commands::Stats { input, eps } => {
            let rnn = weights::load_rnn_weights(&input)?;
            let quantized = quantize::quantize_rnn(&rnn, eps);
            let stats = quantize::weight_stats(&quantized);

            println!("Neural Decompiler — Weight Statistics");
            println!("=====================================");
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

        Commands::Trace { input, sequence, eps, raw } => {
            let rnn = weights::load_rnn_weights(&input)?;
            let quantized = quantize::quantize_rnn(&rnn, eps);

            // Parse input sequence: "1,0,1,1" → one-hot vectors
            let bits: Vec<u8> = sequence.split(',')
                .map(|s| s.trim().parse::<u8>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("Bad sequence (expected comma-separated 0/1): {}", e))?;

            let input_vecs: Vec<Vec<f64>> = bits.iter().map(|&b| {
                if rnn.input_dim == 2 {
                    // One-hot: [1,0] for bit=0, [0,1] for bit=1
                    if b == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
                } else if rnn.input_dim == 1 {
                    vec![b as f64]
                } else {
                    // For multi-input, treat as one-hot index
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

            println!("── Quantized (eps={}) ──", eps);
            let quant_trace = trace::trace_quantized(&quantized, &input_vecs);
            print!("{}", trace::format_trace(&quant_trace));

            Ok(())
        }

        Commands::Diagnose { input, tests, eps } => {
            let rnn = weights::load_rnn_weights(&input)?;
            let quantized = quantize::quantize_rnn(&rnn, eps);
            let test_cases = verify::load_test_cases(&tests)?;
            let report = diagnose::run_diagnosis(&rnn, &quantized, &test_cases);
            print!("{}", diagnose::format_diagnosis(&report));
            Ok(())
        }

        Commands::MakeTestGguf { output } => {
            gguf::create_test_gguf(&output)?;
            println!("Created test GGUF: {}", output.display());
            Ok(())
        }
    }
}
