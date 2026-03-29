mod weights;
mod quantize;
mod emit;
mod verify;
mod fsm;

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
    }
}
