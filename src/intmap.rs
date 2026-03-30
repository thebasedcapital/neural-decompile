use anyhow::Result;
use std::path::Path;

use crate::gguf::GgufFile;

/// Per-tensor integer structure analysis
#[derive(Debug, Clone)]
pub struct TensorIntMap {
    pub name: String,
    pub shape: String,
    pub dtype: String,
    pub n_elements: usize,
    /// Fraction of weights within eps of an integer (raw, absolute)
    pub pct_near_int: f64,
    /// Fraction of weights that are exactly zero (dead)
    pub pct_zero: f64,
    /// Fraction within eps of {-1, 0, 1} (ternary)
    pub pct_ternary: f64,
    /// Mean absolute distance to nearest integer
    pub mean_dist_to_int: f64,
    /// Max absolute weight value
    pub max_abs: f64,
    /// Histogram of nearest-integer values (only counts weights within eps)
    pub int_histogram: Vec<(i64, usize)>,
    /// Per-row integeriness (for matrix tensors): (row_idx, pct_near_int)
    pub row_hotspots: Vec<(usize, f64)>,
    /// Number of "fully integer" rows (>95% near-int)
    pub n_int_rows: usize,
    pub n_total_rows: usize,
    /// SCALE-AWARE: best discovered quantization unit and how well it fits
    pub best_unit: Option<QuantUnit>,
    /// Per-block entropy analysis (Q4_0/Q8_0 only)
    pub block_entropy: Option<BlockEntropy>,
}

/// Scale-aware integer structure: "do these weights snap to multiples of some unit?"
#[derive(Debug, Clone)]
pub struct QuantUnit {
    /// The discovered base unit (e.g., 0.0046 for Q4_0 scales)
    pub unit: f64,
    /// What fraction of non-zero weights are within eps*unit of an integer multiple of unit
    pub pct_on_grid: f64,
    /// How many distinct grid points are actually used
    pub n_grid_points: usize,
    /// The effective "integer range" — max(|round(w/unit)|)
    pub effective_range: i64,
    /// Ratio of used grid points to possible grid points (sparsity of the grid)
    pub grid_utilization: f64,
}

/// Entropy analysis: how many of the possible quantization levels does this tensor actually use?
#[derive(Debug, Clone)]
pub struct BlockEntropy {
    /// Average Shannon entropy per Q4_0 block (max = 4.0 bits for 16 uniform levels)
    pub mean_entropy: f64,
    /// Min entropy block (most structured)
    pub min_entropy: f64,
    /// Number of blocks with entropy < 2.0 (using ≤4 levels effectively)
    pub n_low_entropy_blocks: usize,
    pub n_total_blocks: usize,
    /// Global histogram: how many times each of the 16 Q4_0 levels appears
    pub level_histogram: [usize; 16],
    /// Effective number of levels used (2^entropy)
    pub effective_levels: f64,
}

/// Full scan result
pub struct IntMapReport {
    pub model_name: String,
    pub n_tensors: usize,
    pub n_scanned: usize,
    pub n_skipped: usize,
    pub tensors: Vec<TensorIntMap>,
    pub eps: f64,
}

/// Try to find the best quantization unit: a value `u` such that most weights are near `k*u` for integer k
fn find_quant_unit(data: &[f32], eps: f64) -> Option<QuantUnit> {
    // Skip if too few non-zero values
    let nonzero: Vec<f64> = data.iter()
        .map(|&w| w as f64)
        .filter(|&w| w.abs() > 1e-10)
        .collect();

    if nonzero.len() < 100 {
        return None;
    }

    // Strategy: find the smallest common interval
    // Take absolute values, sort, compute pairwise differences of nearby values
    let mut abs_vals: Vec<f64> = nonzero.iter().map(|w| w.abs()).collect();
    abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Sample differences between consecutive sorted values
    let mut diffs: Vec<f64> = Vec::new();
    let step = (abs_vals.len() / 5000).max(1);
    for i in (0..abs_vals.len() - 1).step_by(step) {
        let d = abs_vals[i + 1] - abs_vals[i];
        if d > 1e-10 {
            diffs.push(d);
        }
    }
    if diffs.is_empty() {
        return None;
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // The quantization unit is likely near the median of small differences
    // Also try: min nonzero abs value, and the scale factor from Q4_0 (data range / 15)
    let max_val = abs_vals.last().copied().unwrap_or(1.0);
    let candidates = vec![
        diffs[diffs.len() / 4],          // 25th percentile diff
        diffs[diffs.len() / 2],          // median diff
        abs_vals[0],                      // smallest nonzero |w|
        max_val / 7.0,                   // divide range by ~half of Q4 levels
        max_val / 15.0,                  // Q4_0: 16 levels, range / 15
        max_val / 31.0,                  // Q5: 32 levels
        max_val / 127.0,                 // Q8: 256 levels
    ];

    let mut best: Option<QuantUnit> = None;

    for unit in candidates {
        if unit < 1e-10 || unit > max_val {
            continue;
        }

        let mut on_grid = 0usize;
        let mut grid_points = std::collections::HashSet::new();
        let mut max_k: i64 = 0;

        for &w in &nonzero {
            let k = (w / unit).round();
            let residual = (w - k * unit).abs();
            if residual < eps * unit {
                on_grid += 1;
                grid_points.insert(k as i64);
                let abs_k = k.abs() as i64;
                if abs_k > max_k {
                    max_k = abs_k;
                }
            }
        }

        let pct = on_grid as f64 / nonzero.len() as f64 * 100.0;
        let effective_range = max_k;
        let possible_points = (2 * max_k + 1) as usize;
        let util = if possible_points > 0 {
            grid_points.len() as f64 / possible_points as f64
        } else {
            0.0
        };

        // Score: prefer high on-grid %, then low grid utilization (sparse = more structured)
        let score = pct * 1.0 + (1.0 - util) * 10.0;

        let is_better = match &best {
            None => true,
            Some(b) => {
                let b_score = b.pct_on_grid * 1.0 + (1.0 - b.grid_utilization) * 10.0;
                score > b_score
            }
        };

        if pct > 50.0 && is_better {
            best = Some(QuantUnit {
                unit,
                pct_on_grid: pct,
                n_grid_points: grid_points.len(),
                effective_range,
                grid_utilization: util,
            });
        }
    }

    best
}

fn analyze_tensor(name: &str, data: &[f32], shape: &str, dtype: &str, eps: f64, dims: &[u64]) -> TensorIntMap {
    let n = data.len();
    let mut near_int = 0usize;
    let mut zero = 0usize;
    let mut ternary = 0usize;
    let mut dist_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut int_counts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    for &w in data {
        let w = w as f64;
        let abs_w = w.abs();
        if abs_w > max_abs {
            max_abs = abs_w;
        }

        let nearest = w.round();
        let dist = (w - nearest).abs();
        dist_sum += dist;

        if dist < eps {
            near_int += 1;
            *int_counts.entry(nearest as i64).or_default() += 1;
        }

        if abs_w < eps {
            zero += 1;
        }

        // Ternary: within eps of -1, 0, or 1
        if (w - -1.0).abs() < eps || abs_w < eps || (w - 1.0).abs() < eps {
            ternary += 1;
        }
    }

    // Sort histogram by count descending
    let mut int_histogram: Vec<(i64, usize)> = int_counts.into_iter().collect();
    int_histogram.sort_by(|a, b| b.1.cmp(&a.1));
    int_histogram.truncate(20); // top 20

    // Scale-aware analysis: find the natural quantization unit
    let best_unit = find_quant_unit(data, eps);

    // Per-row analysis (if 2D matrix)
    let (row_hotspots, n_int_rows, n_total_rows) = if dims.len() >= 2 {
        let rows = dims[dims.len() - 1] as usize; // GGUF stores in reverse: [cols, rows]
        let cols = dims[0] as usize;
        if rows > 0 && cols > 0 && rows * cols == n {
            let mut hotspots = Vec::new();
            let mut int_rows = 0usize;
            for r in 0..rows {
                let start = r * cols;
                let end = start + cols;
                let row_data = &data[start..end];
                let row_near = row_data.iter()
                    .filter(|&&w| ((w as f64).round() - w as f64).abs() < eps)
                    .count();
                let pct = row_near as f64 / cols as f64 * 100.0;
                if pct > 90.0 {
                    hotspots.push((r, pct));
                    if pct > 95.0 {
                        int_rows += 1;
                    }
                }
            }
            hotspots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            hotspots.truncate(10);
            (hotspots, int_rows, rows)
        } else {
            (vec![], 0, 0)
        }
    } else {
        (vec![], 0, 0)
    };

    TensorIntMap {
        name: name.to_string(),
        shape: shape.to_string(),
        dtype: dtype.to_string(),
        n_elements: n,
        pct_near_int: near_int as f64 / n as f64 * 100.0,
        pct_zero: zero as f64 / n as f64 * 100.0,
        pct_ternary: ternary as f64 / n as f64 * 100.0,
        mean_dist_to_int: dist_sum / n as f64,
        max_abs,
        int_histogram,
        row_hotspots,
        n_int_rows,
        n_total_rows,
        best_unit,
        block_entropy: None, // filled in separately for Q4_0 tensors
    }
}

fn compute_block_entropy(blocks: &[[u8; 32]]) -> BlockEntropy {
    let mut global_hist = [0usize; 16];
    let mut min_entropy = f64::MAX;
    let mut entropy_sum = 0.0f64;
    let mut low_entropy_count = 0usize;

    for block in blocks {
        let mut block_hist = [0usize; 16];
        for &nib in block {
            block_hist[nib as usize] += 1;
            global_hist[nib as usize] += 1;
        }

        // Shannon entropy for this block
        let mut h = 0.0f64;
        for &count in &block_hist {
            if count > 0 {
                let p = count as f64 / 32.0;
                h -= p * p.log2();
            }
        }
        entropy_sum += h;
        if h < min_entropy {
            min_entropy = h;
        }
        if h < 2.0 {
            low_entropy_count += 1;
        }
    }

    let mean_entropy = if blocks.is_empty() { 0.0 } else { entropy_sum / blocks.len() as f64 };
    let effective_levels = 2.0f64.powf(mean_entropy);

    BlockEntropy {
        mean_entropy,
        min_entropy: if min_entropy == f64::MAX { 0.0 } else { min_entropy },
        n_low_entropy_blocks: low_entropy_count,
        n_total_blocks: blocks.len(),
        level_histogram: global_hist,
        effective_levels,
    }
}

/// Per-head entropy analysis for attention projection tensors.
/// Splits the Q4_0 blocks by head and computes per-head block entropy.
#[derive(Debug, Clone)]
pub struct HeadEntropy {
    pub head_idx: usize,
    pub mean_entropy: f64,
    pub min_entropy: f64,
    pub n_low_entropy_blocks: usize,
    pub n_blocks: usize,
    pub effective_levels: f64,
    /// Max absolute dequantized weight in this head
    pub max_abs: f64,
}

/// Compute per-head entropy by splitting nibble blocks according to head layout.
/// Returns None if the tensor isn't an attention projection or head count can't be inferred.
fn per_head_entropy(
    gf: &GgufFile,
    tensor_name: &str,
    n_heads: usize,
) -> Option<Vec<HeadEntropy>> {
    let info = gf.find_tensor(tensor_name)?;
    if info.dtype.name() != "Q4_0" || info.dims.len() < 2 {
        return None;
    }

    // GGUF dims are reversed: dims[0]=cols(d_model), dims[1]=rows(out_dim)
    let cols = info.dims[0] as usize;
    let rows = info.dims[1] as usize;
    if rows == 0 || n_heads == 0 || rows % n_heads != 0 {
        return None;
    }
    let head_dim = rows / n_heads;
    let blocks_per_row = cols / 32; // each Q4_0 block = 32 values

    let all_blocks = gf.extract_q4_0_nibbles(tensor_name).ok()?;
    let all_f32 = gf.extract_f32(tensor_name).ok()?;

    let mut heads = Vec::with_capacity(n_heads);

    for h in 0..n_heads {
        let row_start = h * head_dim;
        let row_end = row_start + head_dim;

        // Collect blocks belonging to this head's rows
        let mut head_blocks = Vec::new();
        for r in row_start..row_end {
            let block_start = r * blocks_per_row;
            let block_end = block_start + blocks_per_row;
            if block_end <= all_blocks.len() {
                head_blocks.extend_from_slice(&all_blocks[block_start..block_end]);
            }
        }

        // Compute entropy for this head's blocks
        let be = compute_block_entropy(&head_blocks);

        // Max abs weight for this head
        let val_start = row_start * cols;
        let val_end = row_end * cols;
        let max_abs = if val_end <= all_f32.len() {
            all_f32[val_start..val_end]
                .iter()
                .map(|w| w.abs())
                .fold(0.0f32, f32::max) as f64
        } else {
            0.0
        };

        heads.push(HeadEntropy {
            head_idx: h,
            mean_entropy: be.mean_entropy,
            min_entropy: be.min_entropy,
            n_low_entropy_blocks: be.n_low_entropy_blocks,
            n_blocks: be.n_total_blocks,
            effective_levels: be.effective_levels,
            max_abs,
        });
    }

    Some(heads)
}

/// Try to infer the number of attention heads from GGUF metadata or tensor shapes.
fn infer_n_heads(gf: &GgufFile) -> Option<usize> {
    // Try standard GGUF metadata keys
    for key in &[
        "llama.attention.head_count",
        "gpt2.attention.head_count",
        "mpt.attention.head_count",
        "falcon.attention.head_count",
        "qwen2.attention.head_count",
        "phi2.attention.head_count",
        "gemma.attention.head_count",
    ] {
        if let Some(val) = gf.metadata.get(*key) {
            return match val {
                crate::gguf::MetaValue::Uint32(n) => Some(*n as usize),
                crate::gguf::MetaValue::Int32(n) => Some(*n as usize),
                crate::gguf::MetaValue::Uint64(n) => Some(*n as usize),
                _ => None,
            };
        }
    }

    // Fallback: try architecture.attention.head_count
    let arch = gf.metadata.get("general.architecture")
        .and_then(|v| match v {
            crate::gguf::MetaValue::String(s) => Some(s.as_str()),
            _ => None,
        })?;
    let key = format!("{}.attention.head_count", arch);
    match gf.metadata.get(&key) {
        Some(crate::gguf::MetaValue::Uint32(n)) => Some(*n as usize),
        Some(crate::gguf::MetaValue::Int32(n)) => Some(*n as usize),
        Some(crate::gguf::MetaValue::Uint64(n)) => Some(*n as usize),
        _ => None,
    }
}

pub fn run_intmap(
    path: &Path,
    eps: f64,
    min_pct: f64,
    html: bool,
    filter: Option<&str>,
    deep: usize,
) -> Result<()> {
    let gf = GgufFile::open(path)?;

    let model_name = gf.metadata.get("general.name")
        .map(|v| format!("{}", v))
        .unwrap_or_else(|| path.file_stem().unwrap().to_string_lossy().to_string());

    eprintln!("Scanning {} tensors in {} (eps={})", gf.tensors.len(), model_name, eps);

    let mut results = Vec::new();
    let mut skipped = 0usize;

    for info in &gf.tensors {
        // Apply filter
        if let Some(f) = filter {
            if !info.name.contains(f) {
                continue;
            }
        }

        // Try to extract
        match gf.extract_f32(&info.name) {
            Ok(data) => {
                let mut tm = analyze_tensor(
                    &info.name,
                    &data,
                    &info.shape_str(),
                    info.dtype.name(),
                    eps,
                    &info.dims,
                );

                // Block entropy for Q4_0 tensors
                if info.dtype.name() == "Q4_0" {
                    if let Ok(blocks) = gf.extract_q4_0_nibbles(&info.name) {
                        tm.block_entropy = Some(compute_block_entropy(&blocks));
                    }
                }

                results.push(tm);
            }
            Err(_) => {
                skipped += 1;
            }
        }
    }

    // Sort by block entropy (lower = more structured), with non-Q4_0 tensors last
    results.sort_by(|a, b| {
        let a_ent = a.block_entropy.as_ref().map(|e| e.mean_entropy).unwrap_or(99.0);
        let b_ent = b.block_entropy.as_ref().map(|e| e.mean_entropy).unwrap_or(99.0);
        a_ent.partial_cmp(&b_ent).unwrap()
    });

    let report = IntMapReport {
        model_name: model_name.replace('"', ""),
        n_tensors: gf.tensors.len(),
        n_scanned: gf.tensors.len() - skipped,
        n_skipped: skipped,
        tensors: results,
        eps,
    };

    if html {
        let content = render_html(&report);
        let path = std::env::temp_dir().join("nd-intmap.html");
        std::fs::write(&path, &content)?;
        eprintln!("Wrote: {}", path.display());
        std::process::Command::new("open").arg(&path).spawn()?;
    } else {
        print_report(&report);
    }

    // Deep analysis of top tensors
    if deep > 0 {
        let n_heads = infer_n_heads(&gf);
        if let Some(nh) = n_heads {
            eprintln!("Detected {} attention heads", nh);
        }

        let top: Vec<_> = report.tensors.iter().take(deep).collect();
        println!("\n{}", "=".repeat(80));
        println!("DEEP ANALYSIS — top {} tensors", deep);
        println!("{}", "=".repeat(80));

        for tm in &top {
            println!("\n── {} ──", tm.name);
            println!("  Shape: {}  Type: {}  Elements: {}", tm.shape, tm.dtype, tm.n_elements);
            println!("  Near-int: {:.1}%  Ternary: {:.1}%  Zero: {:.1}%  Max|w|: {:.4}",
                     tm.pct_near_int, tm.pct_ternary, tm.pct_zero, tm.max_abs);
            if let Some(ref be) = tm.block_entropy {
                println!("  Block entropy: {:.3} bits ({:.1} effective levels), {}/{} low-entropy blocks",
                         be.mean_entropy, be.effective_levels, be.n_low_entropy_blocks, be.n_total_blocks);
                println!("  Min block entropy: {:.3} bits", be.min_entropy);
                // Level histogram
                let total: usize = be.level_histogram.iter().sum();
                if total > 0 {
                    println!("  Q4_0 level usage (signed: -8..+7):");
                    for (i, &count) in be.level_histogram.iter().enumerate() {
                        let signed = i as i32 - 8;
                        let pct = count as f64 / total as f64 * 100.0;
                        let bar_len = (pct * 1.5).min(60.0) as usize;
                        if pct > 0.5 {
                            println!("    {:>3}: {:>5.1}% {}", signed, pct, "█".repeat(bar_len));
                        }
                    }
                }
            }
            println!("  Mean dist to int: {:.6}", tm.mean_dist_to_int);

            if !tm.int_histogram.is_empty() {
                println!("  Integer value distribution:");
                for (val, count) in &tm.int_histogram {
                    let pct = *count as f64 / tm.n_elements as f64 * 100.0;
                    let bar_len = (pct * 2.0).min(40.0) as usize;
                    println!("    {:>4}: {:>8} ({:>5.1}%) {}", val, count, pct, "█".repeat(bar_len));
                }
            }

            if tm.n_total_rows > 0 {
                println!("  Integer rows: {}/{} ({:.1}%)",
                         tm.n_int_rows, tm.n_total_rows,
                         tm.n_int_rows as f64 / tm.n_total_rows as f64 * 100.0);
                if !tm.row_hotspots.is_empty() {
                    println!("  Hotspot rows (>90% integer):");
                    for (row, pct) in &tm.row_hotspots {
                        println!("    row {:>5}: {:.1}%", row, pct);
                    }
                }
            }

            // Per-head entropy for attention projections
            if let Some(nh) = n_heads {
                let is_attn = tm.name.contains("attn_q") || tm.name.contains("attn_k") || tm.name.contains("attn_v");
                if is_attn {
                    if let Some(heads) = per_head_entropy(&gf, &tm.name, nh) {
                        println!("  Per-head entropy ({} heads, head_dim={}):",
                                 nh, tm.n_elements / nh / heads.first().map(|h| h.n_blocks * 32 / (tm.n_elements / nh)).unwrap_or(1));

                        // Sort by entropy to find the most structured heads
                        let mut sorted_heads = heads.clone();
                        sorted_heads.sort_by(|a, b| a.mean_entropy.partial_cmp(&b.mean_entropy).unwrap());

                        // Show all heads as a compact table
                        println!("    {:>4} {:>8} {:>8} {:>7} {:>8}",
                                 "HEAD", "ENTROPY", "EFF_LVL", "MAX|W|", "LOW_E%");
                        for h in &sorted_heads {
                            let pct_low = if h.n_blocks > 0 {
                                h.n_low_entropy_blocks as f64 / h.n_blocks as f64 * 100.0
                            } else { 0.0 };
                            let marker = if h.mean_entropy < sorted_heads[0].mean_entropy + 0.05 { " ◀" } else { "" };
                            println!("    {:>4} {:>8.3} {:>8.1} {:>7.3} {:>7.1}%{}",
                                     h.head_idx, h.mean_entropy, h.effective_levels, h.max_abs, pct_low, marker);
                        }

                        // Highlight the outlier
                        let mean_ent: f64 = heads.iter().map(|h| h.mean_entropy).sum::<f64>() / heads.len() as f64;
                        let std_ent: f64 = (heads.iter().map(|h| (h.mean_entropy - mean_ent).powi(2)).sum::<f64>() / heads.len() as f64).sqrt();
                        let min_head = &sorted_heads[0];
                        let max_head = sorted_heads.last().unwrap();
                        let z_score = (mean_ent - min_head.mean_entropy) / std_ent.max(1e-10);

                        println!("    ── Summary: mean={:.3}, std={:.4}, range={:.3}..{:.3}",
                                 mean_ent, std_ent, min_head.mean_entropy, max_head.mean_entropy);
                        if z_score > 2.0 {
                            println!("    ── HEAD {} IS A {:.1}σ OUTLIER (entropy {:.3} vs {:.3} mean) — potentially decompilable!",
                                     min_head.head_idx, z_score, min_head.mean_entropy, mean_ent);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn print_report(report: &IntMapReport) {
    println!("Integer Structure Map — {}", report.model_name);
    println!("Scanned {}/{} tensors (eps={}, {} unsupported types skipped)",
             report.n_scanned, report.n_tensors, report.eps, report.n_skipped);
    println!("Q4_0 max entropy = 4.0 bits (16 uniform levels). Lower = more structured.\n");

    if report.tensors.is_empty() {
        println!("No tensors found.");
        return;
    }

    // Primary table: block entropy
    println!("{:<45} {:>6} {:>8} {:>7} {:>7} {:>7} {:>6} {:>8}",
             "TENSOR", "TYPE", "ELEMS", "ENTROPY", "EFF_LV", "%LOW_E", "MAX|W|", "%ZERO");
    println!("{}", "-".repeat(108));

    for tm in &report.tensors {
        if let Some(ref be) = tm.block_entropy {
            let pct_low = be.n_low_entropy_blocks as f64 / be.n_total_blocks as f64 * 100.0;
            println!("{:<45} {:>6} {:>8} {:>7.3} {:>7.1} {:>6.1}% {:>6.3} {:>7.1}%",
                     truncate_name(&tm.name, 45),
                     tm.dtype,
                     format_count(tm.n_elements),
                     be.mean_entropy,
                     be.effective_levels,
                     pct_low,
                     tm.max_abs,
                     tm.pct_zero);
        } else {
            // Non-Q4_0 (F32 etc)
            println!("{:<45} {:>6} {:>8} {:>7} {:>7} {:>7} {:>6.3} {:>7.1}%",
                     truncate_name(&tm.name, 45),
                     tm.dtype,
                     format_count(tm.n_elements),
                     "F32",
                     "—",
                     "—",
                     tm.max_abs,
                     tm.pct_zero);
        }
    }

    // Summary
    println!("\n{}", "═".repeat(108));

    // Most structured tensors (lowest entropy)
    let structured: Vec<_> = report.tensors.iter()
        .filter(|t| t.block_entropy.as_ref().map(|e| e.mean_entropy < 3.5).unwrap_or(false))
        .collect();

    if !structured.is_empty() {
        println!("\nMOST STRUCTURED (entropy < 3.5 bits, using <~11 of 16 levels):");
        for t in &structured {
            let be = t.block_entropy.as_ref().unwrap();
            let pct_low = be.n_low_entropy_blocks as f64 / be.n_total_blocks as f64 * 100.0;
            println!("  {} — entropy {:.3} ({:.1} effective levels), {:.1}% low-entropy blocks",
                     t.name, be.mean_entropy, be.effective_levels, pct_low);

            // Show which Q4 levels dominate
            let total: usize = be.level_histogram.iter().sum();
            if total > 0 {
                let mut sorted: Vec<(usize, usize)> = be.level_histogram.iter()
                    .enumerate()
                    .map(|(i, &c)| (i, c))
                    .collect();
                sorted.sort_by(|a, b| b.1.cmp(&a.1));
                let top5: Vec<String> = sorted.iter().take(5)
                    .map(|(lvl, cnt)| {
                        let signed = *lvl as i32 - 8;
                        format!("{}:{:.1}%", signed, *cnt as f64 / total as f64 * 100.0)
                    })
                    .collect();
                println!("    └─ top levels: {}", top5.join(", "));
            }
        }
    }

    // Really low entropy = potentially decompilable
    let decompilable: Vec<_> = report.tensors.iter()
        .filter(|t| t.block_entropy.as_ref().map(|e| e.mean_entropy < 2.5).unwrap_or(false))
        .collect();

    if !decompilable.is_empty() {
        println!("\nPOTENTIALLY DECOMPILABLE (entropy < 2.5 bits, using <~6 levels):");
        for t in &decompilable {
            let be = t.block_entropy.as_ref().unwrap();
            println!("  {} — {:.3} bits ({:.1} levels), {} elements",
                     t.name, be.mean_entropy, be.effective_levels, format_count(t.n_elements));
        }
    }

    // F32 tensors with real integer weights
    let real_int: Vec<_> = report.tensors.iter()
        .filter(|t| t.block_entropy.is_none() && t.max_abs > 1.0 && t.pct_near_int > 20.0)
        .collect();
    if !real_int.is_empty() {
        println!("\nF32 TENSORS WITH REAL INTEGER WEIGHTS (max|w| > 1):");
        for t in &real_int {
            println!("  {} — {:.1}% near-int, max|w|={:.3}",
                     t.name, t.pct_near_int, t.max_abs);
        }
    }
}

fn render_html(report: &IntMapReport) -> String {
    let mut rows = String::new();
    for tm in &report.tensors {
        let color = if tm.pct_near_int > 50.0 {
            "#2ecc71"
        } else if tm.pct_near_int > 20.0 {
            "#f39c12"
        } else {
            "#e74c3c"
        };

        let bar_width = tm.pct_near_int.min(100.0);
        rows.push_str(&format!(
            r#"<tr>
                <td class="name">{}</td>
                <td>{}</td>
                <td class="num">{}</td>
                <td>
                    <div class="bar-bg">
                        <div class="bar" style="width:{}%;background:{}"></div>
                    </div>
                    <span class="pct">{:.1}%</span>
                </td>
                <td class="num">{:.1}%</td>
                <td class="num">{:.1}%</td>
                <td class="num">{:.6}</td>
            </tr>"#,
            tm.name, tm.dtype, format_count(tm.n_elements),
            bar_width, color, tm.pct_near_int,
            tm.pct_ternary, tm.pct_zero, tm.mean_dist_to_int
        ));
    }

    // Build layer heatmap data for the visualization
    let mut heatmap_data = String::new();
    for tm in &report.tensors {
        if !heatmap_data.is_empty() {
            heatmap_data.push(',');
        }
        heatmap_data.push_str(&format!(
            r#"{{"name":"{}","pct":{:.2},"ternary":{:.2},"zero":{:.2},"elements":{}}}"#,
            tm.name, tm.pct_near_int, tm.pct_ternary, tm.pct_zero, tm.n_elements
        ));
    }

    format!(r#"<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>nd intmap — {model}</title>
<style>
body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
h1 {{ color: #00ff88; margin-bottom: 5px; }}
.meta {{ color: #888; margin-bottom: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #16213e; color: #00ff88; padding: 8px 12px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #2a2a4a; }}
tr:hover {{ background: #16213e; }}
.name {{ font-weight: bold; color: #fff; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bar-bg {{ display: inline-block; width: 100px; height: 14px; background: #2a2a4a; border-radius: 3px; vertical-align: middle; }}
.bar {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
.pct {{ margin-left: 8px; font-size: 0.9em; }}
.standout {{ background: #1a3a2a !important; }}
#heatmap {{ margin: 20px 0; padding: 20px; background: #16213e; border-radius: 8px; }}
#heatmap h2 {{ color: #00ff88; margin-top: 0; }}
.hm-row {{ display: flex; align-items: center; margin: 2px 0; }}
.hm-label {{ width: 300px; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.hm-cell {{ height: 16px; flex: 1; border-radius: 2px; margin-left: 4px; cursor: pointer; }}
.hm-cell:hover {{ outline: 2px solid #fff; }}
</style>
</head>
<body>
<h1>nd intmap</h1>
<p class="meta">{model} — {scanned}/{total} tensors scanned (eps={eps}) — {skipped} unsupported skipped</p>

<div id="heatmap">
<h2>Integer Structure Heatmap</h2>
<div id="hm-container"></div>
</div>

<table>
<thead>
<tr><th>Tensor</th><th>Type</th><th>Elements</th><th>% Near-Integer</th><th>% Ternary</th><th>% Zero</th><th>Mean Dist</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<script>
const data = [{heatmap_data}];
const container = document.getElementById('hm-container');
data.forEach(d => {{
    const row = document.createElement('div');
    row.className = 'hm-row';
    const pct = d.pct;
    const r = pct > 50 ? 46 : pct > 20 ? 243 : 231;
    const g = pct > 50 ? 204 : pct > 20 ? 156 : 76;
    const b = pct > 50 ? 113 : pct > 20 ? 18 : 60;
    row.innerHTML = `<span class="hm-label">${{d.name}}</span><div class="hm-cell" style="background:rgba(${{r}},${{g}},${{b}},${{Math.max(0.2, pct/100)}})" title="${{d.name}}: ${{pct.toFixed(1)}}% int, ${{d.ternary.toFixed(1)}}% ternary"></div>`;
    container.appendChild(row);
}});
</script>
</body>
</html>"#,
        model = report.model_name,
        scanned = report.n_scanned,
        total = report.n_tensors,
        eps = report.eps,
        skipped = report.n_skipped,
        rows = rows,
        heatmap_data = heatmap_data,
    )
}

fn truncate_name(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("…{}", &s[s.len() - max + 1..])
    }
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
