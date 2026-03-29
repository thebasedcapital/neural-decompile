use crate::quantize::QuantizedRnn;
use crate::compare;
use crate::fsm;

/// Result of comparing two circuits
#[derive(Debug)]
pub struct PairResult {
    pub a: String,
    pub b: String,
    pub structural_sim: f64,   // avg cosine sim of transition function (0 if dim mismatch)
    pub output_sim: f64,       // cosine sim of output weights
    pub is_complement: bool,
    pub behavioral_agree: f64, // % agreement on shared input space
    pub relationship: String,
}

/// Generate all possible binary sequences up to given length, encoded as one-hot vectors
fn generate_binary_inputs(max_len: usize, input_dim: usize) -> Vec<Vec<Vec<f64>>> {
    let mut all = Vec::new();
    for len in 1..=max_len {
        for val in 0..(1u64 << len) {
            let seq: Vec<Vec<f64>> = (0..len).map(|i| {
                let bit = ((val >> (len - 1 - i)) & 1) as usize;
                if input_dim == 2 {
                    if bit == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }
                } else {
                    let mut v = vec![0.0; input_dim];
                    if bit < input_dim { v[bit] = 1.0; }
                    v
                }
            }).collect();
            all.push(seq);
        }
    }
    all
}

/// Compare behavior of two circuits on shared input space
fn behavioral_agreement(a: &QuantizedRnn, b: &QuantizedRnn) -> f64 {
    // Only compare if input dims match
    if a.input_dim != b.input_dim {
        return 0.0;
    }

    // Generate test inputs (binary sequences up to length 6)
    let inputs = generate_binary_inputs(6, a.input_dim);
    if inputs.is_empty() {
        return 0.0;
    }

    let mut agree = 0;
    let mut total = 0;
    for seq in &inputs {
        let out_a = fsm::run_fsm(a, seq);
        let out_b = fsm::run_fsm(b, seq);
        // Can only compare if both have same output range
        if a.output_dim == b.output_dim {
            if out_a == out_b {
                agree += 1;
            }
            total += 1;
        }
    }

    if total == 0 { 0.0 } else { agree as f64 / total as f64 }
}

/// Check if B's outputs are the complement of A's (for binary classifiers)
fn behavioral_complement(a: &QuantizedRnn, b: &QuantizedRnn) -> f64 {
    if a.input_dim != b.input_dim || a.output_dim != 2 || b.output_dim != 2 {
        return 0.0;
    }

    let inputs = generate_binary_inputs(6, a.input_dim);
    if inputs.is_empty() {
        return 0.0;
    }

    let mut complement = 0;
    let mut total = 0;
    for seq in &inputs {
        let out_a = fsm::run_fsm(a, seq);
        let out_b = fsm::run_fsm(b, seq);
        // Complement: when A says 0, B says 1, and vice versa
        if (out_a == 0 && out_b == 1) || (out_a == 1 && out_b == 0) {
            complement += 1;
        }
        total += 1;
    }

    if total == 0 { 0.0 } else { complement as f64 / total as f64 }
}

/// Run pairwise comparison across all circuits
pub fn build_taxonomy(circuits: &[(String, QuantizedRnn)]) -> Vec<PairResult> {
    let mut results = Vec::new();

    for i in 0..circuits.len() {
        for j in (i+1)..circuits.len() {
            let (name_a, qa) = &circuits[i];
            let (name_b, qb) = &circuits[j];

            let cmp = compare::compare(qa, qb);

            let structural_sim = if cmp.hidden_match {
                (cmp.w_hh_sim + cmp.w_hx_sim + cmp.b_h_sim) / 3.0
            } else {
                0.0
            };

            let behavioral = behavioral_agreement(qa, qb);
            let complement = behavioral_complement(qa, qb);

            let relationship = if structural_sim > 0.95 && cmp.output_negated {
                "COMPLEMENT".to_string()
            } else if structural_sim > 0.95 && cmp.w_y_sim > 0.95 {
                "IDENTICAL".to_string()
            } else if structural_sim > 0.8 {
                "SIMILAR_TRANSITION".to_string()
            } else if behavioral > 0.9 {
                "BEHAVIORAL_MATCH".to_string()
            } else if complement > 0.95 {
                "BEHAVIORAL_COMPLEMENT".to_string()
            } else if behavioral > 0.6 {
                "PARTIAL_OVERLAP".to_string()
            } else {
                "DISTINCT".to_string()
            };

            results.push(PairResult {
                a: name_a.clone(),
                b: name_b.clone(),
                structural_sim,
                output_sim: cmp.w_y_sim,
                is_complement: cmp.output_negated || complement > 0.95,
                behavioral_agree: behavioral,
                relationship,
            });
        }
    }

    results
}

/// Format taxonomy as a report
pub fn format_taxonomy(results: &[(String, QuantizedRnn)], pairs: &[PairResult]) -> String {
    let mut out = String::new();

    out.push_str("═══ CIRCUIT TAXONOMY ═══\n\n");

    // Summary table of circuits
    out.push_str(&format!("{:<20} {:>3} {:>3} {:>3}\n", "Task", "hd", "id", "od"));
    out.push_str(&format!("{}\n", "─".repeat(32)));
    for (name, q) in results {
        out.push_str(&format!("{:<20} {:>3} {:>3} {:>3}\n",
            name, q.hidden_dim, q.input_dim, q.output_dim));
    }
    out.push('\n');

    // Group by relationship
    let mut by_rel: std::collections::HashMap<String, Vec<&PairResult>> = std::collections::HashMap::new();
    for p in pairs {
        by_rel.entry(p.relationship.clone()).or_default().push(p);
    }

    // Interesting relationships first
    for rel in ["COMPLEMENT", "IDENTICAL", "SIMILAR_TRANSITION", "BEHAVIORAL_MATCH",
                "BEHAVIORAL_COMPLEMENT", "PARTIAL_OVERLAP", "DISTINCT"] {
        if let Some(group) = by_rel.get(rel) {
            if rel != "DISTINCT" || group.len() < 20 {
                out.push_str(&format!("── {} ({}) ──\n", rel, group.len()));
                for p in group {
                    out.push_str(&format!("  {} ↔ {}  struct={:.2} behav={:.2}{}",
                        p.a, p.b, p.structural_sim, p.behavioral_agree,
                        if p.is_complement { " [C̄]" } else { "" }));
                    out.push('\n');
                }
                out.push('\n');
            } else {
                out.push_str(&format!("── {} ({}) ── [elided]\n\n", rel, group.len()));
            }
        }
    }

    // Families: cluster by behavioral agreement > 0.5
    out.push_str("── FAMILIES ──\n");

    // Find connected components via behavioral agreement
    let task_names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
    let n = task_names.len();
    let mut adj = vec![vec![false; n]; n];

    for p in pairs {
        if p.behavioral_agree > 0.5 || p.is_complement {
            let ia = task_names.iter().position(|&t| t == p.a);
            let ib = task_names.iter().position(|&t| t == p.b);
            if let (Some(i), Some(j)) = (ia, ib) {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }

    // Simple DFS clustering
    let mut visited = vec![false; n];
    let mut family_id = 0;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut cluster = Vec::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            cluster.push(node);
            for next in 0..n {
                if adj[node][next] && !visited[next] {
                    stack.push(next);
                }
            }
        }
        family_id += 1;
        let members: Vec<&str> = cluster.iter().map(|&i| task_names[i]).collect();
        if members.len() > 1 {
            out.push_str(&format!("  Family {}: {}\n", family_id, members.join(", ")));
        } else {
            out.push_str(&format!("  Loner  {}: {}\n", family_id, members[0]));
        }
    }

    out
}
