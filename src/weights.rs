use anyhow::{Context, Result};
use ndarray::Array2;
use serde::Deserialize;
use std::path::Path;

/// Raw RNN weight matrices as loaded from file
#[derive(Debug, Clone)]
pub struct RnnWeights {
    pub w_hh: Array2<f64>,  // (hidden_dim, hidden_dim)
    pub w_hx: Array2<f64>,  // (hidden_dim, input_dim)
    pub b_h: Vec<f64>,      // (hidden_dim,)
    pub w_y: Array2<f64>,   // (output_dim, hidden_dim)
    pub b_y: Vec<f64>,      // (output_dim,)
    pub hidden_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

#[derive(Deserialize)]
struct JsonWeights {
    #[serde(alias = "W_hh")]
    w_hh: Vec<Vec<f64>>,
    #[serde(alias = "W_hx")]
    w_hx: Vec<Vec<f64>>,
    #[serde(alias = "b_h")]
    b_h: Vec<f64>,
    #[serde(alias = "W_y")]
    w_y: Vec<Vec<f64>>,
    #[serde(alias = "b_y")]
    b_y: Vec<f64>,
}

fn vec2d_to_array2(v: &[Vec<f64>]) -> Result<Array2<f64>> {
    let rows = v.len();
    let cols = v.first().map_or(0, |r| r.len());
    let flat: Vec<f64> = v.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((rows, cols), flat)
        .context("Failed to reshape weight matrix")
}

pub fn load_rnn_weights(path: &Path) -> Result<RnnWeights> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let json: JsonWeights = serde_json::from_str(&data)
        .context("Failed to parse weight JSON")?;

    let w_hh = vec2d_to_array2(&json.w_hh)?;
    let w_hx = vec2d_to_array2(&json.w_hx)?;
    let w_y = vec2d_to_array2(&json.w_y)?;

    let hidden_dim = w_hh.nrows();
    let input_dim = w_hx.ncols();
    let output_dim = w_y.nrows();

    Ok(RnnWeights {
        w_hh, w_hx, b_h: json.b_h, w_y, b_y: json.b_y,
        hidden_dim, input_dim, output_dim,
    })
}
