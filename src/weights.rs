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

/// Detect model type from JSON structure
pub fn detect_model_type(path: &Path) -> Result<ModelType> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&data)
        .context("Failed to parse JSON")?;

    if value.get("token_emb").is_some() || value.get("layers").is_some() {
        Ok(ModelType::Transformer)
    } else if value.get("W_hh").is_some() || value.get("w_hh").is_some() {
        Ok(ModelType::Rnn)
    } else {
        anyhow::bail!("Unknown model type: cannot detect RNN or Transformer structure")
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Rnn,
    Transformer,
}

/// Unified neural program that can be either RNN or Transformer
#[derive(Debug, Clone)]
pub enum NeuralProgram {
    Rnn(RnnWeights),
    Transformer(super::transformer::Transformer),
}

impl NeuralProgram {
    pub fn input_dim(&self) -> usize {
        match self {
            NeuralProgram::Rnn(r) => r.input_dim,
            NeuralProgram::Transformer(t) => t.d_model, // Token embedding dimension
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            NeuralProgram::Rnn(r) => r.output_dim,
            NeuralProgram::Transformer(t) => t.vocab_size,
        }
    }

    pub fn model_type(&self) -> ModelType {
        match self {
            NeuralProgram::Rnn(_) => ModelType::Rnn,
            NeuralProgram::Transformer(_) => ModelType::Transformer,
        }
    }
}

/// Auto-detect and load either RNN or Transformer
pub fn load_neural_program(path: &Path) -> Result<NeuralProgram> {
    match detect_model_type(path)? {
        ModelType::Rnn => Ok(NeuralProgram::Rnn(load_rnn_weights(path)?)),
        ModelType::Transformer => Ok(NeuralProgram::Transformer(
            super::transformer::Transformer::from_json(path)?
        )),
    }
}
