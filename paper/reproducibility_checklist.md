# NeurIPS / ICML Reproducibility Checklist

## Code and Data

| Item | Answer | Details |
|------|--------|---------|
| Code available? | Yes | Rust CLI (`nd`), 8.8K LOC, MIT license |
| Data available? | Yes | Trained RNN weights (JSON), GGUF models (public HuggingFace) |
| Pre-trained models used? | Yes | TinyLlama-1.1B-Chat-v1.0 (Q4_0), Llama-2-7B (Q4_0), Qwen2.5-0.5B-Instruct |
| All models publicly available? | Yes | HuggingFace Hub, no gated access |
| Random seeds reported? | Yes | RNG seed 42 for random weight ablation |
| Hardware specified? | Yes | Apple M4, 16GB, MLX 0.31.1, Rust 2021 edition |

## Experiments

| Item | Answer | Details |
|------|--------|---------|
| Number of runs per experiment? | 1 | Deterministic (no stochastic components in entropy scan, SVD, or K projection) |
| Error bars / confidence intervals? | Yes | Perplexity ± std error. Fisher ratio is deterministic given fixed prompts. |
| Statistical tests? | Fisher discriminant ratio | Between-class / within-class variance ratio |
| Hyperparameters listed? | Yes | Entropy threshold (2σ), weight threshold (|w|>0.1), Q4_0 quantization |
| Compute budget? | <1 GPU-hour | All experiments run on Apple M4 CPU/GPU, no cloud compute |

## Claims

| Claim | Evidence Type | Strength |
|-------|-------------|----------|
| L1+quantize beats MIPS normalizers | 13/13 vs published results | Direct comparison |
| 6 Kani formal proofs | Model checking, all inputs verified | Mathematical proof |
| Head 21 is a script classifier | Activation trace Fisher=5.94 | Strong (Fisher>5) |
| Sparse circuit carries signal | Sparse Fisher=5.66 ≈ Full 5.94 | Strong |
| Causal necessity | Ablated Fisher=0.00 | Definitive |
| Cross-architecture | Qwen Fisher=5.44 | Strong (different family) |
| Interchange intervention | KL=0.0004, top-1=97.6% | Near-perfect |

## Limitations Acknowledged

- All models ≤7B parameters
- Only 2 architecture families (Llama, Qwen)
- No downstream task evaluation
- Weight threshold chosen by hand (|w|>0.1)
- Perplexity delta within error bars (addressed by activation trace instead)
