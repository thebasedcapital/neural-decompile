# 1. Introduction

Understanding what neural networks compute remains one of the central problems in machine learning. The dominant approach — mechanistic interpretability — analyzes network behavior by studying activation patterns during inference: identifying induction heads through attention visualization (Olsson et al., 2022), decomposing residual streams with sparse autoencoders (Cunningham et al., 2023), and tracing information flow through causal intervention (Conmy et al., 2023). These methods treat the network as a dynamic system observed at runtime.

We propose a complementary approach: static decompilation of network weights into executable code. Where activation-based methods ask "what does this network do on this input?", weight-based decompilation asks "what algorithm do these weights implement, regardless of input?" The key insight is that trained weights, particularly in attention projection matrices, often contain discrete structure — sparse, low-rank patterns that use far fewer quantization levels than expected. These patterns are interpretable: they map specific embedding features to specific head output dimensions through compact linear formulas that humans can read and machines can verify.

We introduce `nd` (neural decompiler), a Rust CLI tool that implements this approach through four stages: (1) entropy-guided discovery of structurally anomalous weight matrices, (2) SVD-based decomposition into interpretable features, (3) sparse circuit extraction via thresholding, and (4) functional verification through activation tracing and causal intervention. For small networks, we go further: the decompiled code is formally verified using Kani, a model checker for Rust, proving correctness for all possible inputs — not just a test sample.

We apply `nd` to both synthetic RNNs and production LLMs. On 13 RNN binary classification tasks, we achieve perfect decompilation with 6 formal proofs of structural equivalence. On TinyLlama 1.1B, we decompile 32 attention heads at layer 0, revealing a structured taxonomy: 34% are rank 1-2 binary gates, 44% are rank 3-4 classifiers, and 22% are rank 5+ complex representations. One head (Head 21) implements a multi-script content classifier that separates code, LaTeX, CJK, Cyrillic, and Arabic tokens along a single principal axis. The sparse circuit — just 2.7% of the head's weights — reproduces this classification with Fisher discriminant ratio 5.66 (vs. 5.94 for the full head), achieves 97.6% top-1 agreement when substituted during a full forward pass (KL divergence $4.5 \times 10^{-4}$), and is causally necessary (ablation reduces the Fisher ratio to zero). The same pattern replicates in Qwen2.5-0.5B, a model from a different architectural family (YaRN, QKV bias, different tokenizer), with Fisher ratio 5.44.

Our contributions are:

1. **Neural decompilation as a method.** A pipeline that extracts readable, executable sparse circuits from trained weights, with formal verification for RNNs and functional verification for transformers.

2. **Layer-0 K projection structure is universal.** Across three models from two architecture families, layer-0 K projections are entropy outliers (4.3-7.3$\sigma$ below cross-layer mean) containing specialized attention routing circuits. Deeper layers show no such specialization.

3. **First verified decompilation of an LLM attention head.** A sparse circuit (2.7% of weights) in TinyLlama Head 21 implements multi-script classification, verified causally (ablation Fisher = 0.00), functionally (interchange KL = $4.5 \times 10^{-4}$), and cross-architecturally (Qwen2.5 Fisher = 5.44).

Code, data, and Kani proofs are publicly available at [repository URL].
