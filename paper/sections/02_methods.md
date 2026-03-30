# 3. Method

Neural decompilation operates in four stages: entropy-guided discovery, weight-level decomposition, sparse circuit extraction, and functional verification. We implement each stage in `nd`, a Rust CLI tool that operates directly on quantized GGUF model files without requiring a full inference stack.

## 3.1 Entropy-Guided Circuit Discovery

Given a GGUF model file containing $N$ tensors, we compute per-tensor block entropy to identify structurally anomalous weight matrices. For Q4\_0 quantized tensors, each block of 32 values uses one of 16 quantization levels. The Shannon entropy of the level distribution within a block measures how many levels the block actually uses:

$$H_b = -\sum_{i=0}^{15} p_i \log_2 p_i$$

where $p_i$ is the fraction of values in block $b$ at level $i$. We average $H_b$ across all blocks in a tensor to obtain the tensor-level entropy $\bar{H}$. A tensor with $\bar{H} \ll 4.0$ bits (the maximum for 16 uniform levels) uses fewer quantization levels than expected — it contains discrete, structured patterns rather than continuous distributions.

We flag tensors with $\bar{H}$ more than $2\sigma$ below the cross-layer mean as candidate circuits for decomposition. For attention projection weights (Q, K, V), we extend this analysis to the per-head level by partitioning the tensor's Q4\_0 blocks according to head boundaries and computing $\bar{H}$ per head. This identifies specific attention heads with unusually structured weight patterns, even when the tensor-level entropy is unremarkable.

## 3.2 Weight-Level Decomposition

For each candidate head, we extract the weight submatrix $\mathbf{W}_h \in \mathbb{R}^{d_{\text{head}} \times d_{\text{model}}}$ and perform truncated SVD:

$$\mathbf{W}_h = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

The effective rank $r$ (the smallest $k$ such that $\sum_{i=1}^{k} \sigma_i^2 \geq 0.9 \sum_j \sigma_j^2$) determines the circuit's complexity. A rank-1 head projects all tokens onto a single direction in embedding space (a binary gate). A rank-2 head computes two independent features. Higher-rank heads implement richer classification functions.

We identify the semantic meaning of each principal direction $\mathbf{v}_i$ by projecting the full token embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}$ onto $\mathbf{v}_i$ and examining which tokens produce the largest positive and negative projections. This maps each direction to an interpretable axis (e.g., "code tokens vs. natural language tokens").

## 3.3 Sparse Circuit Extraction

From the decomposed head, we extract a sparse circuit by thresholding: retain only weights with $|w_{ij}| > \tau$ for a threshold $\tau$ chosen to capture $\geq 60\%$ of the per-row energy. This produces a weight matrix $\mathbf{W}_h^{\text{sparse}}$ with typically 2-6\% nonzero entries. The sparse circuit is the decompiled output: a compact formula mapping embedding dimensions to head output dimensions, readable by a human and executable as code.

For RNN circuits, the decompiled code is emitted as Python or Rust source with integer-valued weight matrices and explicit state transition logic. For transformer attention heads, the output is a sparse linear map with labeled embedding features.

## 3.4 Formal Verification (RNNs)

For RNN circuits with integer-valued quantized weights, we emit Rust source code and verify structural equivalence using Kani, a model checker for Rust. Kani exhaustively checks all possible inputs up to a bounded length, proving that the decompiled circuit produces identical outputs to the original network for every input in the verified domain. This provides formal correctness guarantees stronger than test-case verification: a Kani proof covers the entire input space, not a sample.

## 3.5 Functional Verification (Transformers)

For transformer attention heads, formal verification is intractable due to floating-point arithmetic and model scale. We instead verify through three complementary experiments:

**Activation tracing.** We run the full model on synthetic prompts spanning $C$ content categories (e.g., English, Chinese, Russian, Python, C++, LaTeX, Japanese, Korean, Arabic). For each prompt, we extract the K projection output at the candidate head and compute the mean activation vector across sequence positions. We measure inter-category separation using the Fisher discriminant ratio:

$$F = \frac{\sigma^2_{\text{between}}}{\sigma^2_{\text{within}}}$$

where $\sigma^2_{\text{between}}$ is the weighted sum of squared distances from category centroids to the global mean, and $\sigma^2_{\text{within}}$ is the total within-category variance. $F > 1$ indicates the head's activations cluster by content category; $F > 5$ indicates strong separation.

**Causal ablation.** We run the activation trace under four conditions: (1) original weights, (2) sparse circuit only ($\mathbf{W}_h^{\text{sparse}}$, all other heads unchanged), (3) head ablated (all weights zeroed), and (4) random weights (Gaussian, matched variance). If the sparse circuit carries the signal (condition 2 $\approx$ condition 1), ablation destroys it (condition 3 $\ll$ condition 1), and random weights do not reproduce it (condition 4 $\ll$ condition 1), we conclude the sparse circuit is the causal mechanism.

**Interchange intervention.** We replace the head's dense K projection weight with $\mathbf{W}_h^{\text{sparse}}$ during a full forward pass and measure the output divergence. We report three metrics: KL divergence between output logit distributions, top-1 token prediction agreement, and Pearson correlation of logit vectors. Near-zero KL divergence and near-perfect agreement confirm that the sparse circuit faithfully replicates the dense head's contribution to the model's output.

## 3.6 Cross-Architecture Replication

To test whether discovered circuits are architecture-specific artifacts, we repeat the entropy scan and activation trace on models from different architectural families. We compare models that differ in tokenizer vocabulary, RoPE variant (standard vs. YaRN), QKV bias terms, GQA ratio, and training data composition. Convergent results across architectures indicate the circuit is a universal property of multilingual transformer training, not a design choice.

## 3.7 Implementation

`nd` is implemented in 8,800 lines of Rust with dependencies on `clap`, `ndarray`, `serde`, `anyhow`, and `memmap2`. It operates directly on memory-mapped GGUF files, supporting Q4\_0, Q8\_0, F16, BF16, and F32 tensor formats. The full toolchain comprises 15 commands including `intmap` (entropy scan), `decompile` (code emission), `verify` (test-case checking), `slice` (dead neuron removal), and `xray` (combined analysis with HTML output). Kani verification uses a separate Cargo workspace with bounded proof harnesses. Activation tracing uses MLX for native Apple Silicon inference. All source code, weight files, and experimental data are publicly available.
