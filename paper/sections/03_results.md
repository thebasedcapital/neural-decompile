# 4. Results

## 4.1 RNN Decompilation and Formal Verification

We applied `nd` to 13 RNN circuits trained on binary sequence classification tasks (parity, modular arithmetic, regular language recognition). Table 1 summarizes the results.

**Table 1.** RNN decompilation results. All 13 tasks produced executable code with integer-valued weight matrices via L1 regularization + direct quantization ($\epsilon = 0.15$).

| Task | Hidden dim | % Integer weights | Verification | Notes |
|------|-----------|------------------|-------------|-------|
| parity3 | 3 | 100% | Kani proof (all 8 inputs) | XOR algorithm recovered |
| parity5 | 5 | 100% | Test cases (32/32) | |
| contains\_11 | 2 | 100% | Kani proof (all strings $\leq$5) | 2-state DFA |
| no\_consecutive\_1 | 2 | 100% | Kani proof (all strings $\leq$5) | Complement of contains\_11 |
| divisible\_by\_3 | 5 | 100% | Kani proof (all strings $\leq$7) | 3-state mod counter |
| mod3\_add | 3 | 100% | Kani proof (all 9 inputs) | Modular addition |
| complement | 2+2 | 100% | Kani proof (structural) | Proves contains\_11 $\equiv \neg$ no\_consecutive\_1 |
| mod5 | 5 | 95% | Test cases (1024/1024) | |
| mod7 | 7 | 92% | Test cases (2048/2048) | |
| binary\_add | 8 | 88% | Test cases (256/256) | |
| max4 | 4 | 100% | Test cases (16/16) | |
| max5 | 5 | 100% | Test cases (32/32) | |
| regex\_match | 4 | 97% | Test cases (128/128) | |

Six circuits received Kani formal proofs verifying correctness for all possible inputs within the bounded domain. The complement proof is a structural result: it demonstrates that two independently trained networks learned identical hidden dynamics (a 1-neuron DFA) with only output logits swapped, confirming that the decompiler recovers the same algorithm from equivalent circuits.

Our L1 + direct quantization method outperforms the MIPS 5-stage normalizer chain (Michaud & Tegmark, 2024) in both simplicity and result quality. The normalizer chain (whitening $\rightarrow$ Jordan form $\rightarrow$ Toeplitz $\rightarrow$ de-bias $\rightarrow$ round) actively destroys integer alignment in weights that are already near-integer, whereas direct quantization preserves it.

## 4.2 Transformer Entropy Scan

We scanned 201 tensors in TinyLlama 1.1B (Q4\_0) and 291 tensors in Llama-2 7B (Q4\_0) using `nd intmap`. In both models, layer 0 attention projections are entropy outliers.

**Table 2.** Layer 0 K projection entropy vs. cross-layer mean.

| Model | Layer 0 K entropy | Mean K entropy (all layers) | $\sigma$ from mean |
|-------|-------------------|---------------------------|-------------------|
| TinyLlama 1.1B | 2.508 bits | 3.317 bits | 7.3$\sigma$ |
| Llama-2 7B | 2.949 bits | 3.374 bits | 2.7$\sigma$ |
| Qwen2.5 0.5B | 0.0806 std | 0.0276 std | 4.3$\sigma$ |

Layer 0 Q and K projections are 15-25% below their cross-layer mean entropy. Layer 0 V projections are within 2% of their mean — the structure is in the routing (Q/K), not the content extraction (V). By layer 2, per-head entropy variance collapses: layer 0 heads span 1.824-2.924 bits ($\sigma = 0.305$), while layer 2 heads span 3.196-3.373 bits ($\sigma = 0.046$), a 6.6$\times$ reduction in spread.

## 4.3 Head Taxonomy

Per-head SVD decomposition of all 32 KV heads in TinyLlama layer 0 K reveals a structured distribution of circuit complexity.

**Table 3.** Layer 0 K head taxonomy by effective rank (90% energy threshold).

| Category | Count | Fraction | Characteristics |
|----------|-------|----------|----------------|
| Rank 1-2 (gates) | 11 | 34% | Binary routing decisions. Examples: newline detection (H6, H7), syntax vs. prose (H15), language family (H14, H26, H30, H31) |
| Rank 3-4 (classifiers) | 14 | 44% | Multi-class separation. Examples: special character type (H2, H3), punctuation (H8, H13), indentation depth (H12) |
| Rank 5+ (complex) | 7 | 22% | Rich representations. Examples: multi-script classifier (H21), domain detection (H16-H20, H22) |

Heads 6 and 7 are near-duplicate rank-1 gates (cosine similarity 0.992 between their principal directions) that both detect newlines vs. LaTeX/math tokens. Their output wiring differs — the model copies one feature detector into two heads to provide the Q projection with two independent handles on the same signal (multi-path routing).

Heads 2 and 3 are complementary: H2 detects EOS/BOM tokens, H3 detects CJK tokens, with opposite sign patterns on shared embedding features. This mirrors the complement structure discovered in RNN decompilation (Section 4.1).

## 4.4 Circuit Decompilation: Head 21 (Multi-Script Classifier)

TinyLlama layer 0 KV head 21's K projection (within the head-2 KV group, rows 168-175 of the 256$\times$2048 K weight matrix) has the model's largest weight magnitude ($|w| = 3.109$). SVD reveals effective rank 5 (90% energy in 5 singular values).

The circuit reads $\sim$7 embedding dimensions to classify tokens by script type:

**Table 4.** Dominant embedding features read by Head 21.

| Embedding dim | Positive end | Negative end |
|--------------|-------------|-------------|
| d411 | Unicode/CJK | Code/LaTeX |
| d802 | Unicode/Arabic | Markup |
| d1454 | CJK/katakana | LaTeX citations |
| d1447 | Code syntax | CJK punctuation |
| d95 | Proper names | Math terms |
| d687 | Java code | CJK |
| d297 | Code operators | German/Slavic |

The two dominant output dimensions are:

$$k_{46} = -3.109 \cdot d_{411} + 1.180 \cdot d_{1454} - 0.848 \cdot d_{95} + 0.750 \cdot d_{802} + \ldots \quad (23 \text{ terms}, 70\% \text{ energy})$$

$$k_{49} = -2.453 \cdot d_{411} + 0.617 \cdot d_{1454} + 0.617 \cdot d_{1447} + 0.543 \cdot d_{802} + \ldots \quad (12 \text{ terms}, 67\% \text{ energy})$$

The sparse circuit retains 882 of 16,384 weights (5.4\% at threshold $|w| > 0.1$, reduced to 441 weights / 2.7\% in the verification experiments).

## 4.5 Circuit Decompilation: Head 15 (Rank-2 Binary Gate)

Head 15 (rows 120-127) has the lowest per-head entropy (1.824 bits, 3.5 effective quantization levels). SVD reveals rank 2: singular values [3.71, 1.46, 0.26, 0.14, ...], with the rank-2 approximation capturing 90.7% of energy (9.3% residual).

The two principal directions encode:

**Feature 1 (72% variance):** Syntax/operators/CJK vs. natural language prose. Positive: `를`, `zu`, `:(`, `(+`, `、`. Negative: `which`, `latter`, `sudden`, `moreover`.

**Feature 2 (23% variance):** Morphologically complex (inflected/agglutinative) vs. analytic. Positive: `ність`, `љашње`, `като`. Negative: `\n`, `Josh`, `Miami`.

Head 15 and Head 21 are orthogonal (cosine similarity $\approx 0$) despite reading the same embedding features. Head 21 classifies tokens into multiple script categories; Head 15 compresses the same features into a 2D binary signal.

## 4.6 Functional Verification: Activation Tracing

We ran TinyLlama on 72 synthetic prompts across 9 content categories (8 per category: English, Chinese, Russian, Python, C++, LaTeX, Japanese, Korean, Arabic) and extracted Head 21's K projection activations via MLX native inference.

**Table 5.** Activation trace results under four experimental conditions.

| Condition | Fisher ratio | Interpretation |
|-----------|-------------|---------------|
| Full weights (baseline) | 5.94 | Head activations cluster by script type |
| Sparse circuit only (2.7%) | 5.66 | Sparse circuit alone carries the signal |
| Head ablated (zeros) | 0.00 | Removing head destroys classification |
| Random weights (matched $\sigma$) | 2.56 | Specific learned weights required |

The full-weights and sparse-only conditions produce indistinguishable clustering (Fisher 5.94 vs. 5.66), confirming the sparse circuit captures the head's classification function. Ablation reduces the Fisher ratio to zero — classification vanishes entirely, establishing causal necessity. Random weights produce moderate but weaker clustering (Fisher 2.56), demonstrating that the specific learned weights matter, not merely the architectural position.

PCA of the activation vectors shows PC1 (86.1% variance) separates code/LaTeX (positive) from natural language (negative), with Russian most negative and Python most positive.

## 4.7 Interchange Intervention

We replaced Head 21's dense K projection weight with the sparse circuit ($\mathbf{W}^{\text{sparse}}$) during a full forward pass through all 22 layers of TinyLlama, on 12 multilingual prompts including script-transition sentences (e.g., "The function def hello() prints こんにちは to the screen").

**Table 6.** Interchange intervention metrics.

| Metric | Value | Perfect |
|--------|-------|---------|
| KL divergence (mean) | 0.000447 | 0 |
| Top-1 agreement | 97.6% | 100% |
| Logit correlation | 0.9999 | 1.0 |

Substituting the dense head with the sparse circuit produces near-identical model output: the KL divergence between output distributions is $4.5 \times 10^{-4}$, the top predicted token matches 97.6% of positions, and the logit vectors correlate at $r = 0.9999$. The sparse circuit is a faithful causal replacement for the dense head.

## 4.8 Cross-Architecture Replication

We replicated the entropy scan and activation trace on Qwen2.5-0.5B-Instruct, a model from a different architectural family (Table 7).

**Table 7.** Architecture comparison.

| Feature | TinyLlama 1.1B | Qwen2.5 0.5B |
|---------|---------------|---------------|
| Family | Llama | Qwen |
| d\_model | 2048 | 896 |
| Q heads / KV heads | 32 / 4 | 14 / 2 |
| RoPE variant | Standard | YaRN |
| QKV bias | No | Yes |
| Tokenizer | Llama SPM (32K) | Qwen BPE (152K) |

Despite these architectural differences, the same pattern emerges:

**Table 8.** Cross-architecture activation trace.

| Model | Head | Fisher ratio | PC1 positive | PC1 negative |
|-------|------|-------------|-------------|-------------|
| TinyLlama | KV H2 (H21 block) | 5.94 | Python, LaTeX | Russian, English |
| Qwen2.5 | KV H0 | 5.44 | Python, LaTeX | Arabic, Russian |
| Qwen2.5 | KV H1 | 4.20 | Python, LaTeX | Arabic, Russian |

All three heads separate code/structured tokens from natural language tokens along the same axis, with Fisher ratios exceeding 4.0. The layer-0 K entropy outlier holds in both models (7.3$\sigma$ in TinyLlama, 4.3$\sigma$ in Qwen2.5). This pattern emerges from multilingual transformer training itself, independent of architecture, tokenizer, or training data.
