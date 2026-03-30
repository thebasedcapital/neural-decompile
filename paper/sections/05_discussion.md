# 4. Discussion

## 4.1 Summary of Findings

We introduced neural decompilation — extracting executable sparse circuits from trained weights — and demonstrated it at two scales: formal verification on 13 RNN tasks, and functional verification on production LLM attention heads. The core discovery is that layer-0 K projections in multilingual transformers contain discrete, low-rank circuits that classify tokens by script type. These circuits are sparse (2.7% of weights), causally necessary (ablation Fisher = 0.00), faithful (interchange KL = $4.5 \times 10^{-4}$), and cross-architectural (Fisher > 4.0 in both Llama and Qwen families).

## 4.2 Relation to Prior Work

**Comparison with MIPS.** Michaud & Tegmark (2024) introduced the MIPS normalizer chain for extracting interpretable programs from RNN weights. Our L1 + direct quantization method is simpler (1 step vs. 5) and produces higher-quality results: the normalizer chain's whitening step actively destroys integer alignment in weights that are already near-integer after L1 training. We extend beyond MIPS in two directions: formal verification via Kani (which MIPS does not attempt), and application to transformer attention heads (which MIPS does not address).

**Comparison with activation-based interpretability.** Sparse autoencoders (Cunningham et al., 2023; Bricken et al., 2023) decompose activation vectors into interpretable features. Our approach is complementary: SAEs identify features from activations at runtime, while we identify circuits from weights at rest. The distinction matters because a weight-level circuit is input-independent — it describes what the head can compute, not what it happens to compute on a particular input. Our interchange intervention (Section 3.7) bridges the two: we verify that the weight-level circuit produces the same activations as the dense head during inference.

**Comparison with causal abstraction.** Geiger et al. (2023) formalize interchange intervention as a test of whether a hypothesized mechanism causally governs a model's behavior. Our interchange intervention (Section 3.7) follows this framework: we substitute the dense head with the decompiled sparse circuit and measure output divergence. The distinction is that our hypothesis comes from static weight analysis, not from a researcher's prior belief about model behavior.

**Comparison with circuit discovery.** ACDC (Conmy et al., 2023) and related methods discover circuits by measuring which edges in the computational graph are necessary for a task. These approaches require a defined task and a dataset. Our entropy-guided discovery requires neither — it identifies structured circuits from weight statistics alone, before any inference. The circuits we find may not correspond to researcher-defined tasks; they reflect whatever structure the model learned during training.

**Layer-0 attention routing.** The observation that layer-0 attention heads implement coarse routing has partial support in prior work. Olsson et al. (2022) identified "previous token heads" and "induction heads" in early layers but characterized them by attention pattern, not weight structure. Our contribution is showing that this routing function is encoded as discrete, decompilable circuits in the K projection weights, with a specific taxonomy (gates, classifiers, complex heads) and a universal structure across architectures.

## 4.3 Why Layer 0?

The layer-0 K entropy outlier (Section 3.2) admits a natural explanation. Layer 0 operates directly on token embeddings, which are fixed vectors assigned during training. The K projection at layer 0 must route attention using only these static features — it cannot rely on contextual representations built by earlier layers (there are none). This constraint favors discrete routing: rather than computing a nuanced, context-dependent key, the layer-0 K head reads a few embedding dimensions that encode token-level properties (script type, syntactic category, whitespace structure) and produces a fixed routing signal.

By layer 2, the residual stream contains information from layer 0 and layer 1's attention and FFN computations. The K projection can now compute context-dependent keys, which require the full quantization palette (all 16 Q4\_0 levels), explaining the entropy convergence.

The rank distribution supports this: 34% of layer-0 heads are rank 1-2 (binary decisions on 1-2 features), while no layer-2 head has comparably low entropy. The model transitions from fixed routing to flexible computation within the first two layers.

## 4.4 Limitations

**Model scale.** All verified models are small ($\leq$ 7B parameters). Whether the same discrete circuits exist in 70B+ models is untested. Larger models have more heads and more layers, which could allow finer-grained specialization or greater redundancy.

**Perplexity insensitivity.** Our GGUF patching experiment (Section 3.5, Table 6 in verification.md) showed that ablating Head 21 changes perplexity by only 0.24% — within error bars. Perplexity is a coarse metric: it averages over all tokens, diluting the signal from the $\sim$15% of tokens where script classification matters. The activation trace (Fisher = 5.94) and interchange intervention (KL = $4.5 \times 10^{-4}$) are more sensitive to the head's specific function.

**Two architecture families.** We replicated on Llama and Qwen. Testing on architecturally distinct models (Mamba, RWKV, or mixture-of-experts architectures) would strengthen the universality claim.

**Threshold sensitivity.** The sparse circuit depends on the weight threshold $\tau$. We used $|w| > 0.1$, which retains 2.7% of weights. A principled selection method (e.g., elbow detection on cumulative energy) would be more robust than a fixed threshold.

**No downstream task evaluation.** We verified the circuit's classification function through activation tracing and interchange intervention, but did not measure downstream task performance (e.g., multilingual NER, code-switching detection). The circuit may contribute to capabilities not captured by our probes.

## 4.5 Implications

Neural decompilation offers a path toward auditable AI. If a model's attention routing can be read as a formula — "this head classifies tokens by script type using 7 embedding dimensions and 23 weighted terms" — then that formula can be inspected for bias, tested for edge cases, and formally verified against a specification. The gap between "the model seems to do X" (activation-based) and "the model's weights implement X" (decompilation-based) is the gap between behavioral observation and structural understanding.

The practical implication is selective pruning with guarantees. If a sparse circuit with 2.7% of weights reproduces a head's function (interchange KL = $4.5 \times 10^{-4}$), the remaining 97.3% can be pruned without functional loss — not as an approximation, but as a verified simplification. Scaling this to all heads and layers could yield principled compression ratios grounded in circuit-level understanding rather than statistical heuristics.
