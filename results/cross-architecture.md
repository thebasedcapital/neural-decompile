# Cross-Architecture Verification

**Date:** 2026-03-30
**Models:** TinyLlama 1.1B (Llama arch), Qwen2.5-0.5B (Qwen arch)

## Architecture Differences

| Feature | TinyLlama | Qwen2.5 |
|---------|-----------|---------|
| Family | Llama | Qwen |
| Params | 1.1B | 0.5B |
| d_model | 2048 | 896 |
| Q heads | 32 | 14 |
| KV heads | 4 | 2 |
| GQA ratio | 8:1 | 7:1 |
| RoPE variant | Standard | YaRN |
| QKV bias | No | Yes |
| Tokenizer | Llama SPM | Qwen BPE |

## Layer 0 K Entropy Outlier

| Model | Layer 0 std | Mean std (all layers) | σ from mean |
|-------|-------------|----------------------|-------------|
| TinyLlama | — | — | 7.3σ |
| Qwen2.5 | 0.0806 | 0.0276 | **4.3σ** |

## Activation Trace: Script Classification

### TinyLlama Layer 0 KV Head (Head 21 block)

Fisher = 5.94. PC1 axis:
- code_python: +0.92
- latex: +0.57
- english: -0.48
- russian: -1.10

### Qwen2.5 Layer 0 KV Head 0

Fisher = **5.44**. PC1 axis:
- code_python: +2.15
- latex: +1.54
- english: +0.57
- chinese: -0.33
- japanese: -0.83
- russian: -1.38
- arabic: -1.71

### Qwen2.5 Layer 0 KV Head 1

Fisher = **4.20**. PC1 axis:
- code_python: +3.83
- latex: +2.43
- english: +0.98
- arabic: -3.37

## Conclusion

The layer-0 K projection script classification pattern is **cross-architectural**:
1. Layer 0 is an entropy outlier in both Llama and Qwen families
2. KV head activations cluster by script type (Fisher > 4.0 in both)
3. The same axis: code/structured → natural language, with CJK intermediate
4. This emerges independently despite different tokenizers, training data, RoPE variants, and QKV architectures
