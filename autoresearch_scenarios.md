# Autoresearch Test Corpus — Neural Decompiler

## Scenario 1: Happy Path (Known Algorithm)
**Task**: Parity of 5 bits
**Model**: 3-neuron RNN trained with L1 regularization
**Expected**: Decompiled code IS the XOR algorithm, 100% integer weights
**Why**: Baseline — tool should nail this perfectly

## Scenario 2: Missing Infrastructure (No L1 Training)
**Task**: Binary addition (x + y mod 4)
**Model**: 8-neuron RNN trained WITHOUT L1 regularization
**Expected**: Still decompiles to integer-coefficient code after quantization
**Why**: Real models aren't trained with L1 — does the tool work on standard training?

## Scenario 3: Scale Challenge (100+ Neurons)
**Task**: Decimal digit classification from 7-segment display patterns
**Model**: 50-neuron RNN, 2 layers
**Expected**: Extracts the decision rules for each digit, identifies dead neurons
**Why**: Toy examples are 2-5 neurons — does it scale to real sizes?

## Scenario 4: Transformer Discovery (Novel Algorithm)
**Task**: Induction head behavior (predict next token in repeated sequence)
**Model**: 2-layer, 4-head transformer trained on "ABAB..." sequences
**Expected**: Discovers the "copy from 2 positions ago" algorithm in attention patterns
**Why**: This is the actual mechanism in real LLMs — can we extract it?

## Scenario 5: Edge Case (Non-Integer Weights)
**Task**: Soft classification (probability output, not argmax)
**Model**: RNN with sigmoid output, weights NOT near integers
**Expected**: Either (a) decompiles to floating-point code with error bounds, or (b) clearly reports "not decompilable — weights not quantizable"
**Why**: Not everything decompiles — tool should gracefully fail, not silently produce wrong code

## Scenario 6: Wrong Domain (CNN-Transformer Hybrid)
**Task**: Syntactic structure recovery from obfuscated code
**Model**: CNN → Transformer hybrid (convolutional pre-filtering)
**Expected**: Separates CNN feature detection from transformer sequence reasoning
**Why**: Real models are hybrids — can the tool disentangle components?

---

## Eval Rubric (Per Scenario)

For each scenario, answer:
1. **Fidelity**: Does decompiled code match original on all test cases? (YES/NO)
2. **Interpretability**: Can you name the algorithm in one sentence? (YES/NO)
3. **Completeness**: Does output include all relevant info (dead neurons, integer %, circuit diagram)? (YES/NO)
4. **Novelty**: Is the extracted algorithm non-trivial? (YES/NO — only for scenarios 4, 6)

**Pass threshold**: 4/4 YES for scenarios 1-3, 5; 3/4 YES for scenarios 4, 6 (novelty is a bonus there)
