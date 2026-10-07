# Automatic algorithm extraction, falsification, and repair

## Observed result

```text
Original trained mod-3 RNN       One recurrent edge removed after quantization
9 reachable integer states  ->  5 reachable integer states       (-4)
7 minimal Moore states      ->  3 minimal Moore states           (-4)
wrong at 10100001            ->  equivalent to mod-3 specification

Unchanged: original model file, input alphabet, output labels, quantization epsilon.
Changed deliberately: the learned behavior outside the original fixture domain.
```

The original `examples/regex/divisible_by_3.json` passes its 254 fixture cases, covering every nonempty binary string of lengths 1–7. Its exact integer-state graph has nine reachable states and minimizes to seven behaviorally distinct states. Comparing that extracted graph with the textbook three-state remainder machine finds the shortest disagreement:

- Input: `10100001`, interpreted MSB-first as 161.
- Correct label: 0, because `161 % 3 == 2`.
- Original quantized network: 1, incorrectly accepting the string.

This is not a quantization-fidelity result about the original floating weights. It is an exact characterization of the model after epsilon-0.15 integer snapping.

## Automatic repair without training

The tool enumerates every single-coefficient ±1 edit to the 52 coefficients of the snapped network: **104 candidates**. An exact breadth-first search of each candidate's hidden-state/specification-state product rejects the candidate at its first disagreement, or establishes equivalence when the reachable product closes. Hitting the state limit is unresolved, not a pass.

Observed candidate outcomes:

| Outcome | Count |
|---|---:|
| Counterexample found | 97 |
| Exact equivalent product closure | 7 |
| State-limit unresolved | 0 |

Among certified candidates, the selection rule prefers smaller edits, then edge deletion over edge addition, then fewer reachable product states. It chooses:

```text
W_hh[3][0]: 1 -> 0
```

This deletes the contribution of hidden state 0 to the update of hidden state 3. The repaired network has five reachable integer states; minimization recovers three states. Its product with the true modulo-three automaton has three reachable pairs and no output disagreement.

**What follows:** the repaired mathematical-integer network and the specified modulo-three machine classify every finite one-hot binary sequence identically. The search used no gradients, no retraining, and no test-set optimization.

For this repaired model, the checker also establishes an arithmetic bridge: every coefficient, hidden value, multiplication, and partial sum on the closed reachable graph is bounded in absolute value by **24**, well below the exact-integer range of `f64` and `i64`. Thus the snapped recurrence can execute exactly in either type on the certified alphabet; this does not apply to the original unsnapped weights. The actual `nd` CLI still reports **254/254** on the original fixtures, and its trace on `10100001` now ends with logits `[16, -16]`, class 0, instead of the original `[-4, 4]`, class 1.

**What does not follow:** the original network learned modulo three, every neural network admits a small repair, or this procedure establishes the safety of a medical decoder. The specification is supplied. The repair deliberately changes the original network's behavior. It is one edit to the *quantized* weights, not one edit to the raw floating-point JSON.

## Extracting a finite algorithm from an unbounded recurrence

For `contains_11`, the active recurrence is:

```text
h' = max(0, 2h - x0 + 2x1 - 1)
```

The integer state can grow without bound after `11`. Exact raw-state enumeration therefore does not close. Rather than treating a truncated graph as a proof, the extractor tries a coordinate saturation map:

```text
alpha(h)_i = min(h_i, cap)
```

For each candidate cap it asks Z3 whether any nonnegative integer hidden state and either one-hot binary input violates either condition:

```text
output(h) = output(alpha(h))
alpha(step(h, bit)) = alpha(step(alpha(h), bit))
```

`unsat` means no such violation exists over that entire integer domain. Together with `alpha(0) = 0`, the two identities establish an output-preserving quotient by induction on sequence length. The abstract transition is `alpha(step(representative, bit))`; exact reachable-state enumeration then closes on the finite quotient.

Observed for both `contains_11` and `no_consecutive_1`: cap 2 passes, producing three reachable and three minimal states, each equivalent to its supplied language specification. The same generic code handles both models; it does not recognize task names during extraction.

The generated finite-state program keeps only a state index. A behavioral smoke scenario executes the extracted substring machine on 100,000 symbols; it does not accumulate the growing hidden integer. That execution is a smoke check, not the all-length argument—the homomorphism and product closure provide the latter.

## Reproduction

```bash
make breakthrough
make automata-test
```

Or run the key experiment directly:

```bash
uv run scripts/discover_automata.py examples/regex/divisible_by_3.json \
  --spec divisible_by_3 \
  --certificate results/automata/divisible_by_3.json \
  --emit results/automata/divisible_by_3.py

uv run scripts/discover_automata.py examples/regex/divisible_by_3.json \
  --spec divisible_by_3 \
  --repair-one results/automata/mod3_repaired_weights.json \
  --certificate results/automata/mod3_repaired.json \
  --emit results/automata/mod3_repaired.py

uv run scripts/discover_automata.py results/automata/mod3_repaired_weights.json \
  --check results/automata/mod3_repaired.json \
  --original examples/regex/divisible_by_3.json
```

Extraction requires two input channels encoding bit 0 as `[1,0]` and bit 1 as `[0,1]`. Padding is **not** in this certified alphabet. Empty sequences are included, with output evaluated at zero hidden state. Ties select the first class.

## Certificate and trust boundary

The JSON evidence binds to SHA-256 of the exact weight-file bytes, the snapping epsilon, reachable states and transitions, output labels, minimized partition, and specification comparison. Replay reconstructs the graph from the weights and repeats the saturation query when needed. A repair replay also requires the original weights and repeats the bounded edit search; changed repair metadata is rejected.

This is replayable evidence, not a digitally signed certificate or an independently checked Z3 proof term. The trusted implementation includes Python's integer arithmetic, the extractor/checker, the symbolic encoding, and Z3 for saturation. Reachable-state closure itself needs no SMT solver. The tests exercise tampered transitions, outputs, source digests, specification results, repair metadata, strict quantization boundaries, and refusal when state exploration or abstraction is insufficient.

The equivalence domain is the snapped **mathematical-integer** network. It does not cover rounding in the original float model, overflow of growing fixed-width hidden states, or execution of arbitrary emitted Rust on unbounded inputs. Hybrid noninteger models are rejected. State limits and solver unknown/timeouts are explicitly unresolved. These failures do not imply that no other finite abstraction exists.

For exact reachable-state closure, the certificate additionally bounds the sum of absolute terms in each update and output expression, independently of accumulation order. A bound at most `2**53` is sufficient for every integer operation to be exact in both `f64` and `i64`. Otherwise numerical equivalence is marked `not_established`. Saturation certificates do not transfer this bound to the unbounded raw recurrence; their emitted finite-state implementation avoids that recurrence instead.

## Prior art and contribution boundary

Automata extraction from recurrent networks is established work, including [Weiss, Goldberg & Yahav, ICML 2018](https://proceedings.mlr.press/v80/weiss18a.html). State minimization and product-machine equivalence are classical algorithms.

The project-level result is the executable combination: automatic integer-state/quotient extraction, a shortest counterexample exposing an existing perfect-fixture claim, and a replayably certified single-edge repair of that actual trained model. Novelty beyond that needs a literature review and results on more models; one toy repair is not sufficient for a general research claim.
