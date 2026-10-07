ND := target/release/nd

.PHONY: all build test bench train demo proofs bci breakthrough automata-test recovery recovery-test

all: build test bench demo

build:
	cargo build --release

test:
	cargo test

# Rebuilds nd, runs it on every example pair, rewrites results/benchmark.md
bench:
	scripts/bench.sh

# Retrain the RNN benchmark tasks (parity5, evens_detector, max4, max5,
# bitwise_xor, mod5_add) and overwrite examples/<task>{,_tests}.json.
train:
	uv run --with torch --with numpy scripts/train_rnn_tasks.py

# Decompile + verify contains_11, then write an xray HTML report for divisible_by_3.
demo: build
	@mkdir -p results
	$(ND) decompile examples/regex/contains_11.json --format python --output results/demo_contains_11.py
	@cat results/demo_contains_11.py
	$(ND) verify examples/regex/contains_11.json examples/regex/contains_11_tests.json
	ND_NO_BROWSER=1 $(ND) xray examples/regex/divisible_by_3.json examples/regex/divisible_by_3_tests.json --html
	cp /tmp/nd-xray-divisible_by_3.html results/xray_divisible_by_3.html
	@echo "Wrote results/demo_contains_11.py and results/xray_divisible_by_3.html"

# Kani proofs (requires `cargo install --locked kani-verifier && cargo kani setup`)
proofs: build
	kani-proofs/check_circuits.sh
	cd kani-proofs && cargo kani -j 2 --output-format terse

# BCI pipeline (scripts/bci/run.sh)
bci:
	scripts/bci/run.sh

# Automatic behavioral-quotient extraction + certified single-edit repair search
# (scripts/discover_automata.py; counterexample-driven, not a manual proof).
# Not part of `all`: pulls in z3-solver via uv on first run.
breakthrough:
	uv run scripts/discover_automata.py examples/regex/divisible_by_3.json --spec divisible_by_3 \
		--certificate results/automata/divisible_by_3.json --emit results/automata/divisible_by_3.py
	uv run scripts/discover_automata.py examples/regex/contains_11.json --spec contains_11 \
		--certificate results/automata/contains_11.json --emit results/automata/contains_11.py
	uv run scripts/discover_automata.py examples/regex/no_consecutive_1.json --spec no_consecutive_1 \
		--certificate results/automata/no_consecutive_1.json --emit results/automata/no_consecutive_1.py
	uv run scripts/discover_automata.py examples/regex/divisible_by_3.json --spec divisible_by_3 \
		--repair-one results/automata/mod3_repaired_weights.json \
		--certificate results/automata/mod3_repaired.json --emit results/automata/mod3_repaired.py
	uv run scripts/discover_automata.py examples/regex/divisible_by_3.json --spec divisible_by_3 \
		--check results/automata/divisible_by_3.json
	uv run scripts/discover_automata.py examples/regex/contains_11.json --spec contains_11 \
		--check results/automata/contains_11.json
	uv run scripts/discover_automata.py examples/regex/no_consecutive_1.json --spec no_consecutive_1 \
		--check results/automata/no_consecutive_1.json
	uv run scripts/discover_automata.py results/automata/mod3_repaired_weights.json --spec divisible_by_3 \
		--check results/automata/mod3_repaired.json --original examples/regex/divisible_by_3.json

# Behavioral and certification-boundary tests for automatic extraction.
automata-test:
	uv run --with z3-solver==5.1.0.0 python -m unittest discover -s tests -p 'test_automata.py'

# Offline human FALCON H1: frozen selection, matched baselines, Rust replay parity.
recovery:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --python 3.12 --with numpy==2.5.3 --with h5py==3.16.0 scripts/recovery/experiment.py
	uv run --python 3.12 scripts/recovery/render.py results/recovery.json results/recovery.html

recovery-test:
	uv run --python 3.12 --with numpy==2.5.3 python -m unittest discover -s tests -p 'test_recovery.py'
