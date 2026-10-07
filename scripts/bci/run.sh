#!/usr/bin/env bash
# End-to-end BCI demo: DANDI download -> train sparse RNN -> nd decompile/verify/stats/slice/xray -> interpretation.
# Usage: scripts/bci/run.sh [extra args for train.py, e.g. --repeats 8]
# Needs: cargo, uv, internet (first run only; ~29 MB NWB file is cached in examples/bci/data/).
set -euo pipefail
cd "$(dirname "$0")/../.."
ND=target/release/nd
B=examples/bci
mkdir -p results "$B/decompiled"

cargo build --release

echo "== 1/4 download + bin spikes (DANDI:000140 MC_Maze_Small)"
uv run --quiet --python 3.12 --with numpy==2.5.3 --with h5py==3.16.0 python scripts/bci/prepare.py

echo "== 2/4 train sparse ReLU RNN (train trials only; held-out trials touched at the end)"
uv run --quiet --python 3.12 --with numpy==2.5.3 --with torch==2.14.1 python scripts/bci/train.py --workers 2 "$@"

echo "== 3/4 nd decompile / verify / stats / slice / xray"
W=$B/reach_decoder.json
T=$B/reach_decoder_tests.json
$ND decompile "$W" --format python -o $B/decompiled/reach_decoder.py
$ND decompile "$W" --format rust   -o $B/decompiled/reach_decoder.rs
$ND decompile "$W" --format table  -o $B/decompiled/reach_decoder.table.txt
$ND stats  "$W"
$ND verify "$W" "$T"
echo "-- eps sweep (nd verify on the 25 held-out trials)"
for e in 0.01 0.05 0.10 0.15 0.25 0.35 0.50; do
  report="$($ND verify "$W" "$T" --eps "$e")"
  printf 'eps=%s  %s\n' "$e" "${report%%$'\n'*}"
done
$ND slice "$W" "$T" -o $B/decompiled/reach_decoder_sliced.json --emit python > $B/decompiled/reach_decoder_sliced.py
ND_NO_BROWSER=1 $ND xray "$W" "$T" --html
cp /tmp/nd-xray-reach_decoder.html results/bci_xray.html
$ND xray "$W" "$T" > $B/decompiled/reach_decoder.xray.txt

echo "== 4/4 interpretation + float-vs-decompiled agreement"
uv run --quiet --python 3.12 --with numpy==2.5.3 python scripts/bci/interpret.py | tee $B/interpretation.txt
