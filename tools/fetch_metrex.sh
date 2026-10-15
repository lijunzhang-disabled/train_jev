#!/usr/bin/env bash
# Download MetRex (scale-lab/MetRex, BSD-3-Clause) into data/raw/metrex/ (git-ignored): 25,868 Verilog
# designs, the clean pool for tools/build_mutants.py. Pinned to a fixed revision and checked by SHA-256.
#
# Usage: tools/fetch_metrex.sh        (safe to re-run; skips what is already there)
#        HF_ENDPOINT=https://hf-mirror.com tools/fetch_metrex.sh    (when the proxy blocks Hugging Face)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/data/raw/metrex"
REVISION="b9cb51395445da1586e5b0b3ba6f7c91851823c5"
SHA256=921c5de4882519854a70dcf31a384c213699752b7035fe8ed208c7a05f5e8c51   # metrex.json, 137 MB
HF_URL=${HF_ENDPOINT:-https://huggingface.co}   # honours a mirror, as tools/fetch_benchmarks.sh does

mkdir -p "$DIR"
f="$DIR/metrex.json"
if [ ! -s "$f" ]; then
    curl -fsSL --retry 3 "$HF_URL/datasets/scale-lab/MetRex/resolve/$REVISION/metrex.json" -o "$f.part"
    mv "$f.part" "$f"
fi
echo "$SHA256  $f" | sha256sum --check --quiet || { echo "FAILED: $f differs from the pinned version; delete it and rerun" >&2; exit 1; }
echo "MetRex: $(wc -l < "$f") designs in $f"
