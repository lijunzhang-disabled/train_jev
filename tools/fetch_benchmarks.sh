#!/usr/bin/env bash
# Download the two real-bug benchmarks into data/raw/ (git-ignored), pinned to fixed revisions.
#
#   verification-benchmarks  SVA security properties on buggy Hack@DAC SoCs and OR1200
#   hwe-bench                real RTL bug fixes (JSONL from Hugging Face; RTL files are
#                            fetched later by tools/build_benchmarks.py)
#
# Usage: tools/fetch_benchmarks.sh        (safe to re-run; skips what is already there)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"

VB_URL="https://github.com/HWSec-UNC/verification-benchmarks.git"
VB_COMMIT="f5da736829b4020fe76ec1d13cd2761199efde31"
HWE_DATASET="henryen/hwe-bench"
HWE_REVISION="82a42e0a05719366a326e09ddc668ea0d46c91f6"
HWE_FILES=(hwe_bench_full.jsonl)

mkdir -p "$RAW"

echo "== verification-benchmarks @ ${VB_COMMIT:0:12}"
VB_DIR="$RAW/verification-benchmarks"
if [ ! -d "$VB_DIR/.git" ]; then
    git clone --quiet "$VB_URL" "$VB_DIR"
fi
git -C "$VB_DIR" fetch --quiet origin
git -C "$VB_DIR" -c advice.detachedHead=false checkout --quiet "$VB_COMMIT"
git -C "$VB_DIR" submodule update --quiet --init --recursive
du -sh "$VB_DIR" | cut -f1 | sed 's/^/   size: /'

echo "== hwe-bench @ ${HWE_REVISION:0:12}"
HWE_DIR="$RAW/hwe-bench"
mkdir -p "$HWE_DIR"
for f in "${HWE_FILES[@]}"; do
    if [ ! -s "$HWE_DIR/$f" ]; then
        curl -fsSL --retry 3 \
            "https://huggingface.co/datasets/$HWE_DATASET/resolve/$HWE_REVISION/$f" \
            -o "$HWE_DIR/$f.part"
        mv "$HWE_DIR/$f.part" "$HWE_DIR/$f"
    fi
    echo "   $f: $(wc -l < "$HWE_DIR/$f") cases"
done

echo "done: $RAW"
