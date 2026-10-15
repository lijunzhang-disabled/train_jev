#!/usr/bin/env bash
# Download the labeled Hugging Face sets that tools/build_labeled_sets.py and tools/build_mutants.py read, into
# data/raw/ (git-ignored), each pinned to a fixed revision:
#
#   rtl-augmented-v3   injected bugs in 49 GitHub repos, each confirmed by simulation (MIT; patches and
#                      sources only, not the waveforms)
#   equivsva           120 behavior families: 4 equivalent designs, properties and 3 mutants each (CC BY 4.0)
#   hiersva            342 BaseJump STL modules with formally proven assertions, 28 buggy variants (SHL-2.1)
#   openrtlset         13,625 lint-clean Verilog designs with their repo, a second clean pool for mutants
#                      (CC BY 4.0)
#
# Usage: tools/fetch_labeled_sets.sh        (safe to re-run; `hf download` skips files it already has)
#        HF_ENDPOINT=https://hf-mirror.com tools/fetch_labeled_sets.sh    (when the proxy blocks Hugging Face)
# rtl-augmented-v3 is about 6,000 small files; if Hugging Face rate-limits, the CLI waits and retries.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
HF=${HF:-hf}   # the Hugging Face CLI from the jev environment

fetch() {   # fetch <repo> <revision> <dir> [include patterns...]
    local repo=$1 rev=$2 dir=$3
    shift 3
    echo "== $repo @ ${rev:0:12}"
    local inc=()
    [ $# -gt 0 ] && inc=(--include "$@")
    "$HF" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$RAW/$dir" "${inc[@]}" --quiet
    echo "   $(find "$RAW/$dir" -type f -not -path '*/.cache/*' | wc -l) files in data/raw/$dir"
}

fetch architect-ubc-capstone/rtl-augmented-v3 7ff786193821cf0fcdf750039831fba4b0ff011f rtl-augmented-v3 \
    "README.md" "*/simulations.jsonl" "*.patch" "*/metadata.json" "*.v" "*.sv" "*.svh" "*.vh"
fetch aditigupta/EquivSVA fe2158ac0ea3733e8696a2bd2a31025bab8ec60f equivsva
fetch AnonymousHierSVA/HierSVA ef49c97247badca6e6fe5f5a851371af0e88a259 hiersva
fetch theepicflyer/openrtlset-permissive-verified c378919c7e0f9105a60434931b402ba57a535412 openrtlset

# the single-file sets are checked byte for byte
cd "$RAW"
sha256sum --check --quiet <<'EOF' || { echo "FAILED: a file differs from the pinned version; delete it and rerun" >&2; exit 1; }
b862e5c0759c978a3c376c293fb2af76b43fbd3f2ed4fcdf88e997dfa91cd0ea  equivsva/train.jsonl
b552b055dd71018d439f7822a205c06a6ff53d1fc66c3891180aa0e2995649b0  equivsva/dev.jsonl
f67b9ad59dedf521f418a377e61bb09d2d421e4326bf1a0b53b1c1652d07108c  equivsva/test.jsonl
6b28b25aee602e304d03908651eebde3e68e8a87161c44cd464fceafe12f80cc  openrtlset/data.jsonl
EOF
echo "done: $RAW"
