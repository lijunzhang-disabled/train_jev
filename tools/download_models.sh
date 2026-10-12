#!/usr/bin/env bash
# Download Qwen3 checkpoints into the Hugging Face cache (~/.cache/huggingface; set HF_HOME to move it).
# Sizes: Qwen3-0.6B ~1.5 GB, Qwen3-1.7B ~4 GB, Qwen3-8B ~16 GB.
#
# Usage: tools/download_models.sh [model ...]     (default: Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B)
set -euo pipefail

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B)
fi
for m in "${MODELS[@]}"; do
    echo "== $m"
    hf download "$m" --quiet
done
echo "done; check with: python tools/check_env.py --skip-bench"
