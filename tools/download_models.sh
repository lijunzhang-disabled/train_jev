#!/usr/bin/env bash
# Download Qwen3 checkpoints into models/<name> (e.g. models/Qwen3-8B), where every script loads them from.
# Sizes: Qwen3-0.6B ~1.5 GB, Qwen3-1.7B ~4 GB, Qwen3-8B ~16 GB. To keep them on a bigger disk, first run
#   ln -s /big/disk/models models
#
# Usage: tools/download_models.sh [model ...]     (default: Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B)
#
# The source is ModelScope by default. Its Qwen3 weights, tokenizer and configs are byte-identical to
# Hugging Face's (checked 2026-10-13). To use Hugging Face instead:
#   SOURCE=hf tools/download_models.sh
#   SOURCE=hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 tools/download_models.sh   # via a mirror
# Re-running is safe: finished files are skipped and partial ones resume.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE=${SOURCE:-modelscope}
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B)
fi
case "$SOURCE" in
    modelscope) TOOL=modelscope; ENDPOINT=${MODELSCOPE_DOMAIN:-modelscope.cn} ;;
    hf) TOOL=hf; ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} ;;
    *) echo "SOURCE must be modelscope or hf, not '$SOURCE'" >&2; exit 2 ;;
esac
command -v "$TOOL" > /dev/null || { echo "'$TOOL' not found: activate the env and pip install -r requirements.txt" >&2; exit 1; }
PROXY=${https_proxy:-${HTTPS_PROXY:-${all_proxy:-${ALL_PROXY:-}}}}
PROXY_HOST=${PROXY#*://}
PROXY_HOST=${PROXY_HOST##*@}   # never print proxy credentials
echo "source: $SOURCE ($ENDPOINT)${PROXY:+, via proxy $PROXY_HOST}"

for m in "${MODELS[@]}"; do
    dest="models/${m##*/}"
    echo "== $m -> $dest"
    if [ "$SOURCE" = modelscope ]; then
        cmd=(modelscope download "$m" --local-dir "$dest")
    else
        cmd=(hf download "$m" --local-dir "$dest" --quiet)
    fi
    for attempt in 1 2 3; do
        "${cmd[@]}" && break
        if [ "$attempt" -eq 3 ]; then
            echo "FAILED: could not download $m from $SOURCE after 3 attempts." >&2
            echo "  See docs/phase0_guide.md, step 4, for proxy problems and the other sources." >&2
            exit 1
        fi
        echo "   attempt $attempt failed; retrying in 15 s (finished files are kept)"
        sleep 15
    done
done
echo "done; check with: python tools/check_env.py --skip-bench"
