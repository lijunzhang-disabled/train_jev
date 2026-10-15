#!/usr/bin/env bash
# Evaluate a checkpoint on every $DATA/eval_*.jsonl, or the base model zero-shot when CHECKPOINT is unset.
#
#   NGPU=4 launch/eval.sh                                                   # zero-shot Qwen3-8B baseline
#   NGPU=4 POINTER=lm launch/eval.sh       # zero-shot, bug location read from the LM's text answer
#   NGPU=4 CHECKPOINT=runs/qwen3-8b-lora-v1/final OUT=runs/qwen3-8b-lora-v1/eval launch/eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
# No Hugging Face requests during runs (a slow proxy would stall them); models load from models/.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
NGPU=${NGPU:-4}
MODEL=${MODEL:-models/Qwen3-8B}   # from tools/download_models.sh
DATA=${DATA:-data/phase0}
CHECKPOINT=${CHECKPOINT:-}
POINTER=${POINTER:-head}   # head | lm (see jev/eval.py)
suffix=; [ "$POINTER" = lm ] && suffix=-lmpointer
OUT=${OUT:-runs/zeroshot-$(basename "$MODEL")$suffix}
MAX_SEQ=${MAX_SEQ:-8192}
LIMIT=${LIMIT:-}

[ -d "$MODEL" ] || [[ "$MODEL" != models/* ]] || { echo "$MODEL not found: run tools/download_models.sh" >&2; exit 1; }
args=(--model "$MODEL" --max-seq-tokens "$MAX_SEQ" --data "$DATA"/eval_*.jsonl --out "$OUT"
      --pointer-readout "$POINTER")
[ -n "$CHECKPOINT" ] && args+=(--checkpoint "$CHECKPOINT")
[ -n "$LIMIT" ] && args+=(--limit "$LIMIT")
mkdir -p "$OUT"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" -m jev.eval "${args[@]}" 2>&1 \
    | tee "$OUT/eval.log"
