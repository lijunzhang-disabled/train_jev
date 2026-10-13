#!/usr/bin/env bash
# Step 8 of docs/phase0_guide.md: a short end-to-end run on GPUs with a small model.
# Trains 20 LoRA steps on Qwen3-0.6B, validates, saves, then evaluates on 64 examples of each test set.
#
#   NGPU=2 launch/smoke_gpu.sh
#   CUDA_VISIBLE_DEVICES=2,3 NGPU=2 launch/smoke_gpu.sh      # pick specific cards
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
# No Hugging Face requests during runs (a slow proxy would stall them); models load from models/.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
NGPU=${NGPU:-2}
MODEL=${MODEL:-models/Qwen3-0.6B}   # from tools/download_models.sh
DATA=${DATA:-data/phase0}
OUT=${OUT:-runs/smoke_gpu}
MAX_SEQ=${MAX_SEQ:-8192}
STEPS=${STEPS:-20}
LIMIT=${LIMIT:-64}

[ -d "$MODEL" ] || [[ "$MODEL" != models/* ]] || { echo "$MODEL not found: run tools/download_models.sh" >&2; exit 1; }
if [ ! -f "$DATA/train.jsonl" ]; then
    "$PYTHON" tools/build_phase0_data.py --tokenizer "$MODEL" --max-seq-tokens "$MAX_SEQ" --out "$DATA"
fi
mkdir -p "$OUT"
RUN=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NGPU")

"${RUN[@]}" -m jev.train --model "$MODEL" --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --out "$OUT" \
    --max-seq-tokens "$MAX_SEQ" --max-steps "$STEPS" --grad-accum 2 --eval-every $((STEPS / 2)) \
    --val-limit 32 --log-every 2 --gradient-checkpointing 2>&1 | tee "$OUT/train.log"

"${RUN[@]}" -m jev.eval --model "$MODEL" --checkpoint "$OUT/final" --max-seq-tokens "$MAX_SEQ" --limit "$LIMIT" \
    --data "$DATA/eval_hwe.jsonl" "$DATA/eval_vb.jsonl" --out "$OUT/eval" 2>&1 | tee "$OUT/eval.log"

echo "GPU smoke run OK: logs in $OUT/train.log and $OUT/eval.log"
