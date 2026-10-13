#!/usr/bin/env bash
# Step 9 of docs/phase0_guide.md: the first real run. LoRA on Qwen3-8B over the Phase 0 training set,
# then evaluation on both test sets. Keeps 16 sequences per optimizer step whatever NGPU is.
#
#   NGPU=4 launch/train.sh
#   NGPU=2 RUN=runs/qwen3-8b-lora-v0-2gpu launch/train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
# No Hugging Face requests during runs (a slow proxy would stall them); models load from models/.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
NGPU=${NGPU:-4}
MODEL=${MODEL:-models/Qwen3-8B}   # from tools/download_models.sh
DATA=${DATA:-data/phase0}
RUN=${RUN:-runs/qwen3-8b-lora-v0}
EPOCHS=${EPOCHS:-2}
SEQ_PER_STEP=${SEQ_PER_STEP:-16}
GRAD_ACCUM=${GRAD_ACCUM:-$((SEQ_PER_STEP / NGPU))}
LR=${LR:-1e-4}
MAX_SEQ=${MAX_SEQ:-8192}
EXTRA=${EXTRA:-}   # extra jev.train flags, e.g. EXTRA="--max-steps 10"
EVAL_LIMIT=${EVAL_LIMIT:-}   # evaluate only the first N examples of each test set

[ -d "$MODEL" ] || [[ "$MODEL" != models/* ]] || { echo "$MODEL not found: run tools/download_models.sh" >&2; exit 1; }
mkdir -p "$RUN"
RUNNER=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NGPU")

# shellcheck disable=SC2086
"${RUNNER[@]}" -m jev.train --model "$MODEL" --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --out "$RUN" \
    --max-seq-tokens "$MAX_SEQ" --epochs "$EPOCHS" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --gradient-checkpointing --eval-every 25 --save-every 50 $EXTRA 2>&1 | tee "$RUN/train.log"

eval_args=()
[ -n "$EVAL_LIMIT" ] && eval_args=(--limit "$EVAL_LIMIT")
"${RUNNER[@]}" -m jev.eval --model "$MODEL" --checkpoint "$RUN/final" --max-seq-tokens "$MAX_SEQ" "${eval_args[@]}" \
    --data "$DATA/eval_hwe.jsonl" "$DATA/eval_vb.jsonl" --out "$RUN/eval" 2>&1 | tee "$RUN/eval.log"

echo "done: $RUN (train.log, eval.log, eval/*.metrics.json)"
