#!/usr/bin/env bash
# CPU smoke test for the dev box (no GPU): the whole pipeline on Qwen3-0.6B with tiny settings.
# Builds a 1.5k-token dataset, trains 3 steps with 2 DDP processes (gloo), then evaluates 6 examples.
# Takes about 5 minutes on 32 cores.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-models/Qwen3-0.6B}   # from tools/download_models.sh
DATA=data/phase0_cpu
OUT=runs/smoke_cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

[ -f "$DATA/train.jsonl" ] || "$PYTHON" tools/build_phase0_data.py --tokenizer "$MODEL" --max-seq-tokens 1536 --out "$DATA"
rm -rf "$OUT"
RUN=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node 2)
"${RUN[@]}" -m jev.train --model "$MODEL" --train "$DATA/train.jsonl" --val "$DATA/val.jsonl" --out "$OUT" \
    --max-seq-tokens 1536 --max-steps 3 --grad-accum 2 --limit-train 16 --val-limit 6 --log-every 1 \
    --lora-r 8 --lora-alpha 16 --gradient-checkpointing
"${RUN[@]}" -m jev.eval --model "$MODEL" --checkpoint "$OUT/final" --max-seq-tokens 1536 --limit 6 \
    --data "$DATA/eval_hwe.jsonl" "$DATA/eval_vb.jsonl" --out "$OUT/eval"
echo "CPU smoke test OK"
