"""Fine-tune the decision model with LoRA: one process, or DDP when launched with torchrun.

    torchrun --nproc_per_node 4 -m jev.train --model models/Qwen3-8B \\
        --train data/phase0/train.jsonl --val data/phase0/val.jsonl --out runs/qwen3-8b-lora-v0 --epochs 2

One optimizer step = batch_size x grad_accum sequences per process x number of processes.
Writes <out>/log.jsonl, <out>/step-N/ every --save-every steps, and <out>/final/.
"""
import argparse
import contextlib
import json
import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from jev import distributed
from jev.data import JevDataset, collate
from jev.eval import predict
from jev.format import Renderer
from jev.metrics import summarize
from jev.model import JevModel, add_lora, load_lm, save_checkpoint, soft_cross_entropy


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="base model directory (or Hugging Face id), e.g. models/Qwen3-8B")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    ap.add_argument("--batch-size", type=int, default=1, help="sequences per process per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="pointer head learning rate")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--attn", default="sdpa", help="sdpa (default) or flash_attention_2 if installed")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=0, help="validate every N steps (0: only at the end)")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None, help="use only the first N training examples")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def batch_loss(logits, items, device):
    """Weighted mean over all questions in the micro-batch of the soft-label cross-entropy."""
    total, weights = 0.0, 0.0
    for row, item in zip(logits, items):
        for lg, q in zip(row, item["questions"]):
            w = q.get("weight", 1.0)
            total = total + w * soft_cross_entropy(lg, torch.tensor(q["target"], device=device))
            weights += w
    return total / max(weights, 1e-8)


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # cosine down to 10%
    return f


def validate(model, val_ds, rank, world, device, step, log):
    preds = predict(model, val_ds, rank, world, device)
    if rank == 0:
        report = summarize(preds)
        log({"step": step, "val": report})
        line = "  ".join(f"{qid} acc {m['accuracy']:.3f} nll {m['nll']:.3f}" for qid, m in report.items())
        print(f"   val @ step {step}: {line}", flush=True)


def main():
    args = parse_args()
    rank, world, device = distributed.setup()
    torch.manual_seed(args.seed)

    def say(msg):
        if rank == 0:
            print(msg, flush=True)

    log_path = os.path.join(args.out, "log.jsonl")
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)

    def log(record):
        if rank == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    renderer = Renderer(tok, args.max_seq_tokens)
    train_ds = JevDataset(args.train, renderer, args.limit_train)
    val_ds = JevDataset(args.val, renderer, args.val_limit) if args.val else None
    say(f"train: {len(train_ds)} examples, {train_ds.tokens():,} tokens ({train_ds.dropped} too long)"
        + (f" | val: {len(val_ds)} examples" if val_ds else ""))

    lm = load_lm(args.model, device, args.attn)
    if args.gradient_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm = add_lora(lm, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = JevModel(lm).to(device)
    lora_params = [p for p in model.lm.parameters() if p.requires_grad]
    say(f"model {args.model} on {world} x {device.type}; trainable LoRA params "
        f"{sum(p.numel() for p in lora_params) / 1e6:.1f}M + pointer head "
        f"{sum(p.numel() for p in model.heads()) / 1e6:.2f}M")
    groups = [{"params": lora_params, "lr": args.lr}, {"params": model.heads(), "lr": args.head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    ddp = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model

    if world > 1:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    else:
        sampler = RandomSampler(train_ds, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=lambda b: collate(b, pad_id))
    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, total))
    say(f"{steps_per_epoch} steps per epoch, {total} steps total "
        f"({args.batch_size * args.grad_accum * world} sequences per step)")

    run_config = {**vars(args), "world_size": world, "train_examples": len(train_ds),
                  "train_tokens": train_ds.tokens(), "total_steps": total}
    step, micro, epoch = 0, 0, 0
    losses, tokens, t_start, t_log = [], 0, time.time(), time.time()
    model.train()
    while step < total:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            micro += 1
            sync = micro % args.grad_accum == 0
            with ddp.no_sync() if (world > 1 and not sync) else contextlib.nullcontext():
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = ddp(ids, mask, [it["questions"] for it in batch["items"]])
                loss = batch_loss(logits, batch["items"], device)
                # keep every head parameter in the graph, so DDP never waits for a gradient that doesn't come
                loss = loss + 0.0 * sum(p.sum() for p in model.heads())
                (loss / args.grad_accum).backward()
            losses.append(loss.item())
            tokens += int(mask.sum())
            if not sync:
                continue

            torch.nn.utils.clip_grad_norm_(lora_params + model.heads(), args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total:
                mean_loss = distributed.all_sum(sum(losses) / len(losses), device) / world
                tok_s = distributed.all_sum(tokens, device) / (time.time() - t_log)
                mem = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
                say(f"step {step}/{total}  loss {mean_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                    f"tok/s {tok_s:,.0f}  mem {mem:.1f}G  elapsed {(time.time() - t_start) / 60:.1f}m")
                log({"step": step, "epoch": epoch, "loss": mean_loss, "lr": sched.get_last_lr()[0],
                     "tokens_per_s": tok_s, "max_mem_gb": mem})
                losses, tokens, t_log = [], 0, time.time()
            if val_ds and args.eval_every and step % args.eval_every == 0 and step < total:
                validate(ddp, val_ds, rank, world, device, step, log)
            if args.save_every and step % args.save_every == 0 and step < total:
                if rank == 0:
                    save_checkpoint(model, os.path.join(args.out, f"step-{step}"), {**run_config, "step": step})
                distributed.barrier()
            if step >= total:
                break
        epoch += 1

    if rank == 0:
        save_checkpoint(model, os.path.join(args.out, "final"), {**run_config, "step": step})
        say(f"saved {os.path.join(args.out, 'final')}  ({(time.time() - t_start) / 60:.1f} min)")
    distributed.barrier()
    if val_ds:
        validate(ddp, val_ds, rank, world, device, step, log)
    distributed.cleanup()


if __name__ == "__main__":
    main()
