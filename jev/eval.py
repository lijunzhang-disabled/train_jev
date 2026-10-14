"""Evaluate a checkpoint (or the bare base model, zero-shot) on one or more example files.

    torchrun --nproc_per_node 4 -m jev.eval --model models/Qwen3-8B --checkpoint runs/x/final \\
        --data data/phase0/eval_hwe.jsonl data/phase0/eval_vb.jsonl --out runs/x/eval

Writes <out>/<file stem>.predictions.jsonl and <out>/<file stem>.metrics.json, and prints the metrics.

--pointer-readout lm (eval only) answers pointer questions from the LM's own text instead of the pointer
head, which is untrained, so exactly uniform, in a zero-shot run. That gives bug location a real zero-shot number.
"""
import argparse
import json
import os
import time

import torch

from jev import distributed
from jev.data import JevDataset
from jev.format import Renderer
from jev.metrics import summarize
from jev.model import load_for_inference


class LMPointerReadout:
    """Scores each option of a pointer question by the log-probability that the LM writes it right after the
    answer slot: " 12" or " none", as the prompt asks for "a block number, or none". A number must end there,
    so the next token can't be another digit. The eval softmax then renormalizes over the options."""

    def __init__(self, tok):
        self.tok = tok
        self.digits = [tok(str(d), add_special_tokens=False)["input_ids"] for d in range(10)]
        if any(len(ids) != 1 for ids in self.digits):
            raise ValueError("--pointer-readout lm needs a tokenizer that writes each digit as one token")
        self.digits = [ids[0] for ids in self.digits]

    @torch.no_grad()
    def __call__(self, clm, prefix, options):
        """prefix: [1, L] token ids up to and including the answer slot. Returns one score per option."""
        out = clm(input_ids=prefix, use_cache=True, logits_to_keep=1)
        cache, first = out.past_key_values, out.logits[0, -1].float().log_softmax(-1)
        scores = []
        for name in options:
            cont = self.tok(" " + name, add_special_tokens=False)["input_ids"]
            score = first[cont[0]]
            if name.isdigit():  # feed the answer to score its later tokens and that it stops there
                step = clm(input_ids=torch.tensor([cont], device=prefix.device), past_key_values=cache,
                           use_cache=True).logits[0].float().log_softmax(-1)
                cache.crop(-len(cont))
                score = score + step[torch.arange(len(cont) - 1), cont[1:]].sum()
                score = score + torch.log1p(-step[-1, self.digits].exp().sum().clamp(max=1 - 1e-6))
            scores.append(score)
        return torch.stack(scores)


@torch.no_grad()
def predict(model, dataset, rank, world, device, lm_pointer=None):
    """Each rank scores every world-th example; every rank gets all predictions back.
    lm_pointer: an LMPointerReadout to answer pointer questions from the LM's text instead of the head."""
    module = model.module if hasattr(model, "module") else model
    was_training = module.training
    module.eval()
    preds = []
    for i in range(rank, len(dataset), world):
        item = dataset[i]
        ids = torch.tensor([item["input_ids"]], device=device)
        logits = module(ids, torch.ones_like(ids), [item["questions"]])[0]
        for q, lg in zip(item["questions"], logits):
            if lm_pointer and q["type"] == "pointer":
                lg = lm_pointer(module.causal_lm(), ids[:, :q["pos"] + 1], q["options"])
            preds.append({"id": item["id"], "qid": q["id"], "type": q["type"], "options": q["options"],
                          "probs": torch.softmax(lg, -1).tolist(), "target": q.get("target")})
    module.train(was_training)
    return distributed.gather(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="base model directory (or Hugging Face id), e.g. models/Qwen3-8B")
    ap.add_argument("--checkpoint", default=None, help="dir written by jev.train (omit for zero-shot)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="only the first N examples of each file")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--pointer-readout", choices=["head", "lm"], default="head",
                    help="lm: answer pointer questions from the LM's text answer (eval only; see above)")
    args = ap.parse_args()

    rank, world, device = distributed.setup()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    renderer = Renderer(tok, args.max_seq_tokens)
    model = load_for_inference(args.model, args.checkpoint, device, args.attn)
    lm_pointer = LMPointerReadout(tok) if args.pointer_readout == "lm" else None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        mode = f"checkpoint {args.checkpoint}" if args.checkpoint else "zero-shot"
        mode += (", pointer questions read from the LM's text answer" if lm_pointer
                 else "" if args.checkpoint else " (pointer head uniform = chance)")
        print(f"model {args.model}, {mode}, {world} process(es) on {device.type}", flush=True)

    for path in args.data:
        ds = JevDataset(path, renderer, args.limit)
        t0 = time.time()
        preds = predict(model, ds, rank, world, device, lm_pointer)
        if rank == 0:
            stem = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(args.out, f"{stem}.predictions.jsonl"), "w") as fh:
                for p in preds:
                    fh.write(json.dumps(p) + "\n")
            report = summarize(preds)
            with open(os.path.join(args.out, f"{stem}.metrics.json"), "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"\n== {path}: {len(ds)} examples ({ds.dropped} too long), {time.time() - t0:.0f}s", flush=True)
            for qid, m in report.items():
                extra = "".join(f"  {k} {m[k]:.3f}" for k in ("macro_f1", "auroc_yes", "top3_accuracy")
                                if m.get(k) is not None)
                print(f"   {qid:18} n={m['n']:5}  acc {m['accuracy']:.3f}{extra}  nll {m['nll']:.3f}  "
                      f"ece {m['ece']:.3f}", flush=True)
    distributed.cleanup()


if __name__ == "__main__":
    main()
