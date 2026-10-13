#!/usr/bin/env python3
"""Turn data/benchmarks/benchmarks.jsonl into Phase 0 examples (spec/data_format.md).

    python tools/build_phase0_data.py --tokenizer models/Qwen3-8B --max-seq-tokens 8192 --out data/phase0

Needs `transformers` for the tokenizer. Writes train.jsonl, val.jsonl, eval_hwe.jsonl, eval_vb.jsonl and
stats.json. Every example is rendered with jev/format.py to check that it fits in --max-seq-tokens.

Per changed RTL file of a hwe-bench bug fix (generated "autogen" copies are skipped):
  buggy      a window of the buggy file containing a changed region, placed at a random offset so the
             bug's position in the window carries no signal
  fixed      the matching window of the fixed file
  elsewhere  a window of the buggy file away from every changed region (weight 0.5: "no bug" there is
             likely but not certain)
Per verification-benchmarks property: the module's file (or the window that mentions the most of the
property's signals) and a property_violated question.
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jev.format import NONE, Renderer  # noqa: E402

CATEGORIES = {
    "RTL_LOGIC": "logic error",
    "RTL_INTERFACE": "interface or protocol mismatch",
    "RTL_SPEC": "behavior that deviates from the specification",
    "RTL_CONFIG_INTEG": "configuration or integration error",
    "RTL_TIMING_SYNC": "timing or synchronization error",
}
NO_BUG = "no bug"
HWE_QUESTIONS = [
    {"id": "bug_present", "type": "choice", "text": "Does this code contain a functional bug?",
     "options": ["yes", "no"]},
    {"id": "bug_category", "type": "choice", "text": "What kind of bug does this code contain?",
     "options": [NO_BUG] + list(CATEGORIES.values())},
    {"id": "bug_location", "type": "pointer",
     "text": "Which block contains the bug? Point to a block, or to none if there is no bug."},
]
RESERVE_TOKENS = 600     # chat template, header, questions and answer slots
MARKER_TOKENS = 8        # one "// [block j]" line
ELSEWHERE_MARGIN = 5     # lines kept clear around changed regions for "elsewhere" windows


def read_lines(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as fh:
        return fh.read().replace("\r\n", "\n").split("\n")


def make_blocks(n_lines, block_lines):
    return [[s, min(s + block_lines - 1, n_lines)] for s in range(1, n_lines + 1, block_lines)]


class Windower:
    """Picks line windows whose rendered size fits the token budget."""

    def __init__(self, tok, max_seq_tokens, block_lines):
        self.tok, self.block_lines = tok, block_lines
        self.budget = max_seq_tokens - RESERVE_TOKENS

    def line_costs(self, lines):
        ids = self.tok([line + "\n" for line in lines], add_special_tokens=False)["input_ids"]
        prefix = [0]
        for x in ids:
            prefix.append(prefix[-1] + len(x))
        return prefix

    def cost(self, prefix, s, e):  # 1-based inclusive line range
        n_blocks = -(-(e - s + 1) // self.block_lines)
        return prefix[e] - prefix[s - 1] + n_blocks * MARKER_TOKENS

    def max_end(self, prefix, s, n):
        """Largest e >= s with cost(s, e) <= budget, or s - 1 if even one line is too big."""
        lo, hi = s - 1, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.cost(prefix, s, mid) <= self.budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def containing(self, prefix, n, a, b, rng):
        """A window containing lines a..b at a random offset, or None if a..b alone is too big."""
        if self.cost(prefix, 1, n) <= self.budget:
            return 1, n
        valid = [s for s in range(1, a + 1) if self.max_end(prefix, s, n) >= b]
        if not valid:
            return None
        s = rng.choice(valid)
        return s, self.max_end(prefix, s, n)

    def avoiding(self, prefix, n, regions, rng):
        """A full-budget window that stays clear of every region, or None."""
        if self.cost(prefix, 1, n) <= self.budget:
            return None  # the whole file fits, so any window would include the bug
        blocked = [(a - ELSEWHERE_MARGIN, b + ELSEWHERE_MARGIN) for a, b in regions]
        valid = []
        for s in range(1, n + 1):
            e = self.max_end(prefix, s, n)
            if e >= s and all(e < lo or s > hi for lo, hi in blocked):
                valid.append((s, e))
        return rng.choice(valid) if valid else None


def line_map(old, new):
    """Map each old line (1-based) to the corresponding new line."""
    mapping = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        for i in range(i1, i2):
            mapping[i + 1] = j1 + (i - i1) + 1 if tag == "equal" else j1 + 1
    mapping[len(old) + 1] = len(new) + 1
    return mapping


def hwe_examples(record, windower, block_lines, rng, stats):
    group = record["id"]
    category = CATEGORIES[record["bug"]["category"]]
    out = []
    for j, f in enumerate(record["rtl"]):
        if f["autogen"]:
            stats["skipped autogen copies"] += 1
            continue
        if not f["fixed"] or not f["changed_lines"]:
            stats["skipped files without a fix"] += 1
            continue
        buggy, fixed = read_lines(f["buggy"]), read_lines(f["fixed"])
        regions = f["changed_lines"]
        prefix = windower.line_costs(buggy)
        a, b = rng.choice(regions)
        win = windower.containing(prefix, len(buggy), a, b, rng)
        if win is None:
            stats["skipped: changed region alone exceeds the budget"] += 1
            continue
        s, e = win

        # buggy window: label the blocks that overlap changed lines inside the window
        blocks = make_blocks(e - s + 1, block_lines)
        hit = collections.Counter()
        for ra, rb in regions:
            for k, (bs, be) in enumerate(blocks):
                if max(ra, s + bs - 1) <= min(rb, s + be - 1):
                    hit[str(k)] += 1
        location = {k: 1.0 / len(hit) for k in hit}
        out.append(example(f"{group}/{j}/buggy", group, "buggy", f["file"], buggy[s - 1:e], blocks,
                           present="yes", category=category, location=location))

        # fixed window: same content region, re-fitted to the budget in the fixed file
        m = line_map(buggy, fixed)
        fs = m[s]
        fprefix = windower.line_costs(fixed)
        fe = windower.max_end(fprefix, fs, len(fixed))
        if fe >= fs:
            fblocks = make_blocks(fe - fs + 1, block_lines)
            out.append(example(f"{group}/{j}/fixed", group, "fixed", f["file"], fixed[fs - 1:fe], fblocks,
                               present="no", category=NO_BUG, location={NONE: 1.0}))

        away = windower.avoiding(prefix, len(buggy), regions, rng)
        if away:
            ws, we = away
            wblocks = make_blocks(we - ws + 1, block_lines)
            out.append(example(f"{group}/{j}/elsewhere", group, "elsewhere", f["file"], buggy[ws - 1:we],
                               wblocks, present="no", category=NO_BUG, location={NONE: 1.0}, weight=0.5))
    return out


def example(eid, group, kind, path, lines, blocks, present, category, location, weight=1.0):
    def label(dist, source):
        return {"dist": dist, "source": source, "weight": weight}
    return {
        "id": eid, "group": group, "source": "hwe-bench", "kind": kind,
        "state": {"path": path, "code": "\n".join(lines), "blocks": blocks},
        "questions": HWE_QUESTIONS,
        "labels": {
            "bug_present": label({present: 1.0}, "construction"),
            "bug_category": label({category: 1.0}, "hwe-bench" if present == "yes" else "construction"),
            "bug_location": label(location, "construction"),
        },
    }


def vb_example(record, windower, block_lines, stats):
    if not record["rtl"]:
        stats["skipped VB records without RTL"] += 1
        return None
    f = record["rtl"][0]  # the most-referenced module
    lines = read_lines(f["buggy"])
    prefix = windower.line_costs(lines)
    n = len(lines)
    if windower.cost(prefix, 1, n) <= windower.budget:
        s, e = 1, n
    else:  # the window that mentions the most of the property's identifiers
        words = set(re.findall(r"[A-Za-z_]\w*", record["property"]["text"]))
        best = None
        for s0 in range(1, n + 1, 8):
            e0 = windower.max_end(prefix, s0, n)
            if e0 < s0:
                continue
            seen = set(re.findall(r"[A-Za-z_]\w*", "\n".join(lines[s0 - 1:e0]))) & words
            if best is None or len(seen) > best[0]:
                best = (len(seen), s0, e0)
        _, s, e = best
        stats["VB windows cropped"] += 1
    prop = record["property"]["text"]
    return {
        "id": record["id"], "group": record["id"], "source": "verification-benchmarks", "kind": "property",
        "state": {"path": f["file"], "code": "\n".join(lines[s - 1:e]),
                  "blocks": make_blocks(e - s + 1, block_lines)},
        "questions": [{"id": "property_violated", "type": "choice", "options": ["yes", "no"],
                       "text": "A formal tool checks this property on the design:\n" + prop +
                               "\nWill it find a violation?"}],
        "labels": {"property_violated": {
            "dist": {"yes" if record["property"]["fpv_result"] == "violation" else "no": 1.0},
            "source": "jaspergold", "weight": 1.0}},
    }


def split_of(record, val_fraction):
    if record["split"] == "test":
        return "eval_vb" if record["source"] == "verification-benchmarks" else "eval_hwe"
    h = int(hashlib.sha256(record["id"].encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-8B"))
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--block-lines", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--benchmarks", default=os.path.join(ROOT, "data", "benchmarks", "benchmarks.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "phase0"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Renderer(tok, args.max_seq_tokens)
    windower = Windower(tok, args.max_seq_tokens, args.block_lines)
    stats = collections.Counter()
    splits = collections.defaultdict(list)
    lengths = collections.defaultdict(list)

    for line in open(args.benchmarks, encoding="utf-8"):
        record = json.loads(line)
        rng = random.Random(f"{args.seed}/{record['id']}")
        if record["source"] == "hwe-bench":
            examples = hwe_examples(record, windower, args.block_lines, rng, stats)
        else:
            ex = vb_example(record, windower, args.block_lines, stats)
            examples = [ex] if ex else []
        split = split_of(record, args.val_fraction)
        for ex in examples:
            rendered = renderer.render(ex)
            if rendered is None:
                stats["dropped: rendered example too long"] += 1
                continue
            splits[split].append(ex)
            lengths[split].append(len(rendered["input_ids"]))

    os.makedirs(args.out, exist_ok=True)
    summary = {"tokenizer": args.tokenizer, "max_seq_tokens": args.max_seq_tokens,
               "block_lines": args.block_lines, "notes": dict(stats), "splits": {}}
    for name in ("train", "val", "eval_hwe", "eval_vb"):
        exs = splits[name]
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for ex in exs:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        lens = sorted(lengths[name])
        first_q = [ex["questions"][0]["id"] for ex in exs]
        answers = collections.Counter(next(iter(ex["labels"][q]["dist"])) for ex, q in zip(exs, first_q))
        summary["splits"][name] = {
            "examples": len(exs),
            "groups": len({ex["group"] for ex in exs}),
            "kinds": dict(collections.Counter(ex["kind"] for ex in exs)),
            "first_question_answers": dict(answers),
            "tokens": sum(lens),
            "tokens_median": lens[len(lens) // 2] if lens else 0,
            "tokens_max": lens[-1] if lens else 0,
        }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
