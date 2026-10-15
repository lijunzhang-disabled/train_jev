#!/usr/bin/env python3
"""Turn the labeled Hugging Face sets (tools/fetch_labeled_sets.sh) into examples (spec/data_format.md).

    python tools/build_labeled_sets.py --jobs 8 --out data/labeled_v1

Needs Yosys on PATH for EquivSVA's property labels (--yosys picks another binary) and `transformers` for the
tokenizer.

rtl-augmented-v3  Every injected bug that simulation confirmed (simulation_status "sim_ok"): the buggy file
                  ("buggy") and its original ("clean"). A file too long for --max-seq-tokens becomes a window
                  around the bug, paired with the same window of the original ("fixed"). Comments are stripped
                  from both, because some injected lines carry a comment that describes the bug.
EquivSVA          Per behavior family: its 4 equivalent designs (the one the mutants were made from is "clean",
                  the others "variant") and its 3 mutants ("buggy", renamed to the design they were made from:
                  their module names say "mutant"). Property examples ask property_violated for every property
                  on every design. The equivalent designs satisfy all properties: EquivSVA's formal suite proves
                  it, and Yosys checks it again here as a test of our property harness. Each mutant/property
                  pair is checked with Yosys: a counterexample is "yes", an induction proof "no", anything else
                  is dropped.
HierSVA           Its 28 buggy modules (21 injected, 7 from BaseJump STL's history) with their originals, as bug
                  examples; comments stripped. Its 2,273 proven assertions wait for a check of yosys-slang's
                  SVA support on the server.
Splits: rtl-augmented-v3 by repo; EquivSVA by its own split (its test families go to val); HierSVA to train.
Mutants and injected bugs get a uniform bug_category target over the five bug kinds at weight 0.5, as in
tools/build_mutants.py. Writes train.jsonl, val.jsonl, property_checks.jsonl and stats.json.
"""
import argparse
import collections
import difflib
import glob
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_mutants import ELAB, read_and_remove, run_yosys, first_error  # noqa: E402
from build_phase0_data import (CATEGORIES, HWE_QUESTIONS, NO_BUG, Windower, code_only, code_regions,  # noqa: E402
                               line_map, make_blocks)
from jev.format import NONE  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
UNKNOWN_CATEGORY = {"dist": {c: 1.0 for c in CATEGORIES.values()}, "source": "construction", "weight": 0.5}


# ---------------------------------------------------------------- shared

def strip_comments(lines):
    """Comments removed, trailing spaces trimmed, runs of blank lines collapsed to one."""
    out = []
    for line in code_only(lines):
        line = line.rstrip()
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().replace("\r\n", "\n").split("\n")


def bug_example(eid, group, source, kind, path, lines, block_lines, regions=None):
    """regions: buggy line ranges (1-based, inclusive) in `lines`, or None for a bug-free example."""
    blocks = make_blocks(len(lines), block_lines)
    if regions:
        hit = sorted({k for a, b in regions for k, (s, e) in enumerate(blocks) if max(a, s) <= min(b, e)})
        labels = {"bug_present": {"dist": {"yes": 1.0}, "source": "construction", "weight": 1.0},
                  "bug_category": UNKNOWN_CATEGORY,
                  "bug_location": {"dist": {str(k): 1.0 / len(hit) for k in hit}, "source": "construction",
                                   "weight": 1.0}}
    else:
        labels = {"bug_present": {"dist": {"no": 1.0}, "source": "construction", "weight": 1.0},
                  "bug_category": {"dist": {NO_BUG: 1.0}, "source": "construction", "weight": 1.0},
                  "bug_location": {"dist": {NONE: 1.0}, "source": "construction", "weight": 1.0}}
    return {"id": eid, "group": group, "source": source, "kind": kind,
            "state": {"path": path, "code": "\n".join(lines), "blocks": blocks},
            "questions": HWE_QUESTIONS, "labels": labels}


def pair_examples(group, source, path, buggy, clean, bug, windower, block_lines, rng, stats, clean_done):
    """Examples for one bug: the whole buggy file plus (once per file) the clean one, or, for a file too long
    for the token budget, a window around the bug plus the same window of the clean file."""
    regions = code_regions(buggy, clean)
    if not regions:
        stats[f"{source}: skipped, the bug changes only comments or whitespace"] += 1
        return []
    prefix = windower.line_costs(buggy)
    if windower.cost(prefix, 1, len(buggy)) <= windower.budget:
        out = [bug_example(f"{group}/buggy-{bug}", group, source, "buggy", path, buggy, block_lines, regions)]
        if group not in clean_done:
            clean_done.add(group)
            out.append(bug_example(f"{group}/clean", group, source, "clean", path, clean, block_lines))
        return out
    a, b = rng.choice(regions)
    win = windower.containing(prefix, len(buggy), a, b, rng)
    if win is None:
        stats[f"{source}: skipped, the changed region alone exceeds the budget"] += 1
        return []
    s, e = win
    inside = [(max(ra, s) - s + 1, min(rb, e) - s + 1) for ra, rb in regions if ra <= e and rb >= s]
    out = [bug_example(f"{group}/{bug}/buggy", group, source, "buggy", path, buggy[s - 1:e], block_lines, inside)]
    m = line_map(buggy, clean)
    fs = m[s]
    fe = windower.max_end(windower.line_costs(clean), fs, len(clean))
    if fe >= fs:
        out.append(bug_example(f"{group}/{bug}/fixed", group, source, "fixed", path, clean[fs - 1:fe], block_lines))
    stats[f"{source}: files windowed"] += 1
    return out


def hash_split(key, val_fraction):
    return "val" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000 < val_fraction * 1000 else "train"


# ---------------------------------------------------------------- rtl-augmented-v3

def apply_patch(lines, patch):
    """A unified diff applied to `lines` (hunks in order, exact context), or None if it doesn't apply."""
    out, pos = [], 0
    hunks = re.split(r"^(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*)$", patch, flags=re.M)
    for header, body in zip(hunks[1::2], hunks[2::2]):
        start = int(re.match(r"@@ -(\d+)", header).group(1))
        old, new = [], []
        body_lines = body.split("\n")[1:]          # [0] is the rest of the @@ line
        if body_lines and body_lines[-1] == "":     # the newline that ends the hunk
            body_lines.pop()
        for line in body_lines:
            if line.startswith("\\"):
                continue
            if line.startswith(" ") or line == "":  # some tools write an empty context line as ""
                old.append(line[1:])
                new.append(line[1:])
            elif line.startswith("-"):
                old.append(line[1:])
            elif line.startswith("+"):
                new.append(line[1:])
        at = max(start - 1, 0) if old else start
        if lines[at:at + len(old)] != old:
            return None
        out += lines[pos:at] + new
        pos = at + len(old)
    return out + lines[pos:]


def rtl_augmented(windower, args, stats):
    root = os.path.join(RAW, "rtl-augmented-v3")
    rng = random.Random(f"{args.seed}/rtlaug")
    examples, clean_done = [], set()
    for meta_path in sorted(glob.glob(os.path.join(root, "*", "candidates", "*", "bugs", "*", "metadata.json"))):
        meta = json.load(open(meta_path))
        repo = os.path.relpath(meta_path, root).split(os.sep)[0]
        bug = os.path.basename(os.path.dirname(meta_path))
        module = os.path.relpath(meta_path, root).split(os.sep)[2]
        if meta.get("simulation_status") != "sim_ok" or len(meta.get("files_modified", [])) != 1:
            stats["rtl-augmented-v3: skipped, not confirmed by simulation"] += 1
            continue
        path = meta["files_modified"][0]
        golden = read_lines(os.path.join(root, repo, "source", path))
        patch = open(os.path.join(os.path.dirname(meta_path), "diff.patch"), errors="replace").read()
        buggy = apply_patch(golden, patch)
        if buggy is None:
            stats["rtl-augmented-v3: skipped, patch doesn't apply"] += 1
            continue
        group = f"rtlaug/{repo}/{path}"
        exs = pair_examples(group, "rtl-augmented-v3", path, strip_comments(buggy), strip_comments(golden),
                            f"{module}-{bug}", windower, args.block_lines, rng, stats, clean_done)
        split = hash_split(repo, args.val_fraction)
        for ex in exs:
            if ex["kind"] == "buggy":
                ex["bug_type"] = meta.get("bug_type")
            examples.append((split, ex))
    return examples


# ---------------------------------------------------------------- HierSVA

def hiersva(windower, args, stats):
    root = os.path.join(RAW, "hiersva")
    originals = {}
    for p in glob.glob(os.path.join(root, "rtl", "**", "*.*v"), recursive=True):
        originals.setdefault(os.path.basename(p), p)
    rng = random.Random(f"{args.seed}/hiersva")
    examples, clean_done = [], set()
    for p in sorted(glob.glob(os.path.join(root, "buggy_rtl*", "*", "*.sv")) +
                    glob.glob(os.path.join(root, "buggy_rtl*", "*", "*.v"))):
        name = os.path.basename(p)
        if name not in originals:
            stats["hiersva: skipped, original not found"] += 1
            continue
        buggy = strip_comments(read_lines(p))
        clean = strip_comments(read_lines(originals[name]))
        regions = code_regions(buggy, clean)
        if sum(b - a + 1 for a, b in regions) > max(10, len(buggy) // 4):
            stats["hiersva: skipped, differs from the rtl/ copy beyond the bug"] += 1
            continue
        kind = os.path.basename(os.path.dirname(os.path.dirname(p)))   # buggy_rtl (injected) or buggy_rtl2 (real)
        group = f"hiersva/{os.path.splitext(name)[0]}"
        for ex in pair_examples(group, "hiersva", name, buggy, clean, kind, windower, args.block_lines, rng, stats,
                                clean_done):
            examples.append(("train", ex))
    return examples


# ---------------------------------------------------------------- EquivSVA

MODULE_RE = re.compile(r"\bmodule\s+(\w+)")
SVA_RE = re.compile(r"(\w+)\s*:\s*assert\s+property\s*\((.*?)\)\s*;", re.S)


def checker(props, clock, reset_name, reset_active):
    """Verilog that checks the properties as immediate assertions, inserted before the design's endmodule.
    `disable iff (d) A |=> C` holds when C follows every cycle with A, unless d is high in either cycle.
    Reset is assumed in the first cycle, as a formal testbench would."""
    rst_on = reset_name if reset_active == "high" else f"!{reset_name}"
    out = ["  reg jev_started;",
           f"  always @(posedge {clock}) jev_started <= 1'b1;",
           f"  always @* if (!jev_started) assume({rst_on});"]
    for k, p in enumerate(props):
        ok = f"!({p['disable']})" if p.get("disable") else "1'b1"
        if p["type"] == "invariant":
            out.append(f"  always @(posedge {clock}) if (jev_started && {ok}) assert({p['expression']});")
        elif p["type"] == "next_cycle_implication" and p.get("delay", 1) == 1:
            out += [f"  reg jev_a{k};",
                    f"  always @(posedge {clock}) jev_a{k} <= {ok} && ({p['antecedent']});",
                    f"  always @(posedge {clock}) if (jev_started && jev_a{k} && {ok}) assert({p['consequent']});"]
        else:
            return None
    return "\n".join(out) + "\n"


def check_props(task):
    """task: (key, design source, top module, checker text, max steps, timeouts, yosys) -> verdict."""
    key, src, top, chk, cfg = task
    text = src[:src.rindex("endmodule")] + chk + src[src.rindex("endmodule"):]
    with tempfile.TemporaryDirectory(prefix="jevprop") as d:
        with open(os.path.join(d, "design.sv"), "w") as fh:
            fh.write(text)
        script = "\n".join([
            "read_verilog -sv -formal design.sv", f"hierarchy -check -top {top}", ELAB,
            f"tee -o sat.log sat -tempinduct -prove-asserts -set-assumes -set-init-zero -maxsteps {cfg['max_steps']} "
            f"-timeout {cfg['sat_timeout']} {top}", ""])
        log, rc, secs = run_yosys(cfg["yosys"], d, script, cfg["timeout"])
        sat = read_and_remove(d, "sat.log")
    if rc is None:
        verdict = "timeout"
    elif "model found for base case: FAIL" in sat:
        verdict = "violated"
    elif "Induction step proven: SUCCESS" in sat:
        verdict = "holds"
    elif "Reached maximum number of time steps" in sat:
        verdict = "step limit"
    elif rc != 0:
        verdict = "error"
    else:
        verdict = "unknown"
    return {"key": key, "verdict": verdict, "seconds": round(secs, 3), "error": first_error(log) if rc else None}


def property_example(eid, group, path, lines, block_lines, sva, violated, source):
    return {"id": eid, "group": group, "source": "equivsva", "kind": "property",
            "state": {"path": path, "code": "\n".join(lines), "blocks": make_blocks(len(lines), block_lines)},
            "questions": [{"id": "property_violated", "type": "choice", "options": ["yes", "no"],
                           "text": "A formal tool checks this property on the design:\n" + sva +
                                   "\nWill it find a violation?"}],
            "labels": {"property_violated": {"dist": {"yes" if violated else "no": 1.0}, "source": source,
                                             "weight": 1.0}}}


def equivsva(args, cfg, stats, log):
    rows = []
    for split in ("train", "dev", "test"):
        for line in open(os.path.join(RAW, "equivsva", f"{split}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            r["split"] = "val" if split == "test" else "train"
            rows.append(r)
    families = collections.defaultdict(list)
    for r in rows:
        families[r["family_id"]].append(r)

    examples, tasks, plan = [], [], {}
    for fam, variants in sorted(families.items())[:args.limit_families]:
        spec = json.loads(variants[0]["spec_json"])
        props = json.loads(variants[0]["properties_json"])
        svas = SVA_RE.findall(variants[0]["properties_sva"])
        if len(svas) != len(props):
            stats["equivsva: family skipped, properties don't match their SVA text"] += 1
            continue
        clean = {v["rtl_variant"]: strip_comments(v["rtl"].replace("\r\n", "\n").split("\n")) for v in variants}
        names = {v: MODULE_RE.search("\n".join(lines)).group(1) for v, lines in clean.items()}
        mutants = {}
        for mname, msrc in json.loads(variants[0]["mutants_json"]).items():
            mlines = strip_comments(msrc.replace("\r\n", "\n").split("\n"))
            base = max(clean, key=lambda v: difflib.SequenceMatcher(
                None, [ln for ln in clean[v] if not MODULE_RE.search(ln)],
                [ln for ln in mlines if not MODULE_RE.search(ln)], autojunk=False).ratio())
            mlines = [MODULE_RE.sub(f"module {names[base]}", ln, count=1) for ln in mlines]
            mutants[os.path.splitext(mname)[0]] = (base, mlines)
        bases = collections.Counter(b for b, _ in mutants.values())
        base = bases.most_common(1)[0][0] if bases else sorted(clean)[0]
        group, split = f"equivsva/{fam}", variants[0]["split"]

        # bug examples: the base design is "clean", the other equivalent designs "variant", mutants "buggy"
        examples.append((split, bug_example(f"{group}/clean", group, "equivsva", "clean", f"{names[base]}.sv",
                                            clean[base], args.block_lines)))
        for v, lines in sorted(clean.items()):
            if v != base:
                examples.append((split, bug_example(f"{group}/variant-{os.path.splitext(v)[0]}", group, "equivsva",
                                                    "variant", f"{names[v]}.sv", lines, args.block_lines)))
        for m, (b, mlines) in sorted(mutants.items()):
            regions = code_regions(mlines, clean[b])
            if not regions:
                stats["equivsva: mutant skipped, identical to its design"] += 1
                continue
            ex = bug_example(f"{group}/buggy-{m}", group, "equivsva", "buggy", f"{names[b]}.sv", mlines,
                             args.block_lines, regions)
            examples.append((split, ex))

        # property checks: one run per equivalent design (every property at once), one per mutant/property pair
        chk_all = checker(props, spec["clock"]["name"], spec["reset"]["name"], spec["reset"]["active"])
        if chk_all is None:
            stats["equivsva: family skipped, property form not supported"] += 1
            continue
        designs = {f"variant-{os.path.splitext(v)[0]}": (lines, names[v]) for v, lines in clean.items()}
        for v, (lines, top) in designs.items():
            tasks.append((f"{fam}|{v}|all", "\n".join(lines), top, chk_all, cfg))
        for m, (b, mlines) in mutants.items():
            for k, p in enumerate(props):
                chk = checker([p], spec["clock"]["name"], spec["reset"]["name"], spec["reset"]["active"])
                tasks.append((f"{fam}|mutant-{m}|{k}", "\n".join(mlines), names[b], chk, cfg))
        plan[fam] = (group, split, designs, mutants, svas, names)

    print(f"equivsva: {len(tasks)} Yosys property checks with {args.jobs} jobs", flush=True)
    results, t0 = {}, time.time()
    with Pool(args.jobs) as workers:
        for r in workers.imap_unordered(check_props, tasks, chunksize=4):
            results[r["key"]] = r
            log.write(json.dumps(r) + "\n")
            if len(results) % 500 == 0:
                print(f"   {len(results)}/{len(tasks)} checks, {(time.time() - t0) / 60:.1f} min", flush=True)
    stats["equivsva: property checks"] = dict(collections.Counter(r["verdict"] for r in results.values()))

    for fam, (group, split, designs, mutants, svas, names) in sorted(plan.items()):
        harness_ok = all(results[f"{fam}|{v}|all"]["verdict"] == "holds" for v in designs)
        if not harness_ok:   # our harness disagrees with EquivSVA's proofs: trust neither for this family
            stats["equivsva: family's properties dropped, harness didn't prove them on the equivalent designs"] += 1
            continue
        for v, (lines, top) in sorted(designs.items()):
            for k, (pname, body) in enumerate(svas):
                sva = f"{pname}: assert property ({body.strip()});"
                examples.append((split, property_example(f"{group}/{v}/{pname}", group, f"{top}.sv", lines,
                                                         args.block_lines, sva, False, "EquivSVA formal suite")))
        for m, (b, mlines) in sorted(mutants.items()):
            for k, (pname, body) in enumerate(svas):
                verdict = results[f"{fam}|mutant-{m}|{k}"]["verdict"]
                if verdict not in ("violated", "holds"):
                    stats["equivsva: mutant property dropped (no proof or counterexample)"] += 1
                    continue
                sva = f"{pname}: assert property ({body.strip()});"
                examples.append((split, property_example(f"{group}/mutant-{m}/{pname}", group, f"{names[b]}.sv",
                                                         mlines, args.block_lines, sva, verdict == "violated",
                                                         "yosys sat")))
    return examples


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "labeled_v1"))
    ap.add_argument("--sources", nargs="+", default=["rtl-augmented-v3", "equivsva", "hiersva"])
    ap.add_argument("--jobs", type=int, default=8, help="worker processes for the Yosys property checks")
    ap.add_argument("--yosys", default="yosys")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--sat-timeout", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-seq-tokens", type=int, default=8192)
    ap.add_argument("--block-lines", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.1, help="share of rtl-augmented-v3 repos in val")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "models", "Qwen3-8B"))
    ap.add_argument("--limit-families", type=int, default=None, help="only the first N EquivSVA families (testing)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from jev.format import Renderer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    renderer = Renderer(tok, args.max_seq_tokens)
    windower = Windower(tok, args.max_seq_tokens, args.block_lines)
    cfg = {"yosys": shlex.split(args.yosys), "max_steps": args.max_steps, "sat_timeout": args.sat_timeout,
           "timeout": args.timeout}
    stats = collections.Counter()
    os.makedirs(args.out, exist_ok=True)

    examples = []
    with open(os.path.join(args.out, "property_checks.jsonl"), "w") as log:
        if "rtl-augmented-v3" in args.sources:
            examples += rtl_augmented(windower, args, stats)
        if "hiersva" in args.sources:
            examples += hiersva(windower, args, stats)
        if "equivsva" in args.sources:
            examples += equivsva(args, cfg, stats, log)

    splits, lengths = collections.defaultdict(list), collections.defaultdict(list)
    for split, ex in examples:
        rendered = renderer.render(ex)
        if rendered is None:
            stats["dropped: rendered example too long"] += 1
            continue
        splits[split].append(ex)
        lengths[split].append(len(rendered["input_ids"]))
    summary = {"args": vars(args), "notes": {k: v for k, v in stats.items()}, "splits": {}}
    for name in ("train", "val"):
        exs = splits[name]
        with open(os.path.join(args.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for ex in exs:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        lens = sorted(lengths[name])
        by = collections.Counter((ex["source"], ex["kind"]) for ex in exs)
        violated = collections.Counter(
            next(iter(ex["labels"]["property_violated"]["dist"])) for ex in exs if ex["kind"] == "property")
        summary["splits"][name] = {
            "examples": len(exs), "by_source_and_kind": {f"{s} {k}": n for (s, k), n in sorted(by.items())},
            "property_answers": dict(violated), "tokens": sum(lens),
            "tokens_median": lens[len(lens) // 2] if lens else 0, "tokens_max": lens[-1] if lens else 0}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary["notes"], indent=2, default=str))
    for name, s in summary["splits"].items():
        print(f"{name}: {s['examples']} examples, {s['tokens']:,} tokens; property answers {s['property_answers']}")
        for k, n in s["by_source_and_kind"].items():
            print(f"   {n:6d}  {k}")
    print(f"done: {args.out} (train.jsonl, val.jsonl, property_checks.jsonl, stats.json)")


if __name__ == "__main__":
    main()
