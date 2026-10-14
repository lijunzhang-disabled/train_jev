"""Accuracy and calibration metrics over predictions, grouped by question id. Numpy only."""
import collections
import math

import numpy as np


def auroc(scores, positives):
    """Area under the ROC curve via the rank-sum statistic (ties get average ranks)."""
    scores, positives = np.asarray(scores, float), np.asarray(positives, bool)
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = scores.argsort()
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(confidence, correct, bins=15):
    """Expected calibration error of the top-class confidence."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence > lo) & (confidence <= hi)
        if sel.any():
            total += sel.mean() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(total)


def topk_credit(probs, gold, k):
    """Chance that at least one gold option is among the top k when ties are broken at random
    (an exactly uniform prediction scores chance, not whatever argmax/argsort happen to pick).
    gold: indices of the correct options; a bug fix can span several blocks, and any of them counts."""
    probs = np.asarray(probs)
    best = probs[gold].max()
    above = int((probs > best).sum())
    tied = int((probs == best).sum())
    tied_gold = int((probs[gold] == best).sum())
    slots = min(max(k - above, 0), tied)
    return 1 - math.comb(tied - tied_gold, slots) / math.comb(tied, slots)


def bug_score(p):
    """The probability that the answer means "there is a bug": P(yes), 1 - P(no bug) or 1 - P(none)."""
    options, probs = p["options"], p["probs"]
    if p["type"] == "pointer":
        return 1 - probs[-1]
    if options[:2] == ["yes", "no"]:
        return probs[0]
    if "no bug" in options:
        return 1 - probs[options.index("no bug")]
    return None


def pair_accuracy(ps):
    """The share of buggy/fixed pairs (ids ending in /buggy and /fixed) in which the buggy version gets the
    higher bug probability; ties count half. Pairs cancel what differs between files, so this is the most
    direct test of whether the model tells a bug from its fix. Returns (accuracy, pairs)."""
    by_file = collections.defaultdict(dict)
    for p in ps:
        key, _, kind = p.get("id", "").rpartition("/")
        if kind in ("buggy", "fixed"):
            by_file[key][kind] = bug_score(p)
    diffs = [d["buggy"] - d["fixed"] for d in by_file.values() if len(d) == 2 and None not in d.values()]
    if not diffs:
        return None, 0
    return float(np.mean([(d > 0) + 0.5 * (d == 0) for d in diffs])), len(diffs)


def macro_f1(pred, gold, classes):
    scores = []
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else None


def summarize(predictions):
    """predictions: [{"qid", "type", "options", "probs", "target"}] -> {qid: metrics}."""
    by_q = collections.defaultdict(list)
    for p in predictions:
        if p.get("target") is not None:
            by_q[p["qid"]].append(p)
    report = {}
    for qid, ps in sorted(by_q.items()):
        pred = [int(np.argmax(p["probs"])) for p in ps]
        gold = [int(np.argmax(p["target"])) for p in ps]
        golds = [np.flatnonzero(np.isclose(p["target"], max(p["target"]))) for p in ps]  # every top-weight option
        conf = [float(np.max(p["probs"])) for p in ps]
        correct = [topk_credit(p["probs"], g, 1) for p, g in zip(ps, golds)]
        m = {
            "n": len(ps),
            "accuracy": float(np.mean(correct)),
            "nll": float(np.mean([-np.dot(p["target"], np.log(np.clip(p["probs"], 1e-12, 1))) for p in ps])),
            "brier": float(np.mean([np.sum((np.asarray(p["probs"]) - np.asarray(p["target"])) ** 2) for p in ps])),
            "ece": ece(conf, correct),
        }
        if ps[0]["type"] == "choice":
            names = [p["options"][g] for p, g in zip(ps, gold)]
            preds = [p["options"][k] for p, k in zip(ps, pred)]
            m["macro_f1"] = macro_f1(preds, names, sorted(set(names)))
            m["label_counts"] = dict(collections.Counter(names))
            if ps[0]["options"][:2] == ["yes", "no"]:
                m["auroc_yes"] = auroc([p["probs"][0] for p in ps], [g == 0 for g in gold])
        else:  # pointer: last option is "none"
            has_bug = [len(p["options"]) - 1 not in g for p, g in zip(ps, golds)]
            top3 = [topk_credit(p["probs"], g, 3) for p, g in zip(ps, golds)]
            m["top3_accuracy"] = float(np.mean(top3))
            m["n_when_bug"] = sum(has_bug)
            m["accuracy_when_bug"] = float(np.mean([c for c, b in zip(correct, has_bug) if b])) if any(has_bug) else None
            m["top3_when_bug"] = float(np.mean([c for c, b in zip(top3, has_bug) if b])) if any(has_bug) else None
            m["accuracy_when_none"] = (float(np.mean([c for c, b in zip(correct, has_bug) if not b]))
                                       if not all(has_bug) else None)
            # where in a buggy file, ranking the blocks only: measures finding the bug apart from deciding there is one
            buggy = [(p["probs"][:-1], g) for p, g, b in zip(ps, golds, has_bug) if b]
            if buggy:
                for k, name in ((1, "accuracy"), (3, "top3")):
                    m[f"block_{name}_when_bug"] = float(np.mean([topk_credit(pr, g, k) for pr, g in buggy]))
                    m[f"block_{name}_chance_when_bug"] = float(np.mean([topk_credit(np.ones(len(pr)), g, k)
                                                                        for pr, g in buggy]))
            m["mean_blocks"] = float(np.mean([len(p["options"]) - 1 for p in ps]))
        m["pair_accuracy"], m["n_pairs"] = pair_accuracy(ps)
        report[qid] = m
    return report


def format_report(report):
    """The printed form of summarize()'s report: one line per question, then for a pointer the buggy/no-bug split
    and blocks-only ranking, and for buggy/fixed pairs how often the buggy version looks buggier."""
    lines = []
    for qid, m in report.items():
        extra = "".join(f"  {k} {m[k]:.3f}" for k in ("macro_f1", "auroc_yes", "top3_accuracy") if m.get(k) is not None)
        lines.append(f"   {qid:18} n={m['n']:5}  acc {m['accuracy']:.3f}{extra}  nll {m['nll']:.3f}  ece {m['ece']:.3f}")
        if m.get("n_when_bug"):  # a pointer: was the bug found, and was "none" said when there is none?
            split = f"{'':18} buggy n={m['n_when_bug']:5}  acc {m['accuracy_when_bug']:.3f}  top3 {m['top3_when_bug']:.3f}"
            if m["accuracy_when_none"] is not None:
                split += f"   no bug n={m['n'] - m['n_when_bug']}  acc {m['accuracy_when_none']:.3f}"
            lines.append("   " + split)
            lines.append(f"   {'':18} buggy, blocks only  acc {m['block_accuracy_when_bug']:.3f}  "
                         f"top3 {m['block_top3_when_bug']:.3f}   chance {m['block_accuracy_chance_when_bug']:.3f} / "
                         f"{m['block_top3_chance_when_bug']:.3f}")
        if m.get("n_pairs"):
            lines.append(f"   {'':18} pairs n={m['n_pairs']:5}  buggy version rated buggier {m['pair_accuracy']:.3f}"
                         f"   chance 0.500")
    return lines


if __name__ == "__main__":  # re-score saved predictions without a GPU: python -m jev.metrics <x.predictions.jsonl> ...
    import json
    import sys

    for path in sys.argv[1:]:
        with open(path) as fh:
            print(f"== {path}")
            print("\n".join(format_report(summarize([json.loads(line) for line in fh]))))

