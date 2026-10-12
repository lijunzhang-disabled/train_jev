"""Accuracy and calibration metrics over predictions, grouped by question id. Numpy only."""
import collections

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
        conf = [float(np.max(p["probs"])) for p in ps]
        correct = [a == b for a, b in zip(pred, gold)]
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
            has_bug = [g != len(p["options"]) - 1 for p, g in zip(ps, gold)]
            top3 = [g in np.argsort(p["probs"])[-3:] for p, g in zip(ps, gold)]
            m["top3_accuracy"] = float(np.mean(top3))
            m["accuracy_when_bug"] = float(np.mean([c for c, b in zip(correct, has_bug) if b])) if any(has_bug) else None
            m["accuracy_when_none"] = (float(np.mean([c for c, b in zip(correct, has_bug) if not b]))
                                       if not all(has_bug) else None)
            m["mean_blocks"] = float(np.mean([len(p["options"]) - 1 for p in ps]))
        report[qid] = m
    return report
