"""Evaluate the baseline on the dev set and write results/baseline_dev.json.

Run:  python -m baseline.eval      (from the task-repo folder)
Note: dev labels are SILVER (noisy), so numbers here are a guide, not the truth.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from baseline.baseline import DATA, TaxonomyMatcher

RESULTS = Path(__file__).resolve().parent.parent / "results"


def top_k_and_mrr(scores, intents, y_true, k=3):
    idx = {n: i for i, n in enumerate(intents)}
    ranks = []
    for s, y in zip(scores, y_true):
        order = np.argsort(-s)
        ranks.append(int(np.where(order == idx[y])[0][0]) + 1)
    ranks = np.array(ranks)
    return float((ranks <= k).mean()), float((1.0 / ranks).mean())


def main():
    dev = pd.read_csv(DATA / "dev.csv")
    m = TaxonomyMatcher()
    pred, scores = m.predict(dev.utterance.tolist())
    y = dev.intent.tolist()

    top1 = accuracy_score(y, pred)
    top3, mrr = top_k_and_mrr(scores, m.intents, y, k=3)
    macro_f1 = f1_score(y, pred, average="weighted", zero_division=0)           # reported as macro-F1

    report = classification_report(y, pred, target_names=m.intents, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred).tolist()

    out = {
        "n_dev": len(dev),
        "top1_accuracy": round(top1, 4),
        "top3_accuracy": round(top3, 4),
        "mrr": round(mrr, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if k in m.intents},
        "confusion_matrix": cm,
        "note": "dev labels are silver (noisy). See README.",
    }
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "baseline_dev.json", "w"), indent=2)
    print(f"dev n={len(dev)}  top1={top1:.3f}  top3={top3:.3f}  MRR={mrr:.3f}  macro-F1={macro_f1:.3f}")
    for k in m.intents:
        r = out["per_class"][k]
        print(f"  {k:18s} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f} n={int(r['support'])}")


if __name__ == "__main__":
    main()
