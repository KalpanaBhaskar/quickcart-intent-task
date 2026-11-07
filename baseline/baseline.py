"""Naive baseline: match an utterance against the hard-coded intent taxonomy.

Each intent is represented by the TF-IDF centroid of its taxonomy examples (data/taxonomy.json).
An utterance is scored by cosine similarity to every intent centroid; the top intent wins.
Anything whose best score is below THRESHOLD is sent to out_of_scope.

This is deliberately simple. Swapping TF-IDF for a sentence-embedding model is one obvious next step,
but the way this is built has other problems worth finding.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA = Path(__file__).resolve().parent.parent / "data"
THRESHOLD = 0.10          # below this best-match score we call it out_of_scope


def load_taxonomy():
    tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    return [i["name"] for i in tax["intents"]], {i["name"]: i["examples"] for i in tax["intents"]}


def clean(text):
    return " ".join(str(text).lower().split())


class TaxonomyMatcher:
    def __init__(self):
        self.intents, self.examples = load_taxonomy()
        train = pd.read_csv(DATA / "train.csv")
        dev = pd.read_csv(DATA / "dev.csv")
        corpus = [clean(t) for t in pd.concat([train.utterance, dev.utterance])]   # vocabulary / idf source
        corpus += [clean(e) for ex in self.examples.values() for e in ex]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus)
        self.centroids = self._build_centroids()

    def _build_centroids(self):
        rows = []
        for name in self.intents:
            ex = self.examples[name]
            if not ex:
                rows.append(np.zeros(len(self.vec.vocabulary_)))      # no examples -> never matches
                continue
            m = np.asarray(self.vec.transform([clean(e) for e in ex]).mean(axis=0)).ravel()
            rows.append(m)
        return normalize(np.vstack(rows))

    def scores(self, utterances):
        X = normalize(self.vec.transform([clean(u) for u in utterances]))
        return np.asarray(X @ self.centroids.T)                        # (n, n_intents) cosine

    def predict(self, utterances):
        s = self.scores(utterances)
        pred = [self.intents[i] for i in s.argmax(1)]
        best = s.max(1)
        return [("out_of_scope" if b < THRESHOLD else p) for p, b in zip(pred, best)], s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write predictions for data/test.csv in the submission format.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "submissions" / "baseline_predictions.csv"))
    a = ap.parse_args()
    test = pd.read_csv(DATA / "test.csv")
    pred, _ = TaxonomyMatcher().predict(test.utterance.tolist())
    pd.DataFrame({"id": test["id"], "intent": pred}).to_csv(a.out, index=False)
    print("wrote", a.out, len(pred))
