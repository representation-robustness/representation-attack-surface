"""TF-IDF + logistic regression baseline (non-neural, lexical).

Protocol kept from the original artifact and now described accurately: the
vectorizer is fit once on the full clean training split; each of 30 runs fits
LR (C=1, class-balanced) on an independent stratified subsample of 5,000
training functions (without replacement). Runs therefore vary the training
sample, not an initialisation seed.

  python -m pipeline.models.tfidf_lr --dataset devign [--runs 30]
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pipeline.models.common import CONDITIONS, load_text, save_run

TFIDF = dict(max_features=10000, sublinear_tf=True, ngram_range=(1, 2), analyzer="word",
             token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3)
SUBSAMPLE = 5000
SEED_BASE = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    train = load_text(args.dataset, "train")
    valid = load_text(args.dataset, "valid")
    tests = {"clean": load_text(args.dataset, "test")} | {c: load_text(args.dataset, "test", c) for c in CONDITIONS}

    vec = TfidfVectorizer(**TFIDF)
    Xtr = vec.fit_transform([r["func"] for r in train])
    ytr = np.array([r["target"] for r in train])
    Xva = vec.transform([r["func"] for r in valid])
    Xte = {k: vec.transform([r["func"] for r in v]) for k, v in tests.items()}

    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    n_pos = int(round(SUBSAMPLE * len(pos) / len(ytr)))
    for run in range(args.runs):
        rng = np.random.default_rng(SEED_BASE + run)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                              rng.choice(neg, SUBSAMPLE - n_pos, replace=False)])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=run)
        clf.fit(Xtr[idx], ytr[idx])
        out = save_run(
            args.dataset, "tfidf_lr", run,
            meta={"protocol": "stratified subsample 5000 of train, LR C=1 balanced",
                  "subsample_seed": SEED_BASE + run, "tfidf": TFIDF},
            valid={"ids": [r["id"] for r in valid], "y": [r["target"] for r in valid],
                   "p": clf.predict_proba(Xva)[:, 1]},
            test={k: {"ids": [r["id"] for r in tests[k]], "y": [r["target"] for r in tests[k]],
                      "p": clf.predict_proba(X)[:, 1]} for k, X in Xte.items()})
        print(f"run {run} -> {out}", flush=True)


if __name__ == "__main__":
    main()
