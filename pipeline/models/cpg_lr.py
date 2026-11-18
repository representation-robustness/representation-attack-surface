"""CPG + logistic regression (non-neural, structural).

One definition on every dataset: graph-level features from the full Joern graph
(AST view) = normalised node-type histogram, normalised edge-type histogram, and
log node/edge counts. Identifier text never enters, so the representation is
renaming-invariant by construction. (The original artifact used post-GGNN mean
embeddings on Devign but type histograms elsewhere; this replaces both.)
Protocol identical to TF-IDF+LR: 30 runs, each on a stratified subsample of 5,000
training graphs, LR C=1 class-balanced, standardised features.

  python -m pipeline.models.cpg_lr --dataset devign
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.models.common import CONDITIONS, save_run
from pipeline.models.graph_views import EDGE_ORDER, Vocab, load_view

VIEW = {"level": "ast", "edges": EDGE_ORDER, "features": "type"}
SUBSAMPLE, SEED_BASE = 5000, 42


def featurize(graphs, n_types):
    X = np.zeros((len(graphs), n_types + len(EDGE_ORDER) + 2), dtype=np.float32)
    for i, g in enumerate(graphs):
        nt = np.bincount(g.x_type.numpy(), minlength=n_types)
        et = np.bincount(g.edge_type.numpy(), minlength=len(EDGE_ORDER))
        X[i, :n_types] = nt / max(1, nt.sum())
        X[i, n_types:n_types + len(EDGE_ORDER)] = et / max(1, et.sum())
        X[i, -2:] = np.log1p([g.num_nodes, g.edge_index.shape[1]])
    return X, [g.sid for g in graphs], [int(g.y) for g in graphs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    nt = Vocab(args.dataset).n_types
    Xtr, _, ytr = featurize(load_view(args.dataset, "train", "clean", VIEW), nt)
    Xva, idva, yva = featurize(load_view(args.dataset, "valid", "clean", VIEW), nt)
    tests = {k: featurize(load_view(args.dataset, "test", k, VIEW), nt) for k in ["clean"] + CONDITIONS}
    ytr = np.array(ytr)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    n_pos = int(round(SUBSAMPLE * len(pos) / len(ytr)))
    for run in range(args.runs):
        rng = np.random.default_rng(SEED_BASE + run)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, SUBSAMPLE - n_pos, replace=False)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                                                                 random_state=run))
        clf.fit(Xtr[idx], ytr[idx])
        save_run(args.dataset, "cpg_lr", run,
                 meta={"protocol": "stratified subsample 5000 of train, LR C=1 balanced", "view": VIEW,
                       "subsample_seed": SEED_BASE + run},
                 valid={"ids": idva, "y": yva, "p": clf.predict_proba(Xva)[:, 1]},
                 test={k: {"ids": ids, "y": y, "p": clf.predict_proba(X)[:, 1]} for k, (X, ids, y) in tests.items()})
        print("run", run, flush=True)


if __name__ == "__main__":
    main()
