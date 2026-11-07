"""ReGVD (Nguyen et al. 2022) via the authors' own model code (upstream GNN-ReGVD, pinned).

Settings from the upstream README: GraphCodeBERT word embeddings (frozen copy), unique-token
graph ("uni"), window 5, ReGCN, 2 layers, hidden 128, lr 5e-4, batch 128, block 400,
grad-norm 1.0. Differences, applied to every detector in this study: checkpoint chosen by
validation F1 (upstream: accuracy) with early-stopping patience 10 (Table I).

  python -m pipeline.models.regvd --dataset devign --seed 42 [--train-conditions ...] [--smoke]
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from pipeline.models.common import run_name, CONDITIONS, HF_REVISIONS, REPO, ROOT, best_threshold, load_text, save_run, set_seed

UPSTREAM = Path(os.environ.get("RERUN_UPSTREAM", ROOT / "upstream")) / "GNN-ReGVD/code"
PRETRAINED = "microsoft/graphcodebert-base"
ARGS = SimpleNamespace(gnn="ReGCN", feature_dim_size=768, hidden_size=128, num_GNN_layers=2,
                       remove_residual=False, att_op="mul", format="uni", window_size=5, num_classes=2)
BLOCK, BATCH, LR, EPOCHS, PATIENCE = 400, 128, 5e-4, 100, 10


def encode(tok, recs):
    ids = []
    for r in recs:
        toks = tok.tokenize(" ".join(r["func"].split()))[:BLOCK - 2]
        x = tok.convert_tokens_to_ids([tok.cls_token] + toks + [tok.sep_token])
        ids.append(x + [tok.pad_token_id] * (BLOCK - len(x)))
    return TensorDataset(torch.tensor(ids), torch.tensor([r["target"] for r in recs]))


@torch.no_grad()
def predict(model, ds):
    model.eval()
    return torch.cat([model(input_ids=x)[:, 0].float().cpu() for x, _ in DataLoader(ds, batch_size=BATCH)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-conditions", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    sys.path.insert(0, str(UPSTREAM))
    # upstream utils.py imports a scipy path removed in scipy>=1.8; alias it instead of editing upstream
    import types
    import scipy.sparse.linalg
    shim = types.ModuleType("scipy.sparse.linalg.eigen.arpack")
    shim.eigsh = scipy.sparse.linalg.eigsh
    sys.modules.setdefault("scipy.sparse.linalg.eigen", types.ModuleType("scipy.sparse.linalg.eigen"))
    sys.modules["scipy.sparse.linalg.eigen.arpack"] = shim
    import modelGNN_updates  # noqa: E402  (upstream module; reads a module-level `device`)
    import model as upstream_model  # noqa: E402
    dev = torch.device("cpu" if args.smoke else "cuda")
    modelGNN_updates.device = upstream_model.device = dev

    tok = RobertaTokenizer.from_pretrained(PRETRAINED, revision=HF_REVISIONS[PRETRAINED])
    config = RobertaConfig.from_pretrained(PRETRAINED, revision=HF_REVISIONS[PRETRAINED], num_labels=1)
    encoder = RobertaForSequenceClassification.from_pretrained(PRETRAINED, revision=HF_REVISIONS[PRETRAINED], config=config)
    model = upstream_model.GNNReGVD(encoder, config, tok, ARGS).to(dev)  # upstream sets per-layer dtypes

    aug = [x for x in args.train_conditions.split(",") if x]
    train = load_text(args.dataset, "train") + [r for c in aug for r in load_text(args.dataset, "train", c)]
    valid = load_text(args.dataset, "valid")
    tests = {"clean": load_text(args.dataset, "test")} | {k: load_text(args.dataset, "test", k) for k in CONDITIONS}
    epochs = EPOCHS
    if args.smoke:
        train, valid, tests, epochs = train[:64], valid[:32], {k: v[:16] for k, v in tests.items()}, 1

    tr = encode(tok, train)
    gen = torch.Generator().manual_seed(args.seed)
    if args.dataset in ("bigvul",):
        y = np.array([r["target"] for r in train])
        w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
        loader = DataLoader(tr, batch_size=BATCH, sampler=WeightedRandomSampler(w, len(w), True, generator=gen))
    else:
        loader = DataLoader(tr, batch_size=BATCH, shuffle=True, generator=gen)
    params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW(params, lr=LR, eps=1e-8, weight_decay=0.0)

    va = encode(tok, valid)
    yva = np.array([r["target"] for r in valid])
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in loader:
            loss, _ = model(input_ids=x, labels=y.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item()
        pva = predict(model, va)
        f1 = 100 * f1_score(yva, pva >= best_threshold(yva, pva))
        print(f"epoch {ep + 1} loss={tot / len(loader):.4f} valid_f1={f1:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, bad = f1, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state, strict=False)

    pva = predict(model, va)
    test = {k: {"ids": [r["id"] for r in v], "y": [r["target"] for r in v], "p": predict(model, encode(tok, v))}
            for k, v in tests.items()}
    if args.smoke:
        print("smoke ok:", {k: round(float(np.mean(v["p"])), 4) for k, v in test.items()})
        return
    name = run_name("regvd", aug)
    print("saved", save_run(args.dataset, name, args.seed,
                            meta={"upstream_args": vars(ARGS), "block": BLOCK, "batch": BATCH, "lr": LR,
                                  "patience": PATIENCE, "pretrained": PRETRAINED, "train_conditions": aug,
                                  "best_valid_f1": best},
                            valid={"ids": [r["id"] for r in valid], "y": yva.tolist(), "p": pva}, test=test))


if __name__ == "__main__":
    main()
