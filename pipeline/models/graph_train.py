"""Train one graph detector for one seed and save per-sample predictions.

  python -m pipeline.models.graph_train --model reveal --dataset devign --seed 42
         [--train-conditions identifier,deadcode,controlflow]   # robustness training -> "<model>_aug"
         [--smoke]                                               # CPU, tiny subset, nothing saved

Training: Adam, batch/lr/epochs/patience from configs/graph_models.yaml; class-balanced
sampling on datasets listed in configs/models.yaml:balanced_sampler_datasets; checkpoint =
best validation F1. REVEAL additionally runs its phase 2 (SMOTE + triplet representation
learning on phase-1 graph embeddings), as in Chakraborty et al.
"""
import argparse
import copy
import fcntl
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from pipeline.models.common import run_name, CONDITIONS, HF_REVISIONS, REPO, ROOT, best_threshold, save_run, set_seed
CB = "microsoft/codebert-base"
from pipeline.models.graph_models import ARCHS
from pipeline.models.graph_views import EDGE_ORDER, Vocab, load_view, node_features

GCFG = yaml.safe_load(open(REPO / "configs/graph_models.yaml"))
MCFG = yaml.safe_load(open(REPO / "configs/models.yaml"))


def model_cfg(name: str) -> dict:
    c = dict(GCFG[name])
    if "inherit" in c:
        c = {**model_cfg(c.pop("inherit")), **c}
    c["view"]["edges"] = [e for e in EDGE_ORDER if e in c["view"]["edges"]]
    return c


@torch.no_grad()
def predict(model, data, n_types, feats, dev, bs=256, emb=False):
    model.eval()
    out = []
    for b in DataLoader(data, batch_size=bs):
        b = b.to(dev)
        x = node_features(b, n_types, feats)
        out.append((model.embed(b, x) if emb else torch.sigmoid(model(b, x))).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def attach_cls(dataset, sets, dev):
    """Frozen pretrained CodeBERT [CLS] of each function (cached), attached as d.cb for ablation arm 2."""
    from transformers import AutoModel, AutoTokenizer
    from pipeline.models.common import ROOT, load_text
    tok = enc = None
    for (split, cond), data in sets.items():
        path = ROOT / "cache" / "cls" / dataset / f"{split}_{cond}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lk = open(path.with_suffix(".lock"), "w")  # concurrent jobs: one encodes, the others wait and reuse it
        fcntl.flock(lk, fcntl.LOCK_EX)
        if path.exists():
            cache = torch.load(path)
        else:
            if enc is None:
                tok = AutoTokenizer.from_pretrained(CB, revision=HF_REVISIONS[CB])
                enc = AutoModel.from_pretrained(CB, revision=HF_REVISIONS[CB]).to(dev).eval()
            recs = load_text(dataset, split, cond)
            cache = {}
            for i in range(0, len(recs), 64):
                chunk = recs[i:i + 64]
                t = tok([r["func"] for r in chunk], max_length=512, truncation=True, padding=True, return_tensors="pt")
                h = enc(input_ids=t.input_ids.to(dev), attention_mask=t.attention_mask.to(dev)).last_hidden_state[:, 0]
                cache.update({r["id"]: v.cpu().half() for r, v in zip(chunk, h)})
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            torch.save(cache, tmp)
            os.replace(tmp, path)
        lk.close()
        for d in data:
            d.cb = cache[d.sid].float().unsqueeze(0)


def focal(logits, y, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return ((1 - torch.exp(-bce)) ** gamma * bce).mean()


# MetricLearningModel and the reveal_phase2 loss are ported from ReVeal
# (https://github.com/VulDetProject/ReVeal, commit ca31b78, Vuld_SySe/representation_learning/models.py),
# MIT License, Copyright (c) 2021 VulDetProject. See THIRD_PARTY_NOTICES.md.
class MetricLearningModel(nn.Module):
    """REVEAL phase-2 network, as upstream ReVeal Vuld_SySe/representation_learning/models.py (num_layers=1)."""

    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.feature = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(hidden, 2), nn.LogSoftmax(dim=-1))

    def extract(self, x):
        return self.feature(self.layer1(x))


def reveal_phase2(ztr, ytr, zva, yva, seed, dev, epochs=100, patience=5, batch_size=128,
                  margin=0.5, lambda1=0.5, lambda2=1e-3):
    """SMOTE-balanced phase-1 embeddings -> MetricLearningModel trained, per sample, with
    NLL + lambda1*|D+ - D- + margin| (cosine distances to a random same-/other-class training example)
    + lambda2*(|h_a| + |h_p| + |h_n|), summed over the batch; Adam defaults. These are the upstream
    defaults used for Devign (api_test.py: batch 128, max_patience 5). Model selection uses our
    validation split and validation-tuned threshold, as for every other detector."""
    from imblearn.over_sampling import SMOTE
    Xs, ys = SMOTE(random_state=seed).fit_resample(ztr, ytr)
    X, Y = torch.tensor(Xs, dtype=torch.float32, device=dev), torch.tensor(ys, dtype=torch.long, device=dev)
    Xv = torch.tensor(zva, dtype=torch.float32, device=dev)
    pos_idx, neg_idx = torch.where(Y == 1)[0], torch.where(Y == 0)[0]
    net = MetricLearningModel(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters())
    g = torch.Generator(device="cpu").manual_seed(seed)

    def draw(pool, n):
        return pool[torch.randint(len(pool), (n,), generator=g).to(dev)]

    def score_of(x):
        net.eval()
        with torch.no_grad():
            return torch.cat([net.classifier(net.extract(x[i:i + 4096])).exp()[:, 1]
                              for i in range(0, len(x), 4096)]).cpu().numpy()

    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            y = Y[idx]
            same = torch.where(y == 1, draw(pos_idx, len(y)), draw(neg_idx, len(y)))
            diff = torch.where(y == 1, draw(neg_idx, len(y)), draw(pos_idx, len(y)))
            ha, hp, hn = net.extract(X[idx]), net.extract(X[same]), net.extract(X[diff])
            d_pos = 1 - F.cosine_similarity(ha, hp, dim=-1, eps=1e-8)
            d_neg = 1 - F.cosine_similarity(ha, hn, dim=-1, eps=1e-8)
            loss = (F.nll_loss(net.classifier(ha), y, reduction="none")
                    + lambda1 * (d_pos - d_neg + margin).abs()
                    + lambda2 * (ha.norm(dim=-1) + hp.norm(dim=-1) + hn.norm(dim=-1))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        pv = score_of(Xv)
        f1 = f1_score(yva, pv >= best_threshold(yva, pv))
        print(f"phase2 epoch {ep + 1} loss={tot / len(X):.4f} valid_f1={100 * f1:.2f} valid_p_sd={pv.std():.4f}", flush=True)
        if f1 > best:
            best, state, bad = f1, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return (lambda z: score_of(torch.tensor(z, dtype=torch.float32, device=dev))), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-conditions", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    c = model_cfg(args.model)
    view, feats = c["view"], c["view"]["features"]
    set_seed(args.seed)
    dev = torch.device("cpu" if args.smoke else "cuda")
    vocab = Vocab(args.dataset)

    aug = [x for x in args.train_conditions.split(",") if x]
    train = load_view(args.dataset, "train", "clean", view)
    for cond in aug:
        train = train + load_view(args.dataset, "train", cond, view)
    valid = load_view(args.dataset, "valid", "clean", view)
    tests = {"clean": load_view(args.dataset, "test", "clean", view)} | \
            {k: load_view(args.dataset, "test", k, view) for k in CONDITIONS}
    if args.smoke:
        train, valid = train[:256], valid[:64]
        tests = {k: v[:32] for k, v in tests.items()}
        c.update(epochs=1)
    if c.get("codebert_cls"):
        attach_cls(args.dataset, {("train", "clean"): train, ("valid", "clean"): valid,
                                  **{("test", k): v for k, v in tests.items()}}, dev)

    in_dim = vocab.n_types + (100 if feats == "type+w2v" else 0)
    model = ARCHS[c["arch"]](in_dim=in_dim, n_rel=len(view["edges"]), hidden=c.get("hidden", 128),
                             layers=c.get("layers", 3), cb_dim=768 if c.get("codebert_cls") else 0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    ytr = np.array([int(d.y) for d in train])
    gen = torch.Generator().manual_seed(args.seed)
    balanced = args.dataset in MCFG["balanced_sampler_datasets"] or c["arch"] == "reveal"
    if balanced:
        w = np.where(ytr == 1, 0.5 / max(1, ytr.sum()), 0.5 / max(1, len(ytr) - ytr.sum()))
        loader = DataLoader(train, batch_size=c["batch_size"],
                            sampler=WeightedRandomSampler(w, len(w), replacement=True, generator=gen))
    else:
        loader = DataLoader(train, batch_size=c["batch_size"], shuffle=True, generator=gen)
    lossf = focal if c["arch"] == "reveal" else F.binary_cross_entropy_with_logits

    yva = np.array([int(d.y) for d in valid])
    best, state, bad = -1.0, None, 0
    for ep in range(c["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for b in loader:
            b = b.to(dev)
            loss = lossf(model(b, node_features(b, vocab.n_types, feats)), b.y.float())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
        pva = predict(model, valid, vocab.n_types, feats, dev)
        f1 = 100 * f1_score(yva, pva >= best_threshold(yva, pva))
        print(f"epoch {ep + 1} loss={tot / max(1, len(loader)):.4f} valid_f1={f1:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= c["patience"]:
                break
    model.load_state_dict(state)

    if c["arch"] == "reveal":
        emb = lambda data: predict(model, data, vocab.n_types, feats, dev, emb=True)  # noqa: E731
        ztr, zva = emb(train), emb(valid)
        if not args.smoke:  # phase-1 embeddings kept so phase 2 can be inspected without retraining
            ep = ROOT / "cache" / "reveal_emb" / args.dataset / f"{run_name(args.model, aug)}_s{args.seed}.npz"
            ep.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ep, ztr=ztr, ytr=ytr, zva=zva, yva=yva)
        score, best2 = reveal_phase2(ztr, ytr, zva, yva, args.seed, dev)
        prob = lambda data: score(emb(data))  # noqa: E731
        meta_extra = {"phase1_valid_f1": best, "phase2_valid_f1": 100 * best2}
    else:
        prob = lambda data: predict(model, data, vocab.n_types, feats, dev)  # noqa: E731
        meta_extra = {"best_valid_f1": best}

    pva = prob(valid)
    test = {k: {"ids": [d.sid for d in v], "y": [int(d.y) for d in v], "p": prob(v)} for k, v in tests.items()}
    if args.smoke:
        print("smoke ok:", {k: round(float(np.mean(v["p"])), 4) for k, v in test.items()})
        return
    name = run_name(args.model, aug)
    out = save_run(args.dataset, name, args.seed, meta={"config": c, "train_conditions": aug, "n_train": len(train),
                                                         "n_types": vocab.n_types, **meta_extra},
                   valid={"ids": [d.sid for d in valid], "y": yva.tolist(), "p": pva}, test=test)
    print("saved", out)


if __name__ == "__main__":
    main()
