"""Vul-LMGGNN (Liu et al.): gated GNN over the code property graph jointly trained with CodeBERT.

Ported from upstream Vul-LMGNN/vul-LMGGNN (pinned) with one documented deviation:
upstream's graph readout ends in softmax over a size-1 dimension, so the graph branch
always outputs 1.0 and the released model reduces to CodeBERT. Here the readout emits two
logits and the graph and CodeBERT branches are mixed as probabilities, as the paper
describes: p = k * softmax(graph) + (1 - k) * softmax(codebert), k = 0.1, trained with NLL.
Kept from upstream: node features [type id, 100-d Word2Vec of node code] (101-d), 6-step
GatedGraphConv (200 ch), conv readout over a fixed 205-node window, CodeBERT max_len 512.
Optimisation follows Table I: batch 8, lr 2e-5, 10 epochs, AdamW wd 1.3e-6.

  python -m pipeline.models.lmggnn --dataset devign --seed 42 [--smoke]

Portions derived from vul-LMGGNN (https://github.com/Vul-LMGNN/vul-LMGGNN, commit 7006eb6),
Apache License 2.0; modified as described above. See THIRD_PARTY_NOTICES.md.
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_dense_batch
from transformers import RobertaModel, RobertaTokenizer

from pipeline.models.common import CONDITIONS, HF_REVISIONS, best_threshold, load_text, run_name, save_run, set_seed
CB = "microsoft/codebert-base"
from pipeline.models.graph_views import EDGE_ORDER, load_view

VIEW = {"level": "ast", "edges": EDGE_ORDER, "features": "type+w2v"}
NODES, K, BATCH, LR, EPOCHS, WD, MAXLEN = 205, 0.1, 8, 2e-5, 10, 1.3e-6, 512


class ConvReadout(nn.Module):
    """Devign-style readout (upstream layers.Conv) over a fixed NODES x feature window."""

    def __init__(self, hid, inp):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(NODES, 50, 3, padding=1), nn.Conv1d(50, 20, 1, padding=1)
        self.mp1, self.mp2 = nn.MaxPool1d(3, 2), nn.MaxPool1d(2, 2)

        def out(n):
            n = (n - 3) // 2 + 1 + 2  # conv2 padding=1 adds 2
            return 20 * ((n - 2) // 2 + 1)
        self.fz, self.fy = nn.Linear(out(hid + inp), 2), nn.Linear(out(hid), 2)
        self.drop = nn.Dropout(0.2)

    def forward(self, h, x):
        z = self.mp2(self.c2(self.mp1(F.relu(self.c1(torch.cat([h, x], -1))))))
        y = self.mp2(self.c2(self.mp1(F.relu(self.c1(h)))))
        return self.drop(self.fz(z.flatten(1)) * self.fy(y.flatten(1)))


class LMGGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ggnn = GatedGraphConv(200, num_layers=6, aggr="add")
        self.readout = ConvReadout(200, 101)
        self.bert = RobertaModel.from_pretrained(CB, revision=HF_REVISIONS[CB])
        self.cls = nn.Linear(768, 2)

    def forward(self, b, ids, mask):
        x = torch.cat([b.x_type.float().unsqueeze(1), b.x_w2v.float()], 1)
        h = self.ggnn(F.pad(x, (0, 200 - x.shape[1])), b.edge_index)
        hd, _ = to_dense_batch(h, b.batch, max_num_nodes=NODES)
        xd, _ = to_dense_batch(x, b.batch, max_num_nodes=NODES)
        if hd.shape[1] < NODES:
            hd, xd = F.pad(hd, (0, 0, 0, NODES - hd.shape[1])), F.pad(xd, (0, 0, 0, NODES - xd.shape[1]))
        g = F.softmax(self.readout(hd, xd), -1)
        t = F.softmax(self.cls(self.bert(ids, mask)[0][:, 0]), -1)
        return K * g + (1 - K) * t  # class probabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    dev = torch.device("cpu" if args.smoke else "cuda")
    tok = RobertaTokenizer.from_pretrained(CB, revision=HF_REVISIONS[CB])

    text = {}
    for sp, conds in (("train", ["clean"]), ("valid", ["clean"]), ("test", ["clean"] + CONDITIONS)):
        for c in conds:
            text[(sp, c)] = {r["id"]: r["func"] for r in load_text(args.dataset, sp, c)}
    sets = {("train", "clean"): load_view(args.dataset, "train", "clean", VIEW),
            ("valid", "clean"): load_view(args.dataset, "valid", "clean", VIEW)}
    sets |= {("test", c): load_view(args.dataset, "test", c, VIEW) for c in ["clean"] + CONDITIONS}
    epochs = EPOCHS
    if args.smoke:
        sets = {k: v[:16] for k, v in sets.items()}
        epochs = 1

    def batches(key, shuffle=False, sampler=None):
        for b in DataLoader(sets[key], batch_size=BATCH, shuffle=shuffle, sampler=sampler):
            enc = tok([text[key][s] for s in b.sid], max_length=MAXLEN, truncation=True, padding="max_length",
                      return_tensors="pt")
            yield b.to(dev), enc.input_ids.to(dev), enc.attention_mask.to(dev)

    @torch.no_grad()
    def predict(key):
        model.eval()
        return np.concatenate([model(b, i, m)[:, 1].cpu().numpy() for b, i, m in batches(key)])

    model = LMGGNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ytr = np.array([int(d.y) for d in sets[("train", "clean")]])
    sampler = None
    if args.dataset == "bigvul":
        w = np.where(ytr == 1, 0.5 / ytr.sum(), 0.5 / (len(ytr) - ytr.sum()))
        sampler = WeightedRandomSampler(w, len(w), True, generator=torch.Generator().manual_seed(args.seed))
    yva = np.array([int(d.y) for d in sets[("valid", "clean")]])
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b, i, m in batches(("train", "clean"), shuffle=sampler is None, sampler=sampler):
            loss = F.nll_loss(torch.log(model(b, i, m) + 1e-10), b.y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        pva = predict(("valid", "clean"))
        f1 = 100 * f1_score(yva, pva >= best_threshold(yva, pva))
        print(f"epoch {ep + 1} loss={tot / max(1, n):.4f} valid_f1={f1:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)

    pva = predict(("valid", "clean"))
    test = {c: {"ids": [d.sid for d in sets[("test", c)]], "y": [int(d.y) for d in sets[("test", c)]],
                "p": predict(("test", c))} for c in ["clean"] + CONDITIONS}
    if args.smoke:
        print("smoke ok:", {k: round(float(np.mean(v["p"])), 4) for k, v in test.items()})
        return
    print("saved", save_run(args.dataset, run_name("lmggnn", []), args.seed,
                            meta={"view": VIEW, "k": K, "batch": BATCH, "lr": LR, "epochs": EPOCHS,
                                  "readout_nodes": NODES, "best_valid_f1": best,
                                  "deviation": "graph readout emits 2 logits (upstream softmax over size-1 dim is constant)"},
                            valid={"ids": [d.sid for d in sets[("valid", "clean")]], "y": yva.tolist(), "p": pva},
                            test=test))


if __name__ == "__main__":
    main()
