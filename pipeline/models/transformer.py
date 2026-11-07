"""Fine-tune a sequence detector (CodeBERT / CodeT5+) for one seed and save predictions.

  python -m pipeline.models.transformer --model codebert --dataset devign --seed 42 [--train-conditions identifier,deadcode,controlflow]

Checkpoint selection: epoch with best validation F1 (threshold chosen on validation).
--train-conditions adds transformed copies of the training split (robustness training);
the run is then saved under "<model>_aug".
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from pipeline.models.common import run_name, CONDITIONS, HF_REVISIONS, REPO, ROOT, best_threshold, load_text, save_run, set_seed

CFG = yaml.safe_load(open(REPO / "configs/models.yaml"))


class Classifier(nn.Module):
    def __init__(self, name: str, pooling: str):
        super().__init__()
        enc = AutoModel.from_pretrained(name, revision=HF_REVISIONS[name])
        self.enc = enc.encoder if hasattr(enc, "encoder") and "t5" in name.lower() else enc
        self.pooling = pooling
        hid = self.enc.config.hidden_size if hasattr(self.enc, "config") else enc.config.d_model
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hid, 2))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            z = h[:, 0]
        else:
            m = mask.unsqueeze(-1).float()
            z = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(z)


def encode(tok, recs, max_len):
    enc = tok([r["func"] for r in recs], max_length=max_len, truncation=True, padding="max_length",
              return_tensors="pt")
    y = torch.tensor([r["target"] for r in recs])
    return torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"], y)


@torch.no_grad()
def predict(model, ds, bs, dev):  # noqa: D103
    model.eval()
    ps = []
    for ids, mask, _ in DataLoader(ds, batch_size=bs):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(ids.to(dev), mask.to(dev))
        ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu())
    return torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["codebert", "codet5plus"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-conditions", default="")
    ap.add_argument("--smoke", action="store_true", help="CPU, 64 samples, 1 epoch, results not saved")
    args = ap.parse_args()
    c = dict(CFG[args.model])
    set_seed(args.seed)
    dev = torch.device("cpu" if args.smoke else "cuda")
    tok = AutoTokenizer.from_pretrained(c["pretrained"], revision=HF_REVISIONS[c["pretrained"]])

    train = load_text(args.dataset, "train")
    aug = [x for x in args.train_conditions.split(",") if x]
    for cond in aug:
        train += load_text(args.dataset, "train", cond)
    valid = load_text(args.dataset, "valid")
    tests = {"clean": load_text(args.dataset, "test")} | {k: load_text(args.dataset, "test", k) for k in CONDITIONS}
    if args.smoke:
        train, valid = train[:64], valid[:32]
        tests = {k: v[:16] for k, v in tests.items()}
        c.update(epochs=1, batch_size=8, max_len=128)

    tr_ds = encode(tok, train, c["max_len"])
    va_ds = encode(tok, valid, c["max_len"])
    if args.dataset in CFG["balanced_sampler_datasets"]:
        y = np.array([r["target"] for r in train])
        w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
        g = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(tr_ds, batch_size=c["batch_size"], generator=g,
                            sampler=WeightedRandomSampler(w, len(w), replacement=True, generator=g))
    else:
        loader = DataLoader(tr_ds, batch_size=c["batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed))

    model = Classifier(c["pretrained"], c["pooling"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    total = c["epochs"] * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(c["warmup_ratio"] * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.CrossEntropyLoss()

    name = run_name(args.model, aug)
    ckpt = ROOT / "runs" / "ckpt" / args.dataset / ("smoke_" + name if args.smoke else name) / f"seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best, yva = -1.0, np.array([r["target"] for r in valid])
    for ep in range(c["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for ids, mask, y in loader:
            opt.zero_grad()
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = lossf(model(ids.to(dev), mask.to(dev)), y.to(dev))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        pva = predict(model, va_ds, 2 * c["batch_size"], dev)
        f1 = 100 * f1_score(yva, pva >= best_threshold(yva, pva))
        print(f"epoch {ep + 1}/{c['epochs']} loss={tot / len(loader):.4f} valid_f1={f1:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    pva = predict(model, va_ds, 2 * c["batch_size"], dev)
    test = {}
    for k, recs in tests.items():
        test[k] = {"ids": [r["id"] for r in recs], "y": [r["target"] for r in recs],
                   "p": predict(model, encode(tok, recs, c["max_len"]), 2 * c["batch_size"], dev)}
    if args.smoke:
        print("smoke ok:", {k: float(np.mean(v["p"])) for k, v in test.items()})
        return
    out = save_run(args.dataset, name, args.seed,
                   meta={"config": c, "train_conditions": aug, "best_valid_f1": best, "n_train": len(train)},
                   valid={"ids": [r["id"] for r in valid], "y": yva.tolist(), "p": pva}, test=test)
    print("saved", out)


if __name__ == "__main__":
    main()
