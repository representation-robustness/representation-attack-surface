#!/usr/bin/env python3
"""
eval_sequence_7cond.py

Evaluate TF-IDF+LR, CodeBERT, and CodeT5+ on all 7 obfuscation conditions
for Devign, BigVul, and DiverseVul.

Usage:
    CUDA_VISIBLE_DEVICES=X python eval_sequence_7cond.py --model tfidf
    CUDA_VISIBLE_DEVICES=X python eval_sequence_7cond.py --model codebert
    CUDA_VISIBLE_DEVICES=X python eval_sequence_7cond.py --model codet5plus
    CUDA_VISIBLE_DEVICES=X python eval_sequence_7cond.py  # runs all

Output:
    ~/thesis/devign_full/{model}_7cond_{dataset}_results.json
"""

import argparse, json, os, sys, random
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

THESIS    = Path(__file__).resolve().parent
RESULT    = THESIS / "devign_full"
CKPT_BASE = THESIS / "baselines" / "codebert"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIRS = {
    "devign":     Path.home() / "GNN-ReGVD/dataset_devign",
    "bigvul":     Path.home() / "GNN-ReGVD/dataset_bigvul",
    "diversevul": Path.home() / "GNN-ReGVD/dataset_diversevul",
}

CONDITIONS = [
    ("original",    "test"),
    ("identifier",  "test_obf_identifier"),
    ("deadcode",    "test_obf_deadcode"),
    ("controlflow", "test_obf_controlflow"),
    ("ren_dead",    "test_obf_ren_dead"),
    ("ren_cf",      "test_obf_ren_cf"),
    ("dead_cf",     "test_obf_dead_cf"),
    ("compound",    "test_obf_compound"),
]

# CodeBERT checkpoints per dataset
CODEBERT_CKPTS = {
    "devign":     {"42": CKPT_BASE / "ckpts_multiseed/codebert_seed42.pt"},
    "bigvul":     {
        "42":   CKPT_BASE / "ckpts_multiseed/codebert_bigvul_seed42.pt",
        "1337": CKPT_BASE / "ckpts_multiseed/codebert_bigvul_seed1337.pt",
        "7":    CKPT_BASE / "ckpts_multiseed/codebert_bigvul_seed7.pt",
        "100":  CKPT_BASE / "ckpts_multiseed/codebert_bigvul_seed100.pt",
        "999":  CKPT_BASE / "ckpts_multiseed/codebert_bigvul_seed999.pt",
    },
    "diversevul": {
        "42":   CKPT_BASE / "ckpts_diversevul_codebert/codebert_dv_seed42.pt",
        "1337": CKPT_BASE / "ckpts_diversevul_codebert/codebert_dv_seed1337.pt",
        "7":    CKPT_BASE / "ckpts_diversevul_codebert/codebert_dv_seed7.pt",
        "100":  CKPT_BASE / "ckpts_diversevul_codebert/codebert_dv_seed100.pt",
        "999":  CKPT_BASE / "ckpts_diversevul_codebert/codebert_dv_seed999.pt",
    },
}

CODET5_CKPTS = {
    "devign": {
        "42":   CKPT_BASE / "ckpts_codet5plus/codet5plus_seed42.pt",
        "1337": CKPT_BASE / "ckpts_codet5plus/codet5plus_seed1337.pt",
        "7":    CKPT_BASE / "ckpts_codet5plus/codet5plus_seed7.pt",
        "100":  CKPT_BASE / "ckpts_codet5plus/codet5plus_seed100.pt",
        "999":  CKPT_BASE / "ckpts_codet5plus/codet5plus_seed999.pt",
    },
    "bigvul": {
        "42":   CKPT_BASE / "ckpts_codet5plus/codet5plus_bigvul_seed42.pt",
        "1337": CKPT_BASE / "ckpts_codet5plus/codet5plus_bigvul_seed1337.pt",
        "7":    CKPT_BASE / "ckpts_codet5plus/codet5plus_bigvul_seed7.pt",
        "100":  CKPT_BASE / "ckpts_codet5plus/codet5plus_bigvul_seed100.pt",
        "999":  CKPT_BASE / "ckpts_codet5plus/codet5plus_bigvul_seed999.pt",
    },
    "diversevul": {
        "42":   CKPT_BASE / "ckpts_diversevul_codet5plus/codet5plus_dv_seed42.pt",
        "1337": CKPT_BASE / "ckpts_diversevul_codet5plus/codet5plus_dv_seed1337.pt",
        "7":    CKPT_BASE / "ckpts_diversevul_codet5plus/codet5plus_dv_seed7.pt",
        "100":  CKPT_BASE / "ckpts_diversevul_codet5plus/codet5plus_dv_seed100.pt",
        "999":  CKPT_BASE / "ckpts_diversevul_codet5plus/codet5plus_dv_seed999.pt",
    },
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            rows.append({"func": r["func"], "target": int(r["target"])})
    return rows


def metrics(y_true, y_pred):
    return {
        "f1":  round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "acc": round(accuracy_score(y_true, y_pred) * 100, 2),
        "pr":  round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "rc":  round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
    }


def aggregate_seeds(per_seed):
    f1s = [r["f1"] for r in per_seed]
    return {
        "f1_mean": round(float(np.mean(f1s)), 2),
        "f1_std":  round(float(np.std(f1s)), 2),
        "all_f1":  f1s,
    }


# ─── TF-IDF ──────────────────────────────────────────────────────────────────

def run_tfidf(datasets):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    orig_file = THESIS / "devign_full/originals_full_data_with_slices.json"
    split_file = THESIS / "devign_full/devign_full_split_801010.json"

    for dataset in datasets:
        out_path = RESULT / f"tfidf_7cond_{dataset}_results.json"
        if out_path.exists():
            print(f"  SKIP tfidf {dataset} — already done")
            continue

        data_dir = DATASET_DIRS[dataset]
        print(f"\n=== TF-IDF {dataset} ===", flush=True)

        if dataset == "devign":
            split = json.load(open(split_file))
            orig  = {r["file_name"]: r for r in json.load(open(orig_file))}
            train_funcs  = [orig[n]["code"]  for n in split["splits"]["train"]  if n in orig]
            train_labels = [int(orig[n]["label"]) for n in split["splits"]["train"] if n in orig]
        else:
            train_rows   = load_jsonl(data_dir / "train.jsonl")
            train_funcs  = [r["func"]   for r in train_rows]
            train_labels = [r["target"] for r in train_rows]

        print(f"  Train: {len(train_funcs)}", flush=True)

        TFIDF_PARAMS = dict(max_features=10000, sublinear_tf=True,
                            ngram_range=(1, 2), analyzer="word",
                            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=3)

        N_TRIALS = 5
        per_cond = {cname: [] for cname, _ in CONDITIONS}

        for trial in range(N_TRIALS):
            seed = 42 + trial
            random.seed(seed); np.random.seed(seed)
            vec = TfidfVectorizer(**TFIDF_PARAMS)
            X_train = vec.fit_transform(train_funcs)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     C=1.0, random_state=seed)
            clf.fit(X_train, train_labels)

            for cname, split_name in CONDITIONS:
                jsonl_path = data_dir / f"{split_name}.jsonl"
                if not jsonl_path.exists():
                    continue
                rows = load_jsonl(jsonl_path)
                X_test = vec.transform([r["func"] for r in rows])
                y_pred = clf.predict(X_test)
                y_true = [r["target"] for r in rows]
                per_cond[cname].append(metrics(y_true, y_pred)["f1"])

            print(f"  Trial {trial+1}/{N_TRIALS} done", flush=True)

        results = {}
        base_f1 = None
        for cname, _ in CONDITIONS:
            if not per_cond[cname]:
                continue
            f1s = per_cond[cname]
            results[cname] = {
                "f1_mean": round(float(np.mean(f1s)), 2),
                "f1_std":  round(float(np.std(f1s)), 2),
                "all_f1":  f1s,
            }
            if cname == "original":
                base_f1 = results[cname]["f1_mean"]
        if base_f1:
            for cname in results:
                if cname != "original":
                    results[cname]["delta_f1"] = round(results[cname]["f1_mean"] - base_f1, 2)

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved → {out_path.name}", flush=True)


# ─── CodeBERT ────────────────────────────────────────────────────────────────

def run_codebert(datasets):
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    from torch.utils.data import Dataset as TorchDataset, DataLoader

    MODEL_NAME = "microsoft/codebert-base"
    MAX_LEN    = 512
    BATCH      = 32

    class SimpleDataset(TorchDataset):
        def __init__(self, rows, tokenizer):
            self.rows = rows
            self.tok  = tokenizer
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            enc = self.tok(self.rows[i]["func"], max_length=MAX_LEN,
                           padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(self.rows[i]["target"], dtype=torch.long)}

    @torch.no_grad()
    def infer(model, loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(DEVICE),
                           attention_mask=batch["attention_mask"].to(DEVICE)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(batch["label"].tolist())
        return preds, truths

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    base_model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    for dataset in datasets:
        out_path = RESULT / f"codebert_7cond_{dataset}_results.json"
        if out_path.exists():
            print(f"  SKIP codebert {dataset} — already done")
            continue

        data_dir = DATASET_DIRS[dataset]
        ckpts    = CODEBERT_CKPTS.get(dataset, {})
        print(f"\n=== CodeBERT {dataset} ({len(ckpts)} seeds) ===", flush=True)

        per_cond = {cname: [] for cname, _ in CONDITIONS}

        for seed_str, ckpt_path in ckpts.items():
            if not ckpt_path.exists():
                print(f"  SKIP seed {seed_str}: ckpt not found", flush=True)
                continue
            print(f"\n  Seed {seed_str}", flush=True)
            import copy
            model = copy.deepcopy(base_model)
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.to(DEVICE)

            for cname, split_name in CONDITIONS:
                jsonl_path = data_dir / f"{split_name}.jsonl"
                if not jsonl_path.exists():
                    continue
                rows   = load_jsonl(jsonl_path)
                loader = DataLoader(SimpleDataset(rows, tokenizer), batch_size=BATCH, shuffle=False)
                preds, truths = infer(model, loader)
                m = metrics(truths, preds)
                per_cond[cname].append(m["f1"])
                print(f"    {cname}: F1={m['f1']:.2f}%", flush=True)

            del model
            torch.cuda.empty_cache()

        results = {}
        base_f1 = None
        for cname, _ in CONDITIONS:
            if not per_cond[cname]:
                continue
            f1s = per_cond[cname]
            results[cname] = {
                "f1_mean": round(float(np.mean(f1s)), 2),
                "f1_std":  round(float(np.std(f1s)), 2),
                "all_f1":  f1s,
            }
            if cname == "original":
                base_f1 = results[cname]["f1_mean"]
        if base_f1:
            for cname in results:
                if cname != "original":
                    results[cname]["delta_f1"] = round(results[cname]["f1_mean"] - base_f1, 2)

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved → {out_path.name}", flush=True)


# ─── CodeT5+ ─────────────────────────────────────────────────────────────────

def run_codet5(datasets):
    import torch.nn as nn
    from transformers import AutoTokenizer, T5EncoderModel
    from torch.utils.data import Dataset as TorchDataset, DataLoader

    MODEL_NAME = "Salesforce/codet5p-220m"
    MAX_LEN    = 512
    BATCH      = 16

    class CodeT5PlusClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder    = T5EncoderModel.from_pretrained(MODEL_NAME)
            hidden          = self.encoder.config.d_model
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, 2))
        def forward(self, input_ids, attention_mask):
            out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            return self.classifier(pooled)

    class SimpleDataset(TorchDataset):
        def __init__(self, rows, tokenizer):
            self.rows = rows
            self.tok  = tokenizer
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            enc = self.tok(self.rows[i]["func"], max_length=MAX_LEN,
                           padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": torch.tensor(self.rows[i]["target"], dtype=torch.long)}

    @torch.no_grad()
    def infer(model, loader):
        model.eval()
        preds, truths = [], []
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(DEVICE),
                           attention_mask=batch["attention_mask"].to(DEVICE))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(batch["label"].tolist())
        return preds, truths

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = CodeT5PlusClassifier()

    for dataset in datasets:
        out_path = RESULT / f"codet5_7cond_{dataset}_results.json"
        if out_path.exists():
            print(f"  SKIP codet5 {dataset} — already done")
            continue

        data_dir = DATASET_DIRS[dataset]
        ckpts    = CODET5_CKPTS.get(dataset, {})
        print(f"\n=== CodeT5+ {dataset} ({len(ckpts)} seeds) ===", flush=True)

        per_cond = {cname: [] for cname, _ in CONDITIONS}

        for seed_str, ckpt_path in ckpts.items():
            if not ckpt_path.exists():
                print(f"  SKIP seed {seed_str}: ckpt not found", flush=True)
                continue
            print(f"\n  Seed {seed_str}", flush=True)
            import copy
            model = copy.deepcopy(base_model)
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.to(DEVICE)

            for cname, split_name in CONDITIONS:
                jsonl_path = data_dir / f"{split_name}.jsonl"
                if not jsonl_path.exists():
                    continue
                rows   = load_jsonl(jsonl_path)
                loader = DataLoader(SimpleDataset(rows, tokenizer), batch_size=BATCH, shuffle=False)
                preds, truths = infer(model, loader)
                m = metrics(truths, preds)
                per_cond[cname].append(m["f1"])
                print(f"    {cname}: F1={m['f1']:.2f}%", flush=True)

            del model
            torch.cuda.empty_cache()

        results = {}
        base_f1 = None
        for cname, _ in CONDITIONS:
            if not per_cond[cname]:
                continue
            f1s = per_cond[cname]
            results[cname] = {
                "f1_mean": round(float(np.mean(f1s)), 2),
                "f1_std":  round(float(np.std(f1s)), 2),
                "all_f1":  f1s,
            }
            if cname == "original":
                base_f1 = results[cname]["f1_mean"]
        if base_f1:
            for cname in results:
                if cname != "original":
                    results[cname]["delta_f1"] = round(results[cname]["f1_mean"] - base_f1, 2)

        out_path.write_text(json.dumps(results, indent=2))
        print(f"  Saved → {out_path.name}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="all", choices=["all", "tfidf", "codebert", "codet5plus"])
    parser.add_argument("--dataset", default="all", choices=["all", "devign", "bigvul", "diversevul"])
    args = parser.parse_args()

    datasets = ["devign", "bigvul", "diversevul"] if args.dataset == "all" else [args.dataset]
    models   = ["tfidf", "codebert", "codet5plus"] if args.model == "all" else [args.model]

    print(f"Device: {DEVICE}", flush=True)
    print(f"Models:   {models}", flush=True)
    print(f"Datasets: {datasets}", flush=True)

    if "tfidf" in models:
        run_tfidf(datasets)
    if "codebert" in models:
        run_codebert(datasets)
    if "codet5plus" in models:
        run_codet5(datasets)

    print("\nDone.", flush=True)
