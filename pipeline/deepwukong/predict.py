"""Per-sample DeepWukong predictions for one trained seed on valid and every built test condition.

  python -m pipeline.deepwukong.predict --seed 42 [--device cpu]

Writes results/preds/deepwukong/deepwukong/run_<seed>.json.gz in the format of every other detector:
valid {ids, y, p}, test {condition: {ids, y, p}}, threshold (validation-tuned, as for all detectors) and
the argmax threshold 0.5 that DeepWukong itself uses. Test conditions: "clean_shipped" (authors' test XFGs),
"clean" (the same XFGs rebuilt from source by pipeline.deepwukong.xfg) and each transformed condition;
y is always the clean ground-truth label of the paired XFG.
"""
import argparse
import gzip
import json

import numpy as np

from pipeline.deepwukong.common import BASE, CONDITIONS, ROOT, SHIPPED, use_upstream, work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    use_upstream(BASE / "train" / f"s{a.seed}")
    import torch
    from src.datas.graphs import XFG
    from src.datas.samples import XFGBatch, XFGSample
    from src.models.vd import DeepWuKong
    from sklearn.metrics import f1_score

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = DeepWuKong.load_from_checkpoint(str(ROOT / "runs/ckpt/deepwukong" / f"seed{a.seed}.ckpt"),
                                            map_location=dev).to(dev).eval()
    cfg, vocab = model.hparams["config"], model.hparams["vocab"]

    @torch.no_grad()
    def score(paths):
        ids, ys, ps = [], [], []
        for i in range(0, len(paths), 256):
            chunk = paths[i:i + 256]
            xs = [XFG(path=p) for _, p in chunk]
            batch = XFGBatch([XFGSample(graph=x.to_torch(vocab, cfg.dataset.token.max_parts), label=x.label) for x in xs])
            batch.move_to_device(dev)
            ps += torch.softmax(model(batch.graphs), dim=-1)[:, 1].cpu().tolist()
            ys += [int(x.label) for x in xs]
            ids += [i_ for i_, _ in chunk]
        return {"ids": ids, "y": ys, "p": ps}

    def shipped(split):
        return [("/".join(p.split("/")[3:6]).rsplit(".xfg.pkl", 1)[0], str(SHIPPED / "XFG" / "/".join(p.split("/")[3:])))
                for p in json.loads((SHIPPED / f"{split}.json").read_text())]

    valid = score(shipped("val"))
    yv, pv = np.array(valid["y"]), np.array(valid["p"])
    cand = np.unique(pv)
    thr = float(max(cand, key=lambda t: f1_score(yv, pv >= t))) if len(cand) < 5000 else \
        float(max(np.quantile(pv, np.linspace(0, 1, 2001)), key=lambda t: f1_score(yv, pv >= t)))
    test = {"clean_shipped": score(shipped("test"))}
    for c in CONDITIONS + [f"v1_{c}" for c in CONDITIONS[1:]]:
        f = work(c) / "xfg_paths.json"
        if f.exists():
            test[c] = score([(e["id"], e["path"]) for e in json.loads(f.read_text())])
    out = ROOT / "results" / "preds" / "deepwukong" / "deepwukong" / f"run_{a.seed}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump({"model": "deepwukong", "dataset": "deepwukong", "seed": a.seed, "threshold": thr,
                   "argmax_threshold": 0.5, "valid": valid, "test": test}, f)
    print("saved", out, {k: len(v["ids"]) for k, v in test.items()}, "threshold", round(thr, 4))


if __name__ == "__main__":
    main()
