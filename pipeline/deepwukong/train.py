"""Train DeepWukong (upstream code and configs/dwk.yaml) on the authors' clean CWE-119 train/val split.

  python -m pipeline.deepwukong.train --seed 42

Runs upstream src.train.train (Lightning: early stopping and checkpointing on val_loss, then a test pass on
the shipped test split) in $RERUN_ROOT/data/deepwukong/train/s<seed>/, with only the seed and num_workers
changed from the config. The checkpoint with the lowest val_loss is copied to
runs/ckpt/deepwukong/seed<seed>.ckpt for pipeline.deepwukong.predict.
"""
import argparse
import os
import re
import shutil
from pathlib import Path

from pipeline.deepwukong.common import BASE, ROOT, SHIPPED, UPSTREAM, use_upstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    cwd = BASE / "train" / f"s{a.seed}"
    use_upstream(cwd)
    link = cwd / "data" / "CWE119"
    if not link.exists():
        link.symlink_to(SHIPPED)
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    from src.datas.datamodules import XFGDataModule
    from src.models.vd import DeepWuKong
    from src.train import train
    from src.utils import filter_warnings
    from src.vocabulary import Vocabulary

    cap = os.environ.get("RERUN_GPU_MEM_MIB")  # set by runjob.sh when sharing an idle GPU
    if cap and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, int(cap) * 2**20 / torch.cuda.get_device_properties(0).total_memory), 0)
    filter_warnings()
    config = OmegaConf.load(UPSTREAM / "configs" / "dwk.yaml")
    config.seed, config.num_workers = a.seed, 3
    seed_everything(config.seed, workers=True)
    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    model = DeepWuKong(config, vocab, vocab.get_vocab_size(), vocab.get_pad_id())
    train(model, XFGDataModule(config, vocab), config)

    ckpts = list((cwd / "ts_logger").rglob("*.ckpt"))
    best = min(ckpts, key=lambda p: float(re.search(r"val_loss=([0-9.]+?)(?:-v\d+)?\.ckpt$", p.name).group(1)))
    dst = ROOT / "runs" / "ckpt" / "deepwukong" / f"seed{a.seed}.ckpt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, dst)
    print("best checkpoint", best, "->", dst)


if __name__ == "__main__":
    main()
