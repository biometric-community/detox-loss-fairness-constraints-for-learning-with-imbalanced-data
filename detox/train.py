"""Train Softmax vs Detox gender classifiers on imbalanced UTKFace."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from detox.data.utkface import UTKFaceDataset, race_counts, resolve_utkface_root
from detox.losses.detox import detox_loss
from detox.models.classifier import FaceClassifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _train_one(
    cfg: dict,
    method: str,
    train_ds: UTKFaceDataset,
    device: torch.device,
) -> Dict:
    emb_dim = int(cfg["model"].get("emb_dim", 128))
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    student = FaceClassifier(train_ds.num_classes, emb_dim=emb_dim, pretrained=True).to(device)
    teacher = None
    if method == "detox" and float(cfg["detox"].get("lambda3", 0.0)) > 0:
        teacher = FaceClassifier(train_ds.num_classes, emb_dim=emb_dim, pretrained=True).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt = torch.optim.SGD(
        student.parameters(),
        lr=float(cfg["train"].get("lr", 0.05)),
        momentum=0.9,
        weight_decay=5e-4,
    )
    epochs = int(cfg["train"].get("epochs", 8))
    l1 = float(cfg["detox"].get("lambda1", 0.1))
    l2 = float(cfg["detox"].get("lambda2", 0.1))
    l3 = float(cfg["detox"].get("lambda3", 0.1))
    hist = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        student.train()
        losses = []
        pbar = tqdm(loader, desc=f"{method} ep{epoch}/{epochs}", leave=False)
        for images, labels, _race in pbar:
            images, labels = images.to(device), labels.to(device)
            logits, emb = student(images)
            if method == "detox":
                t_emb = teacher.embed(images) if teacher is not None else None
                loss, stats = detox_loss(
                    logits, labels, emb, t_emb, train_ds.num_classes, l1, l2, l3
                )
            else:
                loss = F.cross_entropy(logits, labels)
                stats = {"ce": float(loss.detach()), "total": float(loss.detach())}
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(stats["total"])
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        mean = float(np.mean(losses))
        hist.append({"epoch": epoch, "loss": mean})
        print(f"[train:{method}] epoch={epoch} loss={mean:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": student.state_dict(),
            "num_classes": train_ds.num_classes,
            "emb_dim": emb_dim,
            "method": method,
            "config": cfg,
        },
        ckpt_dir / f"{method}.pt",
    )
    return {
        "method": method,
        "epochs": epochs,
        "n_train": len(train_ds),
        "race_counts": race_counts(train_ds),
        "elapsed_sec": time.time() - t0,
        "history": hist,
    }


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_utkface_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    train_ds = UTKFaceDataset(
        root,
        task=str(cfg["data"].get("task", "gender")),
        protocol=int(cfg["data"].get("protocol", 1)),
        eb_ratio=float(cfg["data"].get("eb_ratio", 0.1)),
        split="train",
        image_size=int(cfg["model"].get("image_size", 112)),
        seed=seed,
        max_n=max_n,
        train=True,
    )
    methods: List[str] = list(cfg["train"].get("methods", ["softmax", "detox"]))
    summaries = [_train_one(cfg, m, train_ds, device) for m in methods]
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol": cfg["train"].get("protocol", "full"),
        "task": cfg["data"].get("task", "gender"),
        "utk_protocol": cfg["data"].get("protocol", 1),
        "eb_ratio": cfg["data"].get("eb_ratio", 0.1),
        "methods": summaries,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "race_counts": race_counts(train_ds),
    }
    (log_dir / "train_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[train] done methods={methods} n={len(train_ds)} counts={race_counts(train_ds)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
