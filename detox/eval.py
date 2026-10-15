"""Balanced accuracy overall and per ethnicity (E-A / E-B)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from detox.data.utkface import UTKFaceDataset, resolve_utkface_root
from detox.models.classifier import FaceClassifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


@torch.no_grad()
def eval_model(model, ds, device, bs, nw, tag) -> Dict:
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    ys, ps, rs = [], [], []
    model.eval()
    for images, labels, race in tqdm(loader, desc=tag, leave=False):
        logits, _ = model(images.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(labels.numpy())
        ps.append(pred)
        rs.append(race.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    r = np.concatenate(rs)
    n_classes = int(ds.num_classes)
    out = {
        "balanced_accuracy": balanced_accuracy(y, p, n_classes),
        "accuracy": float((y == p).mean()),
        "n": int(len(y)),
    }
    for name, rid in (("ea", ds.ea), ("eb", ds.eb)):
        mask = r == rid
        if mask.any():
            out[f"bacc_{name}"] = balanced_accuracy(y[mask], p[mask], n_classes)
            out[f"acc_{name}"] = float((y[mask] == p[mask]).mean())
            out[f"n_{name}"] = int(mask.sum())
    # disparity |bacc_ea - bacc_eb|
    if "bacc_ea" in out and "bacc_eb" in out:
        out["bacc_gap"] = abs(out["bacc_ea"] - out["bacc_eb"])
    print(
        f"[{tag}] bacc={out['balanced_accuracy']:.4f} "
        f"ea={out.get('bacc_ea')} eb={out.get('bacc_eb')} gap={out.get('bacc_gap')}"
    )
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_utkface_root(cfg)
    image_size = int(cfg["model"].get("image_size", 112))
    test_ds = UTKFaceDataset(
        root,
        task=str(cfg["data"].get("task", "gender")),
        protocol=int(cfg["data"].get("protocol", 1)),
        eb_ratio=float(cfg["data"].get("eb_ratio", 0.1)),
        split="test",
        image_size=image_size,
        seed=int(cfg["train"].get("seed", 42)),
        max_n=cfg["eval"].get("max_images"),
        train=False,
    )
    methods: List[str] = list(cfg["train"].get("methods", ["softmax", "detox"]))
    bs = int(cfg["eval"].get("batch_size", 64))
    nw = int(cfg["train"].get("num_workers", 4))
    results = {"methods": {}}
    for method in methods:
        ckpt = torch.load(
            Path(cfg["paths"]["checkpoint_dir"]) / f"{method}.pt",
            map_location=device,
            weights_only=False,
        )
        model = FaceClassifier(
            int(ckpt["num_classes"]), emb_dim=int(ckpt.get("emb_dim", 128)), pretrained=False
        ).to(device)
        model.load_state_dict(ckpt["model"])
        results["methods"][method] = eval_model(model, test_ds, device, bs, nw, method)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
