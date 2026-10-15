"""Predict gender/age-group for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from detox.models.classifier import FaceClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="detox")
    ap.add_argument("--out", default="outputs/predict.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / f"{args.method}.pt",
        map_location=device,
        weights_only=False,
    )
    model = FaceClassifier(
        int(ckpt["num_classes"]), emb_dim=int(ckpt.get("emb_dim", 128)), pretrained=False
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    image_size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred = int(logits.argmax(dim=1).item())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"image": args.image, "method": args.method, "pred": pred, "prob": prob}, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
