"""UTKFace gender / age-group classification with ethnicity imbalance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# race codes in UTKFace filenames
RACE_WHITE, RACE_BLACK, RACE_ASIAN, RACE_INDIAN, RACE_OTHER = 0, 1, 2, 3, 4

PROTOCOLS = {
    # Protocol-1: E-A Indian, E-B White
    1: {"ea": RACE_INDIAN, "eb": RACE_WHITE},
    # Protocol-2: E-A White, E-B Asian
    2: {"ea": RACE_WHITE, "eb": RACE_ASIAN},
}


def resolve_utkface_root(cfg: dict) -> Path:
    root = Path(cfg["data"].get("utkface_root", "../../datasets/utkface/extracted"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _parse(path: Path) -> Optional[Tuple[int, int, int]]:
    parts = path.name.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _age_group(age: int) -> int:
    # coarse bins used for facial analysis lab twin
    if age <= 19:
        return 0
    if age <= 39:
        return 1
    if age <= 59:
        return 2
    return 3


def _tf(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(ops)


class UTKFaceDataset(Dataset):
    def __init__(
        self,
        root: Path,
        task: str = "gender",
        protocol: int = 1,
        eb_ratio: float = 0.5,
        split: str = "train",
        image_size: int = 112,
        seed: int = 42,
        max_n: Optional[int] = None,
        train: bool = False,
    ) -> None:
        self.transform = _tf(image_size, train=train)
        self.task = task
        proto = PROTOCOLS[int(protocol)]
        ea, eb = proto["ea"], proto["eb"]

        rows: List[Tuple[Path, int, int]] = []  # path, label, race
        for p in sorted(Path(root).rglob("*.jpg")):
            parsed = _parse(p)
            if parsed is None:
                continue
            age, gender, race = parsed
            if race not in (ea, eb):
                continue
            if task == "gender":
                label = int(gender)
            else:
                label = _age_group(age)
            rows.append((p, label, race))

        rng = np.random.RandomState(seed)
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        rows = [rows[i] for i in idx]

        # subject-independent random split by images
        n_train = int(round(0.7 * len(rows)))
        if split == "train":
            pool = rows[:n_train]
            # imbalance: keep all EA, subsample EB to target ratio of training set
            ea_rows = [r for r in pool if r[2] == ea]
            eb_rows = [r for r in pool if r[2] == eb]
            # target: eb_ratio of combined = |EB| / (|EA|+|EB|)
            # fix |EA|, choose |EB| = eb_ratio/(1-eb_ratio) * |EA|
            if eb_ratio >= 1.0:
                keep_eb = eb_rows
            else:
                n_eb = int(round(eb_ratio / max(1e-6, 1.0 - eb_ratio) * len(ea_rows)))
                n_eb = min(n_eb, len(eb_rows))
                keep_eb = eb_rows[:n_eb]
            self.samples = ea_rows + keep_eb
            rng.shuffle(self.samples)
        else:
            # balanced eval: equal EA/EB from held-out
            pool = rows[n_train:]
            ea_rows = [r for r in pool if r[2] == ea]
            eb_rows = [r for r in pool if r[2] == eb]
            n = min(len(ea_rows), len(eb_rows))
            self.samples = ea_rows[:n] + eb_rows[:n]

        if max_n is not None:
            self.samples = self.samples[: int(max_n)]

        self.num_classes = 2 if task == "gender" else 4
        self.ea, self.eb = ea, eb

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, race = self.samples[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, int(label), int(race)


def race_counts(ds: UTKFaceDataset) -> Dict[str, int]:
    ea = sum(1 for _, _, r in ds.samples if r == ds.ea)
    eb = sum(1 for _, _, r in ds.samples if r == ds.eb)
    return {"ea": ea, "eb": eb, "total": len(ds)}
