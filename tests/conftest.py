"""Shared test setup: repo root on sys.path, the two configuration files, and a synthetic UTKFace maker."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def smoke_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


def make_utkface(root: Path, counts: dict) -> None:
    """Write tiny images named age_gender_race_date.jpg, with counts[race] images of each race."""
    root.mkdir(parents=True, exist_ok=True)
    i = 0
    for race, n in counts.items():
        for _ in range(n):
            age = 5 + (i * 7) % 80
            gender = i % 2
            Image.new("RGB", (8, 8), color=(i % 255, 0, 0)).save(root / f"{age}_{gender}_{race}_2017{i:08d}.jpg")
            i += 1


@pytest.fixture
def make_utk():
    return make_utkface
