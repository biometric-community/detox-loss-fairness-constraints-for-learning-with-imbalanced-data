"""UTKFace filename parsing, age bins, the imbalanced training split, the balanced evaluation split and balanced accuracy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from detox.data.utkface import UTKFaceDataset, _age_group, _parse, race_counts, resolve_utkface_root
from detox.eval import balanced_accuracy


def test_utkface_filenames_give_age_gender_and_race():
    assert _parse(Path("25_0_1_20170116174525125.jpg")) == (25, 0, 1)
    assert _parse(Path("bad_name.jpg")) is None
    assert _parse(Path("25_x_1_2017.jpg")) is None


@pytest.mark.parametrize(
    "age,group",
    [(0, 0), (19, 0), (20, 1), (39, 1), (40, 2), (59, 2), (60, 3), (90, 3)],
)
def test_age_groups_are_the_four_coarse_bins(age, group):
    assert _age_group(age) == group


def test_balanced_accuracy_averages_per_class_recall_over_the_classes_present():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    # Recalls: class 0 is 1/2, class 1 is 2/2, class 2 is 1/1. Class 3 is absent from y_true and does not count.
    assert balanced_accuracy(y_true, y_pred, n_classes=4) == pytest.approx((0.5 + 1.0 + 1.0) / 3)


def test_training_split_subsamples_the_majority_group_to_the_requested_ratio(tmp_path, make_utk):
    make_utk(tmp_path, {3: 40, 0: 200, 2: 30, 4: 20})
    ds = UTKFaceDataset(tmp_path, task="gender", protocol=1, eb_ratio=0.2, split="train", image_size=16, seed=1)
    counts = race_counts(ds)
    # Protocol 1 keeps Indian (EA, code 3) and White (EB, code 0). EB is cut to round(0.2 / 0.8 * EA) images.
    assert counts["eb"] == round(0.25 * counts["ea"])
    assert counts["ea"] > 0
    assert {race for _, _, race in ds.samples} <= {0, 3}
    assert ds.num_classes == 2


def test_evaluation_split_is_balanced_between_the_two_groups(tmp_path, make_utk):
    make_utk(tmp_path, {3: 40, 0: 200})
    ds = UTKFaceDataset(tmp_path, task="gender", protocol=1, eb_ratio=0.2, split="eval", image_size=16, seed=1)
    counts = race_counts(ds)
    assert counts["ea"] == counts["eb"] > 0


def test_age_task_labels_are_the_four_age_groups(tmp_path, make_utk):
    make_utk(tmp_path, {3: 40, 0: 200})
    ds = UTKFaceDataset(tmp_path, task="age", protocol=1, eb_ratio=0.5, split="eval", image_size=16, seed=0)
    assert ds.num_classes == 4
    assert all(0 <= label <= 3 for _, label, _ in ds.samples)


def test_protocol_two_compares_white_with_asian(tmp_path, make_utk):
    make_utk(tmp_path, {0: 60, 2: 60, 3: 60})
    ds = UTKFaceDataset(tmp_path, task="gender", protocol=2, eb_ratio=0.5, split="eval", image_size=16, seed=0)
    assert {race for _, _, race in ds.samples} <= {0, 2}
    assert ds.ea == 0 and ds.eb == 2


def test_items_are_normalised_images_with_their_label_and_race(tmp_path, make_utk):
    make_utk(tmp_path, {3: 20, 0: 60})
    ds = UTKFaceDataset(tmp_path, task="gender", protocol=1, eb_ratio=0.5, split="eval", image_size=16, seed=0)
    img, label, race = ds[0]
    assert tuple(img.shape) == (3, 16, 16)
    assert float(img.min()) >= -1.0 - 1e-6 and float(img.max()) <= 1.0 + 1e-6
    assert label in (0, 1)
    assert race in (0, 3)


def test_max_n_truncates_a_split(tmp_path, make_utk):
    make_utk(tmp_path, {3: 40, 0: 200})
    ds = UTKFaceDataset(tmp_path, task="gender", protocol=1, eb_ratio=0.5, split="eval", image_size=16, seed=0, max_n=6)
    assert len(ds) == 6


def test_missing_utkface_root_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_utkface_root({"data": {"utkface_root": str(tmp_path / "absent")}})
    assert resolve_utkface_root({"data": {"utkface_root": str(tmp_path)}}) == tmp_path
