"""The classifier, the training entry point run for one epoch on CPU, and the configuration files.

ResNet-18 is built without ImageNet weights in these tests, so nothing is downloaded.
"""

from __future__ import annotations

import json

import pytest
import torch
import torchvision

from detox.models.classifier import FaceClassifier
from detox.train import train


@pytest.fixture
def untrained_resnet(monkeypatch):
    original = torchvision.models.resnet18
    monkeypatch.setattr(torchvision.models, "resnet18", lambda weights=None, **kwargs: original(weights=None))


def test_classifier_returns_logits_and_the_embedding_it_used(untrained_resnet):
    torch.manual_seed(0)
    net = FaceClassifier(n_classes=2, emb_dim=16, pretrained=False).eval()
    x = torch.randn(3, 3, 32, 32)
    logits, emb = net(x)
    assert logits.shape == (3, 2)
    assert emb.shape == (3, 16)
    assert torch.allclose(net.embed(x), emb)


def test_training_runs_one_epoch_for_both_methods_on_a_small_folder(tmp_path, untrained_resnet, make_utk):
    root = tmp_path / "utk"
    make_utk(root, {3: 24, 0: 120})
    cfg = {
        "device": "cpu",
        "data": {"utkface_root": str(root), "task": "gender", "protocol": 1, "eb_ratio": 0.5, "max_images": None},
        "model": {"image_size": 32, "emb_dim": 16},
        "detox": {"lambda1": 0.1, "lambda2": 0.1, "lambda3": 0.1},
        "train": {
            "protocol": "test",
            "methods": ["softmax", "detox"],
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.01,
            "num_workers": 0,
            "seed": 0,
        },
        "paths": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "figure_dir": str(tmp_path / "figs"),
        },
    }
    summary = train(cfg, device_override="cpu")
    assert [m["method"] for m in summary["methods"]] == ["softmax", "detox"]
    for method in ("softmax", "detox"):
        checkpoint = torch.load(tmp_path / "ckpt" / f"{method}.pt", weights_only=False)
        assert checkpoint["method"] == method
        assert checkpoint["num_classes"] == 2
        assert checkpoint["emb_dim"] == 16
    written = json.loads((tmp_path / "logs" / "train_summary.json").read_text(encoding="utf-8"))
    assert written["n_train"] == summary["n_train"] > 0
    assert all(torch.isfinite(torch.tensor(m["history"][0]["loss"])) for m in summary["methods"])


def test_smoke_and_default_configs_share_the_task_protocol_and_input_size(default_cfg, smoke_cfg):
    assert default_cfg["data"]["task"] == smoke_cfg["data"]["task"] == "gender"
    assert default_cfg["data"]["protocol"] == smoke_cfg["data"]["protocol"] == 1
    assert default_cfg["data"]["eb_ratio"] == smoke_cfg["data"]["eb_ratio"] == 0.1
    assert default_cfg["model"]["image_size"] == smoke_cfg["model"]["image_size"] == 112


def test_smoke_run_is_smaller_than_the_full_run(default_cfg, smoke_cfg):
    assert smoke_cfg["train"]["epochs"] <= default_cfg["train"]["epochs"]
    assert smoke_cfg["train"]["batch_size"] <= default_cfg["train"]["batch_size"]
    assert smoke_cfg["data"]["max_images"] is not None


def test_the_full_run_trains_softmax_and_detox_with_the_teacher_term(default_cfg):
    assert set(default_cfg["train"]["methods"]) == {"softmax", "detox"}
    assert default_cfg["detox"]["lambda3"] > 0
