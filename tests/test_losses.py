"""Detox losses (Eqs. 2-6) on hand-computed values.

The k-means step inside the exclusion loss is random, so the tests that need exact values fix its assignment.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import detox.losses.detox as detox_losses
from detox.losses.detox import detox_loss, exclusion_loss, feature_distill_loss, inclusion_loss


def test_exclusion_penalises_the_minority_class_of_each_cluster(monkeypatch):
    # Cluster 0 holds two class-0 samples and one class-1 sample, so class 1 is its minority; cluster 1 holds one
    # class-1 sample. The class-1 mean over the whole batch is (1 + 5) / 2 = 3.
    monkeypatch.setattr(detox_losses, "_kmeans", lambda x, k, iters=10: torch.tensor([0, 0, 0, 1]))
    emb = torch.tensor([[0.0], [2.0], [1.0], [5.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # Cluster 0: the class-1 sample at 1.0 is pulled to the class-1 mean 3.0, a penalty of (1 - 3)^2 = 4.
    # Cluster 1: the class-1 sample at 5.0 is pulled to 3.0, a penalty of (5 - 3)^2 = 4. The mean over clusters is 4.
    assert exclusion_loss(emb, labels, n_classes=2).item() == pytest.approx(4.0)


def test_exclusion_is_zero_when_every_cluster_holds_one_class(monkeypatch):
    monkeypatch.setattr(detox_losses, "_kmeans", lambda x, k, iters=10: torch.tensor([0, 0, 1, 1]))
    emb = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    assert exclusion_loss(emb, labels, n_classes=2).item() == pytest.approx(0.0)


def test_exclusion_is_finite_with_the_real_clustering():
    torch.manual_seed(0)
    emb = torch.randn(12, 4)
    labels = torch.tensor([0] * 8 + [1] * 4)
    value = exclusion_loss(emb, labels, n_classes=2)
    assert torch.isfinite(value) and value.item() >= 0.0


def test_exclusion_needs_at_least_two_samples():
    assert exclusion_loss(torch.zeros(1, 3), torch.tensor([0]), n_classes=2).item() == 0.0


def test_inclusion_is_the_mean_squared_deviation_from_the_batch_mean():
    emb = torch.tensor([[0.0, 0.0], [2.0, 0.0]])  # the mean is (1, 0), so the deviations are +-1 along x
    assert inclusion_loss(emb).item() == pytest.approx(0.5)
    assert inclusion_loss(torch.ones(4, 3)).item() == pytest.approx(0.0)


def test_feature_distillation_is_a_mse_that_does_not_train_the_teacher():
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0]], requires_grad=True)
    loss = feature_distill_loss(student, teacher)
    assert loss.item() == pytest.approx((1.0 + 4.0) / 2)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_total_loss_is_cross_entropy_plus_the_weighted_terms(monkeypatch):
    monkeypatch.setattr(detox_losses, "_kmeans", lambda x, k, iters=10: torch.tensor([0, 0, 1, 1]))
    logits = torch.zeros(4, 2)  # uniform over two classes, so the cross-entropy is log 2
    labels = torch.tensor([0, 0, 1, 1])
    emb = torch.tensor([[0.0], [2.0], [1.0], [3.0]])
    teacher = torch.zeros(4, 1)
    total, stats = detox_loss(logits, labels, emb, teacher, n_classes=2, lambda1=0.5, lambda2=0.25, lambda3=2.0)
    lex = exclusion_loss(emb, labels, 2).item()
    lin = inclusion_loss(emb).item()
    lfd = feature_distill_loss(emb, teacher).item()
    assert total.item() == pytest.approx(math.log(2) + 0.5 * lex + 0.25 * lin + 2.0 * lfd, rel=1e-6)
    assert stats["total"] == pytest.approx(total.item())
    assert stats["exclusion"] == pytest.approx(lex)
    assert stats["inclusion"] == pytest.approx(lin)
    assert stats["fdistill"] == pytest.approx(lfd)


def test_without_a_teacher_and_with_zero_weights_the_loss_is_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 0])
    emb = torch.randn(4, 5)
    total, stats = detox_loss(logits, labels, emb, None, n_classes=3, lambda1=0.0, lambda2=0.0, lambda3=0.0)
    assert total.item() == pytest.approx(F.cross_entropy(logits, labels).item())
    assert stats["fdistill"] == 0.0


def test_detox_loss_back_propagates_into_the_student_embedding():
    torch.manual_seed(0)
    logits = torch.randn(6, 2, requires_grad=True)
    emb = torch.randn(6, 4, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 0])
    total, _ = detox_loss(logits, labels, emb, torch.randn(6, 4), 2, 0.1, 0.1, 0.1)
    total.backward()
    assert emb.grad is not None and torch.isfinite(emb.grad).all()
