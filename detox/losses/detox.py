"""Detox loss: CE + Exclusion + Inclusion + Feature-distillation (Eq. 6)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _kmeans(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """Simple batched k-means; returns cluster id per sample. x: (N,D)."""
    n = x.shape[0]
    k = min(k, n)
    # init from random samples
    idx = torch.randperm(n, device=x.device)[:k]
    centers = x[idx].clone()
    for _ in range(iters):
        dist = torch.cdist(x, centers)
        assign = dist.argmin(dim=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
    return assign


def exclusion_loss(emb: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Push minority-class samples in each cluster toward their class mean (Eq. 2)."""
    if emb.shape[0] < 2:
        return emb.new_zeros(())
    # class means
    means = []
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            means.append(emb[mask].mean(dim=0))
        else:
            means.append(emb.mean(dim=0))
    means_t = torch.stack(means, dim=0)

    assign = _kmeans(emb.detach(), k=n_classes)
    loss = emb.new_zeros(())
    n_terms = 0
    for i in range(n_classes):
        in_c = assign == i
        if not in_c.any():
            continue
        labs = labels[in_c]
        # minority class in this cluster
        counts = torch.bincount(labs, minlength=n_classes)
        # ignore empty classes
        nonzero = counts > 0
        if nonzero.sum() == 0:
            continue
        # set empty to large so min picks among present
        scores = counts.clone().float()
        scores[~nonzero] = 1e9
        cmin = int(scores.argmin().item())
        sel = in_c & (labels == cmin)
        if not sel.any():
            continue
        loss = loss + ((emb[sel] - means_t[cmin].detach()) ** 2).sum(dim=1).mean()
        n_terms += 1
    if n_terms == 0:
        return emb.new_zeros(())
    return loss / n_terms


def inclusion_loss(emb: torch.Tensor) -> torch.Tensor:
    """Minimize feature variance toward batch mean (Eq. 4)."""
    mu = emb.mean(dim=0, keepdim=True)
    return ((emb - mu) ** 2).sum(dim=1).mean()


def feature_distill_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """L2 between student/teacher penultimate features (Eq. 5, single layer)."""
    return F.mse_loss(student_emb, teacher_emb.detach())


def detox_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    student_emb: torch.Tensor,
    teacher_emb: Optional[torch.Tensor],
    n_classes: int,
    lambda1: float,
    lambda2: float,
    lambda3: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ce = F.cross_entropy(logits, labels)
    lex = exclusion_loss(student_emb, labels, n_classes)
    lin = inclusion_loss(student_emb)
    if teacher_emb is not None and lambda3 > 0:
        lfd = feature_distill_loss(student_emb, teacher_emb)
    else:
        lfd = student_emb.new_zeros(())
    total = ce + lambda1 * lex + lambda2 * lin + lambda3 * lfd
    stats = {
        "ce": float(ce.detach()),
        "exclusion": float(lex.detach()),
        "inclusion": float(lin.detach()),
        "fdistill": float(lfd.detach()),
        "total": float(total.detach()),
    }
    return total, stats
