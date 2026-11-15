"""Predeclared content-aware cluster multiplicity for the VisionZip pilot.

This is a heuristic computed once from the VisionZip assignment keys, not an
estimate of decoder attention mass or a claimed exact softmax correction.
Dominant tokens retain weight 1. Context anchors receive 1 plus the sum of
clipped, exponentiated cosine similarities of their assigned members.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def cluster_weights(key, selection, exponent=8):
    """Return (integer counts, effective weights), in sorted keep order."""
    keep, _, _, diag = selection
    tensors = diag["tensors"]
    anchors = tensors["anchor_idx"].long()
    members = tensors["assignee_idx"].long()
    assignment = tensors["assignment"].long()
    if key.ndim != 2 or not bool(torch.isfinite(key).all()):
        raise ValueError("assignment keys must be a finite [N,D] tensor")
    if exponent != 8:
        raise ValueError("this predeclared pilot fixes exponent=8")
    if members.shape != assignment.shape or anchors.numel() == 0:
        raise ValueError("invalid VisionZip assignment")
    if assignment.numel() and (int(assignment.min()) < 0 or int(assignment.max()) >= anchors.numel()):
        raise ValueError("cluster assignment outside anchor range")
    rows = torch.searchsorted(keep, anchors)
    if not torch.equal(keep[rows], anchors):
        raise ValueError("all anchors must belong to the ordered keep set")
    unit = F.normalize(key.float(), p=2, dim=-1, eps=1e-8)
    similarity = (unit[members] * unit[anchors[assignment]]).sum(dim=-1).clamp(0, 1)
    count_clusters = torch.ones(anchors.numel(), dtype=torch.float32, device=key.device)
    count_clusters.index_add_(0, assignment, torch.ones_like(similarity))
    effective_clusters = torch.ones_like(count_clusters)
    effective_clusters.index_add_(0, assignment, similarity.pow(exponent))
    count = torch.ones(keep.numel(), dtype=torch.float32, device=key.device)
    effective = torch.ones_like(count)
    count[rows] = count_clusters
    effective[rows] = effective_clusters
    if not torch.equal(count_clusters.to(tensors["cluster_sizes"].dtype), tensors["cluster_sizes"]):
        raise AssertionError("counts disagree with frozen selector assignments")
    if not bool(((effective >= 1) & (effective <= count + 1e-5)).all()):
        raise AssertionError("effective multiplicity is outside [1, count]")
    return count, effective
