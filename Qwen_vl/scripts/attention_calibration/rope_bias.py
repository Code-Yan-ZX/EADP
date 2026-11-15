"""Qwen3 interleaved mRoPE adaptation of RESTORE's distance term.

Each of the native half-head frequencies belongs to one of the three rotary
axes.  An unweighted mean across frequency slots equals the paper's weighted
mean of the three axis-specific averages.  No reindexing is performed.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def distance_log_bias(query_positions, key_positions, inv_freq, axes):
    """Compute log(2 - mean_f cos((q_axis - k_axis) * frequency)).

    Position inputs are [3, Q] and [3, K]; ``axes`` and ``inv_freq`` are [F].
    The returned matrix is [Q, K], with zero diagonal for equal positions.
    It is a fixed mathematical adaptation, not the authors' Qwen3 code.
    """
    q, k = query_positions, key_positions
    if q.ndim != 2 or k.ndim != 2 or q.shape[0] != 3 or k.shape[0] != 3:
        raise ValueError("query and key positions must be [3, length]")
    if axes.ndim != 1 or inv_freq.ndim != 1 or axes.numel() != inv_freq.numel():
        raise ValueError("frequency axis map does not match rotary frequencies")
    if axes.numel() == 0 or int(axes.min()) < 0 or int(axes.max()) > 2:
        raise ValueError("invalid rotary axis")
    axes = axes.to(q.device, dtype=torch.long)
    frequencies = inv_freq.to(q.device, dtype=torch.float32)
    qf = q.transpose(0, 1).float()[:, axes]
    kf = k.to(q.device).transpose(0, 1).float()[:, axes]
    phase = (qf[:, None, :] - kf[None, :, :]) * frequencies
    decay = phase.cos().mean(-1)
    return (2.0 - decay).clamp_min(1e-7).log()
