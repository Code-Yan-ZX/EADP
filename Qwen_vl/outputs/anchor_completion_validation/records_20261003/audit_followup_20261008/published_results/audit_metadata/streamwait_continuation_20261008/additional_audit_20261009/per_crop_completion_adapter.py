"""Uninstalled candidate correcting only NeXT Completion grouping.

The current frozen port passes all crops of one image to _completion_stream,
although the preregistration promises per-crop Completion. This factory can
wrap the original function for a separately registered comparison. Importing
this file does not patch, import, or install any production model module.

RTG (including its segment0 behavior), alpha/beta, selector, crop quotas,
lambda, features, keep indices and gather order remain the caller's settings.
The original Completion function performs all assignment/merge mathematics.
No GPU accuracy effect has been measured for this candidate.
"""
from __future__ import annotations

from collections.abc import Callable
import torch


def make_per_crop_completion_stream(
    original_completion_stream: Callable,
    *,
    crop_tokens: int = 576,
) -> Callable:
    """Return an adapter; the caller must explicitly install it in a new run.

    feat is [n_crops * crop_tokens, D]. keep contains unique global token
    indices. Like the original function, output follows ascending keep index,
    which is the frozen caller's boolean-mask gather order. Empty crops have
    no selected output, and all-kept crops use the original no-dropped path.

    The factory captures the original callable, including its frozen LAM;
    this adapter never assigns a lambda or changes any other model behavior.
    """
    if not isinstance(crop_tokens, int) or crop_tokens <= 0:
        raise ValueError('crop_tokens must be a positive integer')

    def per_crop_completion_stream(feat, keep, from_amp_common=None):
        if feat.ndim != 2 or keep.ndim != 1:
            raise ValueError('Expected feat[N,D] and keep[K]')
        n_tokens = int(feat.shape[0])
        if n_tokens % crop_tokens:
            raise ValueError('Flattened feature length must contain complete crops')
        if keep.dtype != torch.long:
            raise ValueError('Expected frozen caller long keep indices')
        if keep.numel() == 0:
            return feat[:0]
        if int(keep.min()) < 0 or int(keep.max()) >= n_tokens:
            raise ValueError('Keep index outside flattened features')
        if int(keep.unique().numel()) != int(keep.numel()):
            raise ValueError('Keep indices must be unique')

        parts = []
        for offset in range(0, n_tokens, crop_tokens):
            local_keep = keep[(keep >= offset) & (keep < offset + crop_tokens)] - offset
            if local_keep.numel() == 0:
                continue
            parts.append(original_completion_stream(
                feat[offset:offset + crop_tokens], local_keep, from_amp_common))
        return torch.cat(parts, dim=0)

    per_crop_completion_stream.__name__ = 'per_crop_completion_stream'
    return per_crop_completion_stream
