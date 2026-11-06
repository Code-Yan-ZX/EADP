"""Experimental antidiagonal implementation of PixelPrune's anchored max recurrence.

This preserves the recurrence's left, up and upper-left dependencies while
processing independent grid cells on the same antidiagonal together. It is not
author code, is not integrated into a model, and has no GPU entry point here.
"""
import torch


def select(tokens, height, width, threshold):
    if tokens.ndim != 2 or height < 1 or width < 1 or tokens.shape[0] != height * width:
        raise ValueError('invalid grid')
    if not 0 < threshold < float('inf'):
        raise ValueError('only positive finite max thresholds are studied')
    image = tokens.reshape(height, width, -1)
    anchors = image.clone()
    keep = torch.zeros((height, width), dtype=torch.bool, device=tokens.device)
    keep[0, 0] = True

    def similar(x, y):
        return (x - y).abs().amax(dim=-1) <= threshold

    for diagonal in range(1, height + width - 1):
        rows = torch.arange(max(0, diagonal - width + 1), min(height - 1, diagonal) + 1, device=tokens.device)
        cols = diagonal - rows
        left = anchors[rows, (cols - 1).clamp(min=0)]
        up = anchors[(rows - 1).clamp(min=0), cols]
        upper_left = anchors[(rows - 1).clamp(min=0), (cols - 1).clamp(min=0)]
        a = torch.where((cols > 0).unsqueeze(-1), left, up)
        b = torch.where((rows > 0).unsqueeze(-1), up, a)
        c = torch.where(((rows > 0) & (cols > 0)).unsqueeze(-1), upper_left, b)
        use_b = (rows > 0) & (cols > 0) & similar(c, a) & ~similar(c, b)
        prediction = torch.where(use_b.unsqueeze(-1), b, a)
        current = image[rows, cols]
        retained = ~similar(current, prediction)
        anchors[rows, cols] = torch.where(retained.unsqueeze(-1), current, prediction)
        keep[rows, cols] = retained
    return keep.flatten().nonzero(as_tuple=False)[:, 0]
