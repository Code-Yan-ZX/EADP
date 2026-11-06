"""PixelPrune anchored max recurrence specialized to threshold zero.

The caller must ensure finite inputs and that a zero result of the author's
subtraction/reduction implies equal input vectors on the execution device.
That numerical precondition is not assumed for arbitrary GPU inputs.
"""
import torch


def select(tokens, height, width):
    if tokens.ndim != 2 or height < 1 or width < 1 or tokens.shape[0] != height * width:
        raise ValueError('invalid grid')
    image = tokens.reshape(height, width, -1)
    rows = torch.arange(height, device=tokens.device)[:, None]
    cols = torch.arange(width, device=tokens.device)[None, :]
    left = image[rows, (cols - 1).clamp(min=0)]
    up = image[(rows - 1).clamp(min=0), cols]
    upper_left = image[(rows - 1).clamp(min=0), (cols - 1).clamp(min=0)]

    def similar(a, b):
        return (a - b).abs().amax(dim=-1) <= 0

    a = torch.where((cols > 0).unsqueeze(-1), left, up)
    b = torch.where((rows > 0).unsqueeze(-1), up, a)
    c = torch.where(((rows > 0) & (cols > 0)).unsqueeze(-1), upper_left, b)
    use_b = (rows > 0) & (cols > 0) & similar(c, a) & ~similar(c, b)
    predicted = torch.where(use_b.unsqueeze(-1), b, a)
    keep = ~similar(image, predicted)
    keep[0, 0] = True
    return keep.flatten().nonzero(as_tuple=False)[:, 0]
