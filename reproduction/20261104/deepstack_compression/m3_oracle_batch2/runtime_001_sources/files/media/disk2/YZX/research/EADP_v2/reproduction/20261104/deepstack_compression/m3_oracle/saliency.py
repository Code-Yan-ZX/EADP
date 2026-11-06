"""M3's fixed input-gradient utility, with no model or label access."""
import torch

def utilities(features, gradients):
    if len(features) != 4 or len(gradients) != 4:
        raise ValueError('main and three DeepStack streams required')
    out = []
    for feature, gradient in zip(features, gradients):
        if (feature.ndim != 2 or feature.shape[0] != 1024 or feature.shape != gradient.shape
                or feature.device != gradient.device or not feature.is_floating_point() or not gradient.is_floating_point()):
            raise ValueError('invalid full-image feature or gradient')
        with torch.no_grad():
            score = (feature.float() * gradient.float()).abs().mean(dim=-1)
            if not torch.isfinite(score).all():
                raise ValueError('nonfinite utility')
            out.append(score.detach().cpu().tolist())
    return out
