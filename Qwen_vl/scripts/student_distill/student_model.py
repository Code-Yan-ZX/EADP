"""Forward-only four-stream token scorer; no answer or teacher access."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class StudentScorer(nn.Module):
    """Identical architecture for all teacher targets.

    Visual and text inputs are first cast to fp32 and RMS-normalized over the
    last dimension.  Four log RMS values retain token-specific scale.  The
    question-dependent hidden bias is shared over the token sequence; the GELU
    nonlinearity allows it to interact with the token features.
    """

    def __init__(self, feature_dim: int = 4096, streams: int = 4,
                 hidden_dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.config = dict(feature_dim=feature_dim, streams=streams,
                           hidden_dim=hidden_dim, eps=eps)
        self.feature_dim, self.streams, self.eps = feature_dim, streams, eps
        self.visual = nn.Linear(streams * feature_dim, hidden_dim)
        self.scale = nn.Linear(streams, hidden_dim, bias=False)
        self.question = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or question.ndim != 2:
            raise ValueError("expected features [B,4,N,D] and question [B,D]")
        if (features.shape[0] != question.shape[0] or
            features.shape[1] != self.streams or
            features.shape[-1] != self.feature_dim or
            question.shape[-1] != self.feature_dim):
            raise ValueError("incompatible batch, stream count, or feature dimension")
        x = features.float()
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        x = (x / rms).permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        scales = rms.squeeze(-1).transpose(1, 2).log()
        q = question.float()
        q = q / torch.sqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)
        hidden = self.visual(x) + self.scale(scales) + self.question(q).unsqueeze(1)
        return self.output(F.gelu(hidden)).squeeze(-1)

