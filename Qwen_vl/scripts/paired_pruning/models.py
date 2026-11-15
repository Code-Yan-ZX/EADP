"""Forward-only, question-conditioned selectors used by both paired experiments.

The bag control has the same trainable parameters as the ordered model.  Only
the deterministic question positional encoding is disabled.  No target, answer,
bank mask, or teacher feature is accepted by ``forward``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PruningScorer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        defaults = dict(feature_dim=4096, streams=4, hidden_dim=64,
                        num_heads=4, query_layers=1, mode_heads=1,
                        max_question_tokens=256, ordered=True, dropout=0.0)
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"unknown model configuration: {sorted(unknown)}")
        defaults.update(config)
        self.config = defaults
        d, h = int(defaults["hidden_dim"]), int(defaults["num_heads"])
        if d < 4 or d % h or int(defaults["mode_heads"]) < 1:
            raise ValueError("hidden_dim must be divisible by num_heads; mode_heads >= 1")
        if int(defaults["query_layers"]) < 1 or int(defaults["max_question_tokens"]) < 1:
            raise ValueError("positive query_layers and max_question_tokens required")
        f, streams = int(defaults["feature_dim"]), int(defaults["streams"])
        self.visual = nn.Linear(streams * f, d)
        self.visual_scale = nn.Linear(streams, d, bias=False)
        self.question = nn.Linear(f, d)
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, h, dim_feedforward=2*d,
                dropout=float(defaults["dropout"]), activation="gelu",
                batch_first=True, norm_first=True),
            num_layers=int(defaults["query_layers"]), enable_nested_tensor=False)
        self.visual_norm = nn.LayerNorm(d)
        self.question_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d, h, dropout=float(defaults["dropout"]), batch_first=True)
        self.token_head = nn.Sequential(nn.LayerNorm(2*d), nn.Linear(2*d, d),
                                       nn.GELU(), nn.Linear(d, int(defaults["mode_heads"])))
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.GELU(),
                                  nn.Linear(d, int(defaults["mode_heads"])))
        positions = torch.arange(int(defaults["max_question_tokens"]), dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(positions.shape[0], d)
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies[:d//2])
        self.register_buffer("position_encoding", pe, persistent=False)

    def forward(self, features: torch.Tensor, question: torch.Tensor,
                question_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        if features.ndim != 4 or question.ndim != 3 or question_mask.ndim != 2:
            raise ValueError("expected features [B,S,N,D], question [B,M,D], mask [B,M]")
        b, streams, _, feature_dim = features.shape
        if (streams != cfg["streams"] or feature_dim != cfg["feature_dim"] or
            question.shape[0] != b or question.shape[-1] != feature_dim or
            tuple(question_mask.shape) != tuple(question.shape[:2])):
            raise ValueError("incompatible feature/question shapes")
        if question_mask.dtype != torch.bool or not bool(question_mask.any(-1).all()):
            raise ValueError("question_mask must be bool with at least one real token per question")
        if question.shape[1] > cfg["max_question_tokens"]:
            raise ValueError("question exceeds frozen max_question_tokens; no silent truncation")
        x = features.float()
        rms = (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        normalized = (x/rms).permute(0, 2, 1, 3).flatten(2)
        visual = self.visual(normalized) + self.visual_scale(rms.squeeze(-1).transpose(1, 2).log())
        # Mask before arithmetic, so unused padding values cannot affect results.
        q = question.float().masked_fill(~question_mask[..., None], 0.0)
        q = self.question(q / (q.square().mean(-1, keepdim=True) + 1e-6).sqrt())
        if cfg["ordered"]:
            # Compact positions support both right and left padding.
            pos = (question_mask.long().cumsum(-1) - 1).clamp_min(0)
            q = q + self.position_encoding[pos]
        q = self.query_encoder(q, src_key_padding_mask=~question_mask)
        q = self.question_norm(q)
        attended, _ = self.cross_attention(self.visual_norm(visual), q, q,
                                          key_padding_mask=~question_mask, need_weights=False)
        scores = self.token_head(torch.cat((visual, attended), -1)).transpose(1, 2)
        qmean = (q * question_mask[..., None]).sum(1) / question_mask.sum(1, keepdim=True)
        gate_logits = self.gate(torch.cat((visual.mean(1), qmean), -1))
        return {"scores": scores, "gate_logits": gate_logits}


def select_keep(output: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    """Select one complete mode using the gate, then stable exact-K/raster sort.

    This never averages mode scores and never evaluates an LLM candidate bank.
    """
    scores, gates = output["scores"], output["gate_logits"]
    if scores.ndim != 3 or tuple(gates.shape) != tuple(scores.shape[:2]):
        raise ValueError("invalid mode score/gate shapes")
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError("invalid K")
    if not bool(torch.isfinite(scores).all() and torch.isfinite(gates).all()):
        raise ValueError("nonfinite scores or gate logits")
    head = gates.argmax(-1)
    chosen = scores[torch.arange(scores.shape[0], device=scores.device), head]
    return torch.argsort(chosen, dim=-1, descending=True, stable=True)[:, :k].sort(-1).values


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Load a selected checkpoint and its provenance; inference has no labels."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("kind") != "selected":
        raise ValueError("expected a selected paired_pruning checkpoint")
    model = PruningScorer(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint["metadata"]
