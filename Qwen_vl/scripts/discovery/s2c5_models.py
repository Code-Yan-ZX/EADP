"""
S2-C5 scorer ladder: four capacities to *access information*, one trunk.

The design constraint that makes this ladder interpretable is that the arms must
not differ in capacity. So every arm shares one trunk

    a_i = GELU(P h_i)                P: (4096, d_hid),  d_hid = 128

and the arms differ only in which extra information reaches the read-out:

    LOCAL-MLP     s_i = w2 . GELU(a_i + b)                    token alone
    GLOBAL-CTX    s_i = w2 . GELU(a_i + V c + b),  c = [mean_j a_j ; max_j a_j]
                                                              + image-global,
                                                                permutation-
                                                                invariant
    SET-CTX       s_i = w2 . GELU(a_i + attn_i),   attn_i = sum_j alpha_ij Vv a_j
                                                              + token-token
                                                                (one head, one
                                                                 round, no deep
                                                                 Transformer)

The trunk is 4096 x 128 = 524 416 parameters, which is the scale of the S2-C1 QRY
arm (528 385). The additions are V (128 x 256 = 32 768, +6.2 %) for GLOBAL-CTX and
Q/K/Vv (128 x 32 twice + 128 x 128 = 24 576, +4.7 %) for SET-CTX, so no arm can be
read as "the contextual one simply had more parameters". A fifth arm,

    LOCAL-MLP-WIDE  s_i = w2 . GELU(GELU(P2 h_i)),  P2: (4096, 512)

is 2.16 M parameters and is deliberately *not* capacity-matched. It is not a
deployable arm: it is the ceiling of the token-local hypothesis. If even a 4x
wider token-local MLP fails to beat the linear reference, then "the head signal
is a nonlinear function of the token alone" is dead regardless of how the matched
arm happens to land.

--------------------------------------------------------------------------
Where each arm is allowed to look
--------------------------------------------------------------------------
No arm sees the query. No arm sees any other image at inference except through
its own token set. This is a mechanism diagnosis, so the only degrees of freedom
allowed to differ are the ones named above.

``context`` and ``attn`` are computed by separate methods and passed in
explicitly rather than being fused into ``forward``. That is what makes the
wrong-image control a one-line intervention: the caller computes the context from
a *different* image's tokens and feeds it to this image's queries. A scorer whose
gain survives that substitution was never using sample-specific configuration.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D_MODEL = 4096
D_HID = 128                 # matched trunk width
D_WIDE = 512                # LOCAL-MLP-WIDE trunk width
R_ATT = 32                  # SET-CTX attention rank

ARMS = ("LOCAL-MLP", "GLOBAL-CTX", "SET-CTX", "LOCAL-MLP-WIDE")
CONTEXTUAL = ("GLOBAL-CTX", "SET-CTX")


# ---------------------------------------------------------------------------
class Trunk(nn.Module):
    """a_i = GELU(P h_i); the one module every arm shares by construction."""

    def __init__(self, d_in=D_MODEL, d_hid=D_HID):
        super().__init__()
        self.P = nn.Linear(d_in, d_hid)

    def forward(self, h):
        return F.gelu(self.P(h))


class LocalMLP(nn.Module):
    """score_i = w2 . GELU(a_i + b) -- the token in isolation, nothing else."""

    kind = "local"

    def __init__(self, d_in=D_MODEL, d_hid=D_HID):
        super().__init__()
        self.trunk = Trunk(d_in, d_hid)
        self.bias = nn.Parameter(torch.zeros(d_hid))
        self.w2 = nn.Linear(d_hid, 1)

    def forward(self, h, ctx=None):
        a = self.trunk(h)
        return self.w2(F.gelu(a + self.bias)).squeeze(-1)

    def describe(self) -> dict:
        return {"access": "token only", "context_dim": 0, "attention": False}


class LocalMLPWide(nn.Module):
    """Two hidden layers at width 512 -- the capacity ceiling of the local family.

    Not capacity-matched to anything; reported as an upper bound on what a
    token-local nonlinear function can express, never as a deployable arm.
    """

    kind = "local_wide"

    def __init__(self, d_in=D_MODEL, d_hid=D_WIDE):
        super().__init__()
        self.trunk = Trunk(d_in, d_hid)
        self.mid = nn.Linear(d_hid, d_hid)
        self.w2 = nn.Linear(d_hid, 1)

    def forward(self, h, ctx=None):
        a = self.trunk(h)
        return self.w2(F.gelu(self.mid(F.gelu(a)))).squeeze(-1)

    def describe(self) -> dict:
        return {"access": "token only (4x width)", "context_dim": 0,
                "attention": False, "capacity_matched": False}


class GlobalCtx(nn.Module):
    """score_i = w2 . GELU(a_i + V c + b) with c a permutation-invariant summary.

    c = [mean_j a_j ; max_j a_j] is a DeepSets-style read-out: a learned per-token
    phi (here the shared trunk) followed by symmetric pooling. Both poolings are
    exactly permutation-invariant, so this arm cannot represent any pairwise
    relation between tokens -- it sees only the image-level aggregate.
    """

    kind = "global"

    def __init__(self, d_in=D_MODEL, d_hid=D_HID):
        super().__init__()
        self.trunk = Trunk(d_in, d_hid)
        self.V = nn.Linear(2 * d_hid, d_hid, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_hid))
        self.w2 = nn.Linear(d_hid, 1)

    @staticmethod
    def pool(a):
        return torch.cat([a.mean(dim=1), a.amax(dim=1)], dim=-1)

    def forward(self, h, ctx=None):
        a = self.trunk(h)
        if ctx is None:
            ctx = self.pool(a)                  # the image's own context
        # ctx is per-image (B, 2*d_hid); broadcast it across the token axis
        return self.w2(F.gelu(a + self.V(ctx).unsqueeze(1) + self.bias)).squeeze(-1)

    def context(self, h):
        return self.pool(self.trunk(h))

    def describe(self) -> dict:
        return {"access": "token + permutation-invariant image pool "
                           "(mean, max)", "context_dim": 2 * D_HID,
                "attention": False}


class SetCtx(nn.Module):
    """score_i = w2 . GELU(a_i + attn_i), one head, one round.

    attn_i = sum_j softmax_j(q_i . k_j / sqrt(r)) Vv a_j with q = Q a, k = K a.
    This is the smallest module that can express a genuine pairwise interaction:
    every token's read-out depends on which other tokens are present, and on
    their configuration, not merely on the image's aggregate. There is no depth,
    no layer norm and no feed-forward block -- it is deliberately below the
    threshold at which "we trained a Transformer" would be a fair description.
    Cost is one 1024 x 1024 score matrix per image, 32-dimensional, so the added
    FLOPs are ~5 % of the trunk's.
    """

    kind = "set"

    def __init__(self, d_in=D_MODEL, d_hid=D_HID, r=R_ATT):
        super().__init__()
        self.r = r
        self.trunk = Trunk(d_in, d_hid)
        self.Q = nn.Linear(d_hid, r, bias=False)
        self.K = nn.Linear(d_hid, r, bias=False)
        self.Vv = nn.Linear(d_hid, d_hid, bias=False)
        self.w2 = nn.Linear(d_hid, 1)

    def attend(self, a_q, a_kv):
        """Interaction term for queries ``a_q`` against key/value set ``a_kv``."""
        q = self.Q(a_q)
        k = self.K(a_kv)
        att = torch.softmax(q @ k.transpose(-1, -2) * self.r ** -0.5, dim=-1)
        return att @ self.Vv(a_kv)

    def forward(self, h, kv=None):
        a = self.trunk(h)
        a_kv = a if kv is None else kv
        return self.w2(F.gelu(a + self.attend(a, a_kv))).squeeze(-1)

    def forward_pair(self, h_q, h_kv):
        """Score ``h_q``'s tokens reading the key/value set of ``h_kv``."""
        return self.forward(h_q, kv=self.trunk(h_kv))

    def describe(self) -> dict:
        return {"access": "token + full token set (1-head, 1-round attention)",
                "context_dim": None, "attention": True, "rank": self.r}


FACTORY = {
    "LOCAL-MLP": LocalMLP,
    "GLOBAL-CTX": GlobalCtx,
    "SET-CTX": SetCtx,
    "LOCAL-MLP-WIDE": LocalMLPWide,
}


def build(arm: str) -> nn.Module:
    return FACTORY[arm]()


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------------------
CONTROL_NOTE = {
    "wrong_image_context": (
        "held-out inference with this image's context/interaction set replaced by "
        "another randomly chosen held-out image's. If the contextual gain is "
        "sample-specific it must collapse; if the gain survives, the arm was "
        "exploiting image-level statistics that are shared across images rather "
        "than this image's token configuration."),
}
