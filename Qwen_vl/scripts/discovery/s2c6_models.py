"""
S2-C6 arms: one read-out, five ways of feeding it.

Every arm in this stage ends in the same read-out shape,

    s = w2 . GELU(a + b),        a in R^128,  w2: 128 -> 1,  b in R^128

and the arms differ only in what ``a`` is made of:

    L4          a = GELU(P h4)                          snapshot, the reference
    L2          a = GELU(P h2)                          earlier snapshot
    DELTA       a = GELU(P (h4 - h2))                   trajectory change alone
    L2+L4       a = [GELU(P2 h2) ; GELU(P4 h4)]         current state + history
    L4+DELTA    a = [GELU(P4 h4) ; GELU(Pd (h4-h2))]    current state + change
    L4+L4       a = [GELU(Pa h4) ; GELU(Pb h4)]         architecture control
    L4+QUERY    a = GELU(P h4) + B u,  u = Wq q         token + question

--------------------------------------------------------------------------
The parameter matching is exact, and it is the point
--------------------------------------------------------------------------
    2 x (4096 x 64 + 64)  =  4096 x 128 + 128  =  524 416
    plus b (128) and w2 (129)                  =  524 673   for every arm above
                                                             except L4+QUERY

So a dual-input arm cannot be read as "the two-layer one simply had more
parameters". There is deliberately no 8192 -> 128 arm in this stage: that would
double the projection and make any gain uninterpretable. The dual arms are the
only construction in which adding a second snapshot is free in parameter count.

``L4+L4`` is the control that makes the dual arms readable at all. It has the
same architecture and the same parameter count as ``L2+L4`` and ``L4+DELTA`` but
no second snapshot, so if it moves the number on its own the trajectory arms must
be read against it and not against ``L4``. It is not a deployable arm; it is the
null that says the dual architecture is worth nothing by itself.

``L4+QUERY`` is the one arm that is not capacity-matched: it adds 33 792
parameters (6.44 % of 524 673) against the 32 768 (6.25 %) S2-C5's ``GLOBAL-CTX``
added for its context path. That is the same structural role -- an additive
per-token modulation derived from a side channel -- at the same order of budget,
and the count is reported next to every number the arm produces rather than being
described as "small".

--------------------------------------------------------------------------
Why the query arm is initialised so that it starts at the L4 arm
--------------------------------------------------------------------------
The checkpoint rule selects on validation head_recall@8, which S2-C5A showed
peaks at epoch 0-2. An arm whose extra path starts as a large random
perturbation would therefore be scored while that perturbation is still noise,
and would lose to ``L4`` for a reason that has nothing to do with the query. So
``L4+QUERY`` builds P, b and w2 first -- in the same order and the same shapes as
``TokMLP``, so a reset seed gives them the same draw -- and then zero-initialises
the query expansion ``B``. At step 0 the arm's output is *identical* to the
``L4`` arm's, and the query path grows from zero. ``s2c6_train`` asserts that
identity on the untrained model rather than trusting it.

``B`` zero-initialised does not dead-lock ``Wq``: the first step moves ``B``
(the gradient into ``B`` does not involve ``Wq``), after which ``Wq`` receives
gradient through ``B``. One step of lag, and no wasted epoch.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2c5_models import LocalMLP as _S2C5LocalMLP                 # noqa: E402

D_MODEL = 4096
D_HID = 128
D_HALF = 64
R_QRY = 8


class TokMLP(_S2C5LocalMLP):
    """Single-channel token-local MLP -- exactly ``s2c5_models.LocalMLP``.

    Subclassing rather than reimplementing is deliberate: the L4 arm of S2-C6 is
    then the byte-identical module S2-C5 and S2-C5A trained, so the reference is
    the reference and not a reimplementation of it. Only the call signature is
    widened, to take the channel list every arm in this stage shares.
    """

    def forward(self, xs, q=None):
        return super().forward(xs[0])


class DualTokMLP(nn.Module):
    """Two channels at half width each, concatenated before the shared read-out.

        a = [GELU(P_lo x_lo) ; GELU(P_hi x_hi)]   in R^128
        s = w2 . GELU(a + b)

    Parameter-identical to ``TokMLP``. Which channels arrive as ``xs[0]`` and
    ``xs[1]`` is the caller's business, so the same class instantiates ``L2+L4``,
    ``L4+DELTA`` and the ``L4+L4`` architecture control -- and the control is
    therefore matched to the trajectory arms in every respect but its input.
    """

    kind = "dual"

    def __init__(self, d_in=D_MODEL, d_half=D_HALF):
        super().__init__()
        self.P_lo = nn.Linear(d_in, d_half)
        self.P_hi = nn.Linear(d_in, d_half)
        self.bias = nn.Parameter(torch.zeros(2 * d_half))
        self.w2 = nn.Linear(2 * d_half, 1)

    def forward(self, xs, q=None):
        a = torch.cat([F.gelu(self.P_lo(xs[0])),
                       F.gelu(self.P_hi(xs[1]))], dim=-1)
        return self.w2(F.gelu(a + self.bias)).squeeze(-1)

    def describe(self, lo="?", hi="?") -> dict:
        return {"access": f"token ({lo}) + token ({hi})", "context_dim": 0,
                "attention": False, "capacity_matched_to": "L4"}

    def _extra_repr(self):
        return f"d_half={self.P_lo.out_features}"


class QueryTokMLP(nn.Module):
    """Token-local MLP plus a low-rank additive query term.

        u = Wq q                 in R^8
        s = w2 . GELU(GELU(P h4) + B u + b)

    The query enters as an additive per-token modulation and nothing else: no
    cross-attention, no gating, no second round. Module construction order and
    shapes for P / b / w2 match ``TokMLP`` exactly, and ``B`` is zero-initialised,
    so at step 0 this arm computes the same function as ``TokMLP`` on the same
    seed (see the module docstring).
    """

    kind = "query"
    r = R_QRY

    def __init__(self, d_in=D_MODEL, d_hid=D_HID, r=R_QRY):
        super().__init__()
        self.r = r
        self.P = nn.Linear(d_in, d_hid)
        self.bias = nn.Parameter(torch.zeros(d_hid))
        self.w2 = nn.Linear(d_hid, 1)
        # built after the L4-equivalent modules so a reset seed reproduces their
        # draw; B zero so the arm starts exactly at the L4 arm's function
        self.Wq = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_hid, bias=False)
        nn.init.zeros_(self.B.weight)

    def query_term(self, q):
        return self.B(self.Wq(q))

    def query_contribution(self, xs, q):
        """Per-image query term, (B, 1, d_hid) -- the same vector for every token.

        The ``(B, 1, ...)`` shape is the point: it is broadcast across the token
        axis, so this arm cannot give two tokens different query conditioning.
        """
        return self.query_term(q).unsqueeze(1)

    def forward(self, xs, q=None):
        a = F.gelu(self.P(xs[0]))
        if q is not None:
            a = a + self.query_term(q).unsqueeze(1)
        return self.w2(F.gelu(a + self.bias)).squeeze(-1)

    def n_query_params(self) -> int:
        return int(sum(q.numel() for m in (self.Wq, self.B)
                       for q in m.parameters()))

    def n_token_params(self) -> int:
        return int(sum(p.numel() for m in (self.P, self.w2)
                       for p in m.parameters())
                   + int(self.bias.numel()))

    def describe(self) -> dict:
        return {"access": "token (L4) + question embedding (rank 8)",
                "context_dim": 0, "attention": False,
                "capacity_matched_to": None,
                "query_params": self.n_query_params()}


class QueryBilinear(nn.Module):
    """Token-local MLP plus a low-rank *token-dependent* query term.

        a_i = GELU(P h_i)
        s_i = w2 . GELU(a_i + b)  +  (Wv a_i) . (Wq q) / sqrt(r)

    Added budget: 4096 x 8 + 128 x 8 = 33 792 parameters -- *identical* to
    ``QueryTokMLP``'s, so the two are compared at the same cost.

    This arm exists because the additive one cannot answer the question it was
    built to answer. ``QueryTokMLP`` adds one 128-vector per image to every
    token, so its query contribution is the same for all 1 024 tokens: it can
    reorder tokens only through the nonlinearity of the final GELU, and it
    cannot express "this token matters for this question", which is what query
    conditioning means. It is reported, and its measured behaviour (the model
    learns a near-constant offset from it -- see the results doc) is the reason
    this arm was added after the fact. The bilinear term varies per token, so it
    can express token-specific query relevance, and it is the arm that actually
    tests whether the question carries information about *which* tokens matter.

    ``Wv`` is zero-initialised, so at step 0 the arm computes the same function
    as ``L4`` and the query path grows from nothing -- the same fairness
    property, and the same one-step lag on ``Wq``, as ``QueryTokMLP``.
    """

    kind = "query_bilinear"
    r = R_QRY

    def __init__(self, d_in=D_MODEL, d_hid=D_HID, r=R_QRY):
        super().__init__()
        self.r = r
        self.P = nn.Linear(d_in, d_hid)
        self.bias = nn.Parameter(torch.zeros(d_hid))
        self.w2 = nn.Linear(d_hid, 1)
        self.Wq = nn.Linear(d_in, r, bias=False)
        self.Wv = nn.Linear(d_hid, r, bias=False)
        nn.init.zeros_(self.Wv.weight)

    def _bilinear(self, a, q):
        return ((self.Wv(a) * self.Wq(q).unsqueeze(1)).sum(-1)) * self.r ** -0.5

    def query_contribution(self, xs, q):
        """Per-token query term, (B, N)."""
        return self._bilinear(F.gelu(self.P(xs[0])), q)

    def forward(self, xs, q=None):
        a = F.gelu(self.P(xs[0]))
        s = self.w2(F.gelu(a + self.bias)).squeeze(-1)
        if q is not None:
            s = s + self._bilinear(a, q)
        return s

    def n_query_params(self) -> int:
        return int(sum(p.numel() for m in (self.Wq, self.Wv)
                       for p in m.parameters()))

    def n_token_params(self) -> int:
        return int(sum(p.numel() for m in (self.P, self.w2)
                       for p in m.parameters()) + int(self.bias.numel()))

    def describe(self) -> dict:
        return {"access": "token (L4) + question embedding (rank 8, "
                          "token-dependent bilinear)",
                "context_dim": 0, "attention": False,
                "capacity_matched_to": None,
                "query_params": self.n_query_params()}


# ---------------------------------------------------------------------------
# The arm table. ``channels`` is read by s2c6_train to build the ViewSource;
# ``q`` says whether the arm consumes the cached query.
#
#   part   which stage section the arm belongs to
#   ctrl   whether the arm is a mechanism control rather than a candidate
# ---------------------------------------------------------------------------
ARMS = {
    "L4":        dict(cls=TokMLP,     channels=("h4",),          q=False,
                      part="B", ctrl=False, lo="L4", hi=None),
    "L2":        dict(cls=TokMLP,     channels=("h2",),          q=False,
                      part="B", ctrl=False, lo="L2", hi=None),
    "DELTA":     dict(cls=TokMLP,     channels=("delta",),       q=False,
                      part="B", ctrl=False, lo="delta", hi=None),
    "L2+L4":     dict(cls=DualTokMLP, channels=("h2", "h4"),     q=False,
                      part="B", ctrl=False, lo="L2", hi="L4"),
    "L4+DELTA":  dict(cls=DualTokMLP, channels=("h4", "delta"),  q=False,
                      part="B", ctrl=False, lo="L4", hi="delta"),
    "L4+L4":     dict(cls=DualTokMLP, channels=("h4", "h4"),     q=False,
                      part="B", ctrl=True,  lo="L4", hi="L4"),
    "L4+QUERY":  dict(cls=QueryTokMLP, channels=("h4",),         q=True,
                      part="C", ctrl=False, lo="L4", hi="query"),
    "L4+QRY-BILIN": dict(cls=QueryBilinear, channels=("h4",),    q=True,
                         part="C", ctrl=False, lo="L4", hi="query"),
}
# Post-hoc amendment to the pre-registration: the token-dependent query arm.
# The pre-registered L4+QUERY adds one 128-vector per image to every token, so
# it cannot express token-specific query relevance and its null does not close
# the question. L4+QRY-BILIN is budget-identical (33 792 added parameters) and
# token-dependent, and is reported as an amendment in the results doc.
POSTHOC_ARMS = ("L4+QRY-BILIN",)
# Part A reuses the L4 arm at four training-set sizes; the entry is shared so the
# n=240 point of Part A and the L4 reference of Part B are the same computation.
PART_A_ARM = "L4"
PART_A_TAG = "A{n}"

# Trajectory arms: the ones whose second channel is a *different* snapshot of the
# same image, i.e. the arms the WRONG-TRAJECTORY inference control applies to.
TRAJECTORY_ARMS = ("L2+L4", "L4+DELTA")
DUAL_ARMS = ("L2+L4", "L4+DELTA", "L4+L4")


def build(arm: str) -> nn.Module:
    return ARMS[arm]["cls"]()


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def describe(arm: str, model: nn.Module | None = None) -> dict:
    e = ARMS[arm]
    if e["cls"] is DualTokMLP:
        d = {"access": f"token ({e['lo']}) + token ({e['hi']})",
             "context_dim": 0, "attention": False,
             "capacity_matched_to": "L4 (exactly: 2 x (4096 x 64 + 64) = "
                                    "4096 x 128 + 128)"}
    elif e["cls"] is QueryTokMLP:
        d = model.describe() if model is not None else {}
        d["capacity_matched_to"] = None
    else:
        d = {"access": f"token ({e['lo']})", "context_dim": 0,
             "attention": False, "capacity_matched_to": "L4 (identity)"}
    d["role"] = "mechanism control" if e["ctrl"] else "candidate"
    d["part"] = e["part"]
    return d
