"""
M3-v2 -- MissGuard-v2: a query-conditioned, cross-token HEAD auditor.

Why a second version exists
---------------------------
M3-v0 fixed the *formulation* and refuted its *student*.  The correction

    S* = (S0 \\ E)  U  R,     R = top_r score_i over the 768 dropped tokens

has a measured ceiling of +14.87 macro at r=32 (74.72 vs B2's 59.88) while
keeping the visual-token budget at 256, but the v0 student recovered none of
it: its rescue sits at mean teacher rank 74-132, where the oracle's sits at
3.5-15.5, and every learned arm is matched by a content-free random rescue.
v0's student was

    score_i = f( v_i , handcrafted_i )          -- every token scored alone

and four extra hand features plus three seeds moved its head precision by
0.006 overlap points.  The feature set is not the bottleneck; the *structure*
is.  A token-local MLP has no way to express "this region matters because the
question asks about it" or "this region matters because the retained set is
already dense here", and the v0 listwise head objective collapsed twice.

What changes in v2, and what does not
-------------------------------------
Unchanged, deliberately:

* the base selector (B2 verbatim: EADP importance + `block8` coverage greedy
  at T=256) -- the base set is not touched;
* the budget (|S*| = 256 exactly, asserted);
* the eviction rule used by every arm (`maxred`);
* the pre-LLM, forward-only constraint.  v2 reads the vision tower's own
  post-merger output and the instruction-token **input embeddings**
  (`get_input_embeddings()(ids)` -- a lookup, not a decoder layer).  No L0/L1/…
  hidden state, no backward pass, no teacher score at inference.

Changed, one thing only:

* the auditor is a **query-conditioned, cross-token HeadAuditor** -- visual
  queries attend to the instruction tokens, candidates attend to learned
  summaries of the retained set, and the whole thing is trained directly on
  the teacher ranking's front end (Top-16 positives, ranks 17-128 as hard
  negatives) rather than on Top-64/Top-256 membership.

Ablation axis (brief §11), all at matched parameter scale:

    A  token-local        score_i = MLP([V_i ; hand_i ; g])
    B  query-conditioned  score_i = MLP([V_i ; hand_i ; q_i ; g])
    C  query + retained   score_i = MLP([V_i ; hand_i ; q_i ; c_i ; g])

`g` is the mean projected embedding of the dropped set -- one constant per
image, present in every arm including A, so the A->B contrast isolates query
conditioning and B->C isolates retained-set context.  It is an instance prior
(it shifts every candidate of an image identically), not a token-local term.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m3_common import (BASE_SELECTOR, BUDGET, D_VIS, FEATURES, GRID,  # noqa: E402
                       N_VIS, MissGuardPruner, Standardiser, handcrafted_t,
                       teacher_topk)

# The v1 bank (22 features) is the input bank: X, vis, s0, g2 are reused
# verbatim, so the base sets and the teacher labels are byte-identical to the
# ones every M3-v0 number was measured against.
VIS_BANK = "m3_bank_v1"
TXT_BANK = "m3v2_text"

# Head definition of the training target (brief §4).  Ranks are positions in
# the DROPPED set's own teacher ranking, 0 = the teacher's best dropped token.
POS_RANK = 16               # positives: teacher ranks 0..15
NEG_RANK = 128              # hard negatives: teacher ranks 16..127
N_NEG_SAMPLE = 48           # negatives drawn per image per step (of 112)


# ===========================================================================
# the model
# ===========================================================================
class CrossBlock(nn.Module):
    """One multi-head cross-attention block with a residual connection.

    `q` (B, Lq, d) attends to `kv` (B, Lk, d).  Pre-norm, so a stack of these
    is stable without warm-up tricks.
    """

    def __init__(self, d: int, n_heads: int, p_drop: float):
        super().__init__()
        assert d % n_heads == 0, (d, n_heads)
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.drop = nn.Dropout(p_drop)

    def forward(self, q, kv, kv_mask=None):
        B, Lq, d = q.shape
        Lk = kv.shape[1]
        qq = self.q(self.ln_q(q)).view(B, Lq, self.h, self.dh).transpose(1, 2)
        kk = self.k(self.ln_kv(kv)).view(B, Lk, self.h, self.dh).transpose(1, 2)
        vv = self.v(self.ln_kv(kv)).view(B, Lk, self.h, self.dh).transpose(1, 2)
        a = (qq @ kk.transpose(-1, -2)) / math.sqrt(self.dh)
        if kv_mask is not None:
            # A fully-masked row would produce NaN; the instruction sequences
            # always carry at least one real token, and `prep` asserts it.
            a = a.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        a = a.softmax(dim=-1)
        z = (a @ vv).transpose(1, 2).reshape(B, Lq, d)
        return q + self.drop(self.o(z))


class HeadAuditor(nn.Module):
    """Query-conditioned, cross-token auditor over the 1024 visual tokens.

    Inputs are ALREADY STANDARDISED (`prep_inputs`); the standardisers are
    fitted on fit rows only and stored in the checkpoint, so the live pruner
    and the trainer apply the same numbers from the same function.

    forward(vis, txt, X, s0, txt_mask) -> (B, N) scores, one per visual token.
    Only the dropped positions are ever read, but every token is scored because
    the retained-set summary and the dropped-set prior both need the full map.
    """

    def __init__(self, d_hand: int, d_vis: int = D_VIS, d: int = 128,
                 n_heads: int = 4, n_slots: int = 8, d_h: int = 64,
                 use_query: bool = True, use_retained: bool = True,
                 p_drop: float = 0.1):
        super().__init__()
        self.use_query, self.use_retained = bool(use_query), bool(use_retained)
        self.d, self.d_h, self.n_slots = d, d_h, n_slots
        self.d_hand, self.d_vis = d_hand, d_vis
        self.n_heads = n_heads

        self.proj_v = nn.Linear(d_vis, d)
        self.proj_h = nn.Linear(d_hand, d_h)
        if self.use_query:
            self.proj_t = nn.Linear(d_vis, d)
            self.xt = CrossBlock(d, n_heads, p_drop)
        if self.use_retained:
            self.slots = nn.Parameter(torch.randn(n_slots, d) * 0.02)
            self.slot_attn = CrossBlock(d, n_heads, p_drop)   # slots <- S0
            self.ctx_attn = CrossBlock(d, n_heads, p_drop)    # candidates <- slots

        din = d + d_h + (d if self.use_query else 0) \
            + (d if self.use_retained else 0) + d             # + dropped prior
        self.head = nn.Sequential(
            nn.LayerNorm(din), nn.Linear(din, d), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(d, d_h), nn.GELU(), nn.Linear(d_h, 1))
        self.din = din

    def blocks(self, vis, txt, X, s0, txt_mask=None) -> dict:
        """The head's input blocks, before the concatenation.

        Split out so `m3v2_probe.py` can zero one block at a time and measure
        how much of the deployed ranking actually depends on it -- the
        difference between "the structure did not help" and "the structure was
        never used".
        """
        B, N, _ = vis.shape
        Vh = self.proj_v(vis)
        out = dict(V=Vh, H=F.gelu(self.proj_h(X)))
        if self.use_query:
            out["q"] = self.xt(Vh, self.proj_t(txt), txt_mask)
        if self.use_retained:
            d = Vh.shape[-1]
            s0v = Vh.gather(1, s0[:, :, None].expand(-1, -1, d))
            slots = self.slot_attn(self.slots.unsqueeze(0).expand(B, -1, -1), s0v)
            out["c"] = self.ctx_attn(Vh, slots)
        # Dropped-set prior: the mean projected embedding of the 768 tokens the
        # base selector rejected.  One constant per image, broadcast to every
        # candidate -- it lets the head read "what kind of competition is this"
        # without any 768x768 attention.
        keep = torch.zeros(B, N, dtype=torch.bool, device=vis.device)
        keep.scatter_(1, s0, True)
        w = (~keep).unsqueeze(-1).to(Vh.dtype)
        g = (Vh * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1.0)
        out["g"] = g.expand(-1, N, -1)
        return out

    @property
    def block_order(self) -> tuple:
        o = ["V", "H"]
        if self.use_query:
            o.append("q")
        if self.use_retained:
            o.append("c")
        o.append("g")
        return tuple(o)

    def head_from(self, parts: dict, zero: str = None):
        """Head applied to the blocks; `zero` replaces one block with zeros."""
        cat = [torch.zeros_like(parts[k]) if k == zero else parts[k]
               for k in self.block_order]
        return self.head(torch.cat(cat, dim=-1)).squeeze(-1)

    def forward(self, vis, txt, X, s0, txt_mask=None):
        return self.head_from(self.blocks(vis, txt, X, s0, txt_mask))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return dict(d_hand=self.d_hand, d_vis=self.d_vis, d=self.d,
                    n_heads=self.n_heads, n_slots=self.n_slots, d_h=self.d_h,
                    use_query=self.use_query, use_retained=self.use_retained)


def standardise(x, mu, sd):
    """The one standardisation, used by the trainer and the live pruner alike.

    Kept as a separate two-line function so the trainer can apply it in chunks
    (the fp32 (n, 1024, 4096) result is 5 GB) while the pruner applies it to a
    single image -- both call this, so the arithmetic cannot drift.
    """
    return (x.float() - mu) / sd


def prep_inputs(vis_raw, txt_raw, X_raw, st):
    """THE standardisation, shared by the trainer and the live pruner.

    `st` maps to (mu, sd) tensors for `vis`, `txt` and `hand`.  The vision and
    text statistics are fitted on the fit rows' raw fp16-stored tensors, the
    hand statistics on the fit rows' dropped tokens -- the same fit-only rule
    v0 used, so no val or test row contributes a scale.
    """
    return (standardise(vis_raw, st["mu_vis"], st["sd_vis"]),
            standardise(txt_raw, st["mu_txt"], st["sd_txt"]),
            standardise(X_raw, st["mu_hand"], st["sd_hand"]))


def build_auditor(ck: dict, device=None) -> HeadAuditor:
    m = HeadAuditor(ck["d_hand"], ck["d_vis"], d=ck["d"], n_heads=ck["n_heads"],
                    n_slots=ck["n_slots"], d_h=ck["d_h"],
                    use_query=ck["use_query"], use_retained=ck["use_retained"],
                    p_drop=0.0)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m if device is None else m.to(device)


# ===========================================================================
# proxy metrics -- the head, measured directly (brief §6)
# ===========================================================================
# `hw@16` is the head-weighted rank metric: exp(-rank/8) averaged over the 16
# tokens the auditor would rescue.  1.0 means it found the teacher's own head
# in order; a rescue at mean rank 74 (v0's regime) scores ~0.
HW_TAU = 8.0


def instance_head_metrics(g2, drop, scores, n_rank=16):
    """Head metrics for ONE instance.  `scores` aligns with `drop`."""
    t = np.asarray(g2)[np.asarray(drop)]
    n = t.size
    order = np.argsort(-t, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    sel = np.argsort(-np.asarray(scores), kind="stable")

    m = {}
    for r in (4, 8, 16, 32):
        pick = sel[:r]
        m[f"top{r}_recall@{r}"] = float((rank[pick] < r).mean())
    m["top4_recall@8"] = float((rank[sel[:8]] < 4).mean() * 8 / 4)
    m["top8_recall@16"] = float((rank[sel[:16]] < 8).mean() * 16 / 8)
    m["top16_recall@16"] = float((rank[sel[:16]] < 16).mean())
    m["top16_recall@32"] = float((rank[sel[:32]] < 16).mean() * 32 / 16)

    r16, r8 = rank[sel[:16]], rank[sel[:8]]
    m["mean_rank@8"] = float(r8.mean())
    m["mean_rank@16"] = float(r16.mean())
    m["median_rank@16"] = float(np.median(r16))
    m["frac_in_top8@16"] = float((r16 < 8).mean())
    m["frac_in_top16@16"] = float((r16 < 16).mean())
    m["hw@16"] = float(np.exp(-r16 / HW_TAU).mean())
    m["hw@8"] = float(np.exp(-r8 / HW_TAU).mean())

    # NDCG@16 with graded relevance 1/(rank+1) over the dropped ranking
    rel = 1.0 / (rank + 1.0)
    disc = 1.0 / np.log2(np.arange(2, 18))
    dcg = float((rel[sel[:16]] * disc).sum())
    ideal = float((np.sort(rel)[::-1][:16] * disc).sum())
    m["ndcg@16"] = dcg / ideal if ideal > 0 else 0.0
    m["chance_top16_recall@16"] = 16.0 / n
    return m


def aggregate(rows_metrics):
    """Mean of a list of per-instance metric dicts."""
    keys = [k for k in rows_metrics[0] if k != "chance_top16_recall@16"]
    out = {k: float(np.mean([m[k] for m in rows_metrics])) for k in keys}
    out["chance_top16_recall@16"] = float(np.mean(
        [m["chance_top16_recall@16"] for m in rows_metrics]))
    return out


def selection_key(m: dict) -> tuple:
    """Checkpoint selection: primary = Top-16 recall@16 (the gate metric),
    tie-broken within 0.005 by the head-weighted rank metric.  Both are
    recorded, neither is a token AUC -- the deployment metric is the head."""
    return (round(m["top16_recall@16"], 3), m["hw@16"])


# ===========================================================================
# the bank: v1's cached pre-LLM state + v2's instruction-token embeddings
# ===========================================================================
class BankV2:
    """450 frozen instances.  X/vis/s0/g2 come from `m3_bank_v1.npz` unchanged;
    `txt` is the instruction-token embedding sequence from `m3v2_text.npz`.

    The two files are keyed identically and the keys are asserted equal, so a
    row can never be paired with another instance's question.
    """

    def __init__(self, vis_tag: str = VIS_BANK, txt_tag: str = TXT_BANK):
        z = np.load(os.path.join(OUTPUT_DIR, f"{vis_tag}.npz"),
                    allow_pickle=False)
        self.key = [str(k) for k in z["key"]]
        self.ds = np.array([str(s) for s in z["ds"]])
        self.split = np.array([str(s) for s in z["split"]])
        self.X = z["X"].astype(np.float32)
        self.s0 = z["s0"].astype(np.int64)
        self.g2 = z["g2"].astype(np.float32)
        self.vis = z["vis"]                                  # (N,1024,4096) fp16
        self.features = ([str(x) for x in z["feature_names"]]
                         if "feature_names" in z.files
                         else list(FEATURES)[:self.X.shape[2]])
        assert self.features == list(FEATURES)[:len(self.features)], \
            "bank feature columns are not a prefix of FEATURES"

        t = np.load(os.path.join(OUTPUT_DIR, f"{txt_tag}.npz"),
                    allow_pickle=False)
        tkey = [str(k) for k in t["key"]]
        assert tkey == self.key, "text bank and vision bank are not row-aligned"
        self.txt = t["txt"]                                  # (N,L,4096) fp16
        self.txt_len = t["txt_len"].astype(np.int64)
        assert self.txt.shape[0] == len(self.key)
        assert int(self.txt_len.max()) <= self.txt.shape[1]

        self.n_vis = self.X.shape[1]
        keep = np.zeros((len(self.key), self.n_vis), dtype=bool)
        np.put_along_axis(keep, self.s0, True, axis=1)
        self.keep = keep
        self.drop = [np.where(~keep[i])[0] for i in range(len(self.key))]
        # Dropped-set teacher ranking, precomputed: `rank[i][j]` is the teacher
        # rank of dropped token j (0 = the teacher's own best dropped token).
        self.rank = []
        for i in range(len(self.key)):
            t_ = self.g2[i][self.drop[i]]
            o = np.argsort(-t_, kind="stable")
            rk = np.empty(t_.size, dtype=np.int64)
            rk[o] = np.arange(t_.size)
            self.rank.append(rk)

    def rows(self, split):
        return np.where(self.split == split)[0]

    def head_pools(self, i):
        """(positive positions, hard-negative positions, positive ranks)."""
        rk = self.rank[i]
        pos = np.where(rk < POS_RANK)[0]
        neg = np.where((rk >= POS_RANK) & (rk < NEG_RANK))[0]
        return pos, neg, rk[pos]


def fit_stats(bank: BankV2, fit_rows, chunk: int = 32):
    """Standardiser statistics over FIT rows only, for vis / txt / hand."""
    def acc(arr, rows):
        s = np.zeros(arr.shape[-1], dtype=np.float64)
        sq = np.zeros(arr.shape[-1], dtype=np.float64)
        n = 0
        for a in range(0, len(rows), chunk):
            blk = arr[rows[a:a + chunk]].astype(np.float32)
            flat = blk.reshape(-1, blk.shape[-1])
            s += flat.sum(0, dtype=np.float64)
            sq += np.square(flat, dtype=np.float64).sum(0)
            n += flat.shape[0]
        mu = s / n
        var = np.maximum(sq / n - mu * mu, 1e-12)
        return mu.astype(np.float32), np.sqrt(var).astype(np.float32)

    mu_v, sd_v = acc(bank.vis, fit_rows)
    # text: only the REAL tokens of each sequence take part in the statistics;
    # padding is masked out of the attention and must not move the scale.
    L = bank.txt.shape[1]
    tv = bank.txt[fit_rows].astype(np.float32)
    m = (np.arange(L)[None, :] < bank.txt_len[fit_rows][:, None])
    flat = tv[m]
    mu_t = flat.mean(0).astype(np.float32)
    sd_t = flat.std(0).astype(np.float32)
    sd_t[sd_t < 1e-6] = 1.0
    sd_v[sd_v < 1e-6] = 1.0

    hand = np.concatenate([bank.X[i][bank.drop[i]] for i in fit_rows])
    std_h = Standardiser.fit(hand)
    return dict(mu_vis=mu_v, sd_vis=sd_v, mu_txt=mu_t, sd_txt=sd_t,
                mu_hand=std_h.mu, sd_hand=std_h.sd)


# ===========================================================================
# the pruner
# ===========================================================================
class HeadAuditorPruner(MissGuardPruner):
    """`MissGuardPruner` with the learned branch replaced by the HeadAuditor.

    Only `audit_scores` is overridden; the base class still owns the base
    selection, the identity path, the eviction, the |S*| = 256 assertion and
    the timing events.  With `source != "learned"` this class is byte-identical
    to the v0 pruner.
    """

    def audit_scores(self, g, vis_raw, vis_q, s0, dropped, text_seq=None):
        miss = self.miss
        dev = vis_raw.device
        assert text_seq is not None, "the v2 auditor needs the instruction embeddings"
        st = {k: miss[k] for k in ("mu_vis", "sd_vis", "mu_txt", "sd_txt",
                                   "mu_hand", "sd_hand")}
        Xh = handcrafted_t(g, vis_raw, s0)
        idx = miss.get("feature_idx")
        if idx is not None:
            Xh = Xh[:, idx]
        # `text_embeds_seq_llm` is (num_images, L, D) with num_images == 1 in
        # the prellm path; the text-side bank stored exactly this tensor, so the
        # live sequence is the training sequence token for token.
        txt = text_seq.reshape(-1, text_seq.shape[-1])
        txt_mask = torch.ones(1, txt.shape[0], dtype=torch.bool, device=dev)
        # Both banks store fp16, and the standardisers were fitted on those
        # fp16 values.  Feeding the live fp32 tensors here would be a silent
        # ~1e-3 train/serve skew -- the same trap v0 documented for `vis_q`,
        # now on both inputs.
        v, t, x = prep_inputs(vis_q[None], txt.half().float()[None],
                              Xh[None], st)
        sc = miss["student"](v, t, x, s0[None].to(dev), txt_mask)
        return sc.reshape(-1)[dropped]


def install_auditor(eng, model, miss: dict, selector: str = BASE_SELECTOR):
    """Install the v2 pruner on BOTH hooks (same guard as `install_missguard`)."""
    old = eng.pruner
    mg = HeadAuditorPruner(
        visual_token_num=old.visual_token_num, alpha=old.alpha, beta=old.beta,
        visual_dim=old.visual_dim, spatial_merge_size=old.spatial_merge_size,
        selector=selector, capture=False, miss=miss,
    ).to(next(model.model.parameters()).device)
    mg.eval()
    mg.sim_mode = "rebound"
    eng.pruner = mg
    model.pruner = mg
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return eng
