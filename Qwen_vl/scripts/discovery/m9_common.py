"""
M9 -- shared boundary construction and prompt assembly.

Conventions (declared):
    greedy order     m6_eapd_order.npz row = the incumbent's block8 facility
                     greedy order over all 1024 tokens; S0 = order[:256]
                     (gated: sorted == bank s0).
    tail_r           order[256-r:256] -- the last r tokens the incumbent's own
                     greedy marginal gain picked. NEVER pre-deleted.
    reserve P        the 32 dropped tokens with the LOWEST cos_s0c (the M6/M8
                     orientation; identical to M8's pool P1 by construction).
    boundary_r       tail_r u P, |.| = 32 + r; adjudication keeps exactly r of
                     them, so |S_final| = 256 always. tail_16 u P (48 tokens)
                     is captured once and covers r = 4/8/16.
    S_early          sorted(S0 u P) = 288 visual tokens through the first L
                     layers. Position policy: RENUMBER (arange) -- the
                     incumbent prellm path's own implicit convention; like-
                     for-like with the B2 baseline and with the D-B2 control.
    query rows       every assembled-sequence text row AFTER the visual span
                     (<|vision_end|>, question, <|im_end|>, assistant
                     scaffolding). Question span = rows between <|vision_end|)
                     and the first <|im_end|> after the visual span; the exact
                     A1/A2/A3 aggregations are applied offline.
"""
from __future__ import annotations

import os

import numpy as np
import torch

sys_path = os.path.dirname(os.path.abspath(__file__))

POOL = 32
TAIL_MAX = 16


def load_banks():
    d = os.path.join(os.path.dirname(os.path.dirname(sys_path)),
                     "outputs", "discovery")
    bank = np.load(os.path.join(d, "m5_bank.npz"), allow_pickle=False)
    ordr = np.load(os.path.join(d, "m6_eapd_order.npz"), allow_pickle=False)
    assert (bank["key"] == ordr["key"]).all()
    assert (np.sort(ordr["order"][:, :256], axis=1)
            == np.sort(bank["s0"], axis=1)).all(), "G-ORDER failed"
    fi = int(np.where(bank["feature_names"] == "cos_s0c")[0][0])
    return dict(key=bank["key"], ds=bank["ds"], split=bank["split"],
                idx=bank["idx"], s0=bank["s0"].astype(np.int64),
                g2=bank["g2"].astype(np.float64),
                cos=bank["X"][:, :, fi].astype(np.float64),
                order=ordr["order"].astype(np.int64))


def instance_boundary(banks, row: int, pool: int = POOL):
    """All r-independent quantities for one instance."""
    order = banks["order"][row]
    cos = banks["cos"][row]
    s0 = order[:256]
    in_s0 = np.zeros(1024, dtype=bool)
    in_s0[s0] = True
    dropped = np.flatnonzero(~in_s0)
    reserve = dropped[np.argsort(cos[dropped], kind="stable")[:pool]]
    tail = order[256 - TAIL_MAX:256]
    s_early = np.sort(np.concatenate([s0, reserve]))
    return dict(s0=s0, dropped=dropped, reserve=reserve, tail=tail,
                s_early=s_early,
                boundary=np.concatenate([tail, reserve]))


def assemble(prep, s_early: np.ndarray, im_end_id: int):
    """Early-288 prompt: [prefix text][vis[s_early]][suffix text], renumbered.

    Returns the assembled embeddings, the suffix text query rows, the boundary
    rows in that sequence, and the question row span (within the suffix).
    """
    prompt, vis, (s, e) = prep["prompt"], prep["vis"], prep["vis_slice"]
    dev = prompt.device
    se = torch.as_tensor(s_early, dtype=torch.long, device=dev)
    vis_new = vis[se]
    n_vis = se.numel()
    prompt_new = torch.cat([prompt[:, :s], vis_new[None],
                            prompt[:, e:]], dim=1)
    # text rows after the visual span, in assembled coordinates
    q0 = s + n_vis                       # first suffix row (<|vision_end|>)
    suffix_rows = torch.arange(q0, prompt_new.shape[1], device=dev)
    # question span in the ORIGINAL ids: after <|vision_end|> (position e in
    # the original ids) up to the first <|im_end|> after the visual span
    ids = prep["inputs"]["input_ids"][0]
    after = (ids == im_end_id).nonzero(as_tuple=True)[0]
    im_end = int(after[after > e].min())
    q_start, q_end = e + 1, im_end       # original-id rows of the question
    q_rows_new = torch.arange(q_start - e + q0, q_end - e + q0, device=dev)
    return dict(prompt=prompt_new, suffix_rows=suffix_rows, q_rows=q_rows_new,
                q_span=(int(q_rows_new[0]), int(q_rows_new[-1]) + 1),
                n_vis=n_vis, s=s, e=e, im_end=im_end)


def rows_of(tokens: np.ndarray, s_early: np.ndarray, s: int) -> np.ndarray:
    """Assembled-sequence rows of the given original visual token indices."""
    pos = np.searchsorted(s_early, tokens)
    assert (s_early[pos] == tokens).all(), "token not in S_early"
    return pos + s
