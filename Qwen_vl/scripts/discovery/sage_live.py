"""
SAGE -- the LIVE deployment machinery, shared by val calibration and the fresh
confirmation. Everything here runs inside the pruner's forward, i.e. inside the
measured TTFT window: edge construction, critic scoring and the threshold
decision all happen after the vision tower and before the first decoder block,
on tensors the incumbent's own pass already produced.

The offline phi_batch/unary_batch/build_edge_support are THE functions used
here (m5_vis rows are bit-identical bf16 captures of `image_features`, so
offline training features and live features cannot skew).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from m5_common import BUDGET                                         # noqa: E402
from common import OUTPUT_DIR                                        # noqa: E402
from m7_accuracy import SetPruner, install as _install_set           # noqa: E402
from m5_common import BASE_SELECTOR                                  # noqa: E402
import sage_common as S                                              # noqa: E402


def _lex_argmax(scores: np.ndarray, edges: list) -> int:
    """argmax with the pre-registered tie rule: lexicographically smallest
    (sorted G-, sorted G+) among exactly-tied maxima (decision 10)."""
    best = float(scores.max())
    cand = [j for j in range(len(edges)) if scores[j] == best]
    return min(cand, key=lambda j: (tuple(edges[j]["minus"]),
                                    tuple(edges[j]["plus"])))


class SAGEPruner(SetPruner):
    """B2's own pass, then ONE early exchange decision (eq. 5).

    `family` selects the critic input: "set" (the survivor-conditioned phi) or
    "unary" (the 8-scalar control). Nothing else differs -- the control must sit
    in exactly the same harness, measured the same way.
    """

    def __init__(self, *a, critic=None, mu=None, sd=None, tau=0.0,
                 g: int = 1, family: str = "set", record=False, **kw):
        super().__init__(*a, **kw)
        self.critic = critic
        self.mu = mu
        self.sd = sd
        self.tau = tau
        self.g = g
        self.family = family
        self.record = record
        self.last_sage = {}
        self.last_sage_events = None

    @torch.no_grad()
    def forward(self, image_features, text_embeds_llm, text_embeds_seq_llm,
                grid_thw):
        self.last_sage = {}
        self.last_sage_events = None
        self.keep_gpu = True
        self.last_gpu = {}
        super().forward(image_features, text_embeds_llm, text_embeds_seq_llm,
                        grid_thw)
        self.keep_gpu = False
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()

        g_gpu = self.last_gpu
        self.last_gpu = {}
        s0 = torch.sort(g_gpu["select_idx"][0].to(torch.long)).values
        vis = g_gpu["image_features"]
        imp = g_gpu["importance"].reshape(-1).float()
        cos = S.cos_to_s0c(vis, s0)

        edges = S.build_edge_support(imp.cpu().numpy(), cos.cpu().numpy(),
                                     s0.cpu().numpy(), g=self.g)
        out_set = s0
        decision = dict(n_edges=len(edges), g=self.g, family=self.family,
                        accepted=False, tau=self.tau)
        if edges:
            q = text_embeds_llm[0] if self.family == "set" else None
            if self.family == "set":
                phi = S.phi_batch(vis, q, s0, edges)
                x = (phi - self.mu) / self.sd
            else:
                x = (S.unary_batch(imp, cos, edges) - self.mu) / self.sd
            scores = self.critic(x).detach().cpu().numpy()
            j = _lex_argmax(scores, edges)
            pred = float(scores[j])
            decision["pred_gain"] = pred
            if pred > self.tau:
                e = edges[j]
                Se = torch.tensor(sorted(
                    (set(s0.tolist()) - set(e["minus"])) | set(e["plus"])),
                    dtype=torch.long, device=s0.device)
                assert Se.numel() == BUDGET
                out_set = Se
                decision.update(accepted=True, chosen=j,
                                minus=e["minus"], plus=e["plus"])
            else:
                decision["chosen"] = j
        ev1.record()
        self.last_sage_events = (ev0, ev1)
        if self.record:
            rec = dict(decision)
            rec.update(s0_idx=s0.detach().cpu().tolist(),
                       final_idx=out_set.detach().cpu().tolist())
            if self.record == "full" and edges:
                rec["edges"] = edges
                rec["scores"] = [float(s) for s in scores]
            self.last_sage = rec
        return vis[out_set].to(image_features.dtype), [int(out_set.numel())]

    def read_sage_ms(self) -> float:
        if self.last_sage_events is None:
            return float(self.last_sage.get("sage_ms") or 0.0)
        a, b = self.last_sage_events
        self.last_sage_events = None
        a.synchronize()
        ms = float(a.elapsed_time(b))
        self.last_sage["sage_ms"] = ms
        return ms


def load_critic(path: str, device):
    """Rebuild a saved critic + its standardisation stats on `device`."""
    ckpt = torch.load(os.path.join(OUTPUT_DIR, path), map_location="cpu",
                      weights_only=False)
    meta = ckpt["meta"]
    d_in = S.D_PHI if meta["family"] == "set" else S.D_UNARY
    model = S.SageCritic(d_in, meta["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    mu = torch.tensor(meta["mu"], dtype=torch.float32, device=device)
    sd = torch.tensor(meta["sd"], dtype=torch.float32, device=device)
    return model, mu, sd, meta


def install_sage(eng, model, critic, mu, sd, tau, g, family="set",
                 record=False):
    old = eng.pruner
    p = SAGEPruner(visual_token_num=old.visual_token_num, alpha=old.alpha,
                   beta=old.beta, visual_dim=old.visual_dim,
                   spatial_merge_size=old.spatial_merge_size,
                   selector=BASE_SELECTOR, capture=False, final=None,
                   critic=critic, mu=mu, sd=sd, tau=tau, g=g, family=family,
                   record=record)
    p = p.to(next(model.model.parameters()).device)
    p.eval()
    p.sim_mode = "rebound"
    eng.pruner = p
    model.pruner = p
    assert eng.pruner is model.pruner, "dual-hook install failed"
    return p


def run_one(eng, item, max_new: int):
    """One request with m2_perf_paired's TTFT convention: sync -> [prepare +
    prefill] -> sync; then greedy decode to the end."""
    with S.TTFT() as t:
        prep = eng.prepare(item["msg"], item["ds"])
        state, info = eng.prefill(prep)
    ids = eng.decode(state, max_new)
    text = eng.vlm.processor.tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return dict(prediction=eng.vlm._post_process_response(text),
                n_decode=len(ids), ttft_ms=t.ms, info=info)
