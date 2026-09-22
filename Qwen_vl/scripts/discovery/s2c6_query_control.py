"""
S2-C6 Part C follow-up: what does the query actually change at the selection
level?

The two query arms fail the same way on the metric but for opposite-looking
reasons, and the difference matters for how the null should be read:

  L4+QUERY      the term's across-image variation is 1.8-4.6 % of its own norm
                -- the model used its 33 792 extra parameters as a per-image
                constant. A null from an arm that never learned to condition on
                the question says nothing about whether the question carries
                information.
  L4+QRY-BILIN  the term's across-image variation is 89-97 % of its norm -- this
                arm *did* learn image-specific conditioning, and replacing the
                query moves held-out scores by up to ~1.0. Its null is therefore
                the informative one.

So the question this script answers is the missing link between those two facts:
when the query is replaced, how many of the selected 256 tokens change, and how
many of the teacher's Top-8 are lost? If the selection moves but the head recall
does not, the conditioning is real and lands on tokens that do not matter.

It reads the frozen checkpoints, trains nothing and selects nothing.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import OUT                                          # noqa: E402
from s2c3_common import BUDGET, aggregate, head_metrics, load_plan  # noqa: E402
from s2c6_common import (COL, derangement, dump, jsonable,        # noqa: E402
                         ViewSource, channel_stats)
from s2c6_models import build                                  # noqa: E402

ARMS = ("L4+QUERY", "L4+QRY-BILIN")
SEEDS = (0, 1, 2)


@torch.no_grad()
def score(model, fs, perm, bs=16):
    outs = []
    for a in range(0, len(fs.rows), bs):
        sel = list(range(a, min(a + bs, len(fs.rows))))
        xs, q = fs.batch(sel, True, q_perm=perm)
        outs.append(model(xs, q).float().cpu())
    return torch.cat(outs, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s2c6")
    args = ap.parse_args()

    meta, plan, keys, rows_of = load_plan()
    stats = {"h4": channel_stats("h4", rows_of["fit"])}
    fs = ViewSource(("h4",), rows_of["test"], stats, "cuda", q_layer=4)
    test_keys = [keys[i] for i in rows_of["test"]]
    G = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    orders = {k: np.argsort(-G[k].astype(np.float64), kind="stable")
              for k in test_keys}
    n = len(test_keys)

    out = {"stage": "S2-C6", "step": "query control at the selection level",
           "note": "honest vs another image's query (a derangement, so no image "
                   "keeps its own). 'jaccard256' is the overlap of the two "
                   "selected 256-token sets; 'changed256' is 1 - jaccard, i.e. "
                   "the fraction of the selection the query actually moves.",
           "arms": {}}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            model = build(arm).to("cuda")
            model.load_state_dict(torch.load(
                os.path.join(OUT, f"{args.tag}_{arm}_s{seed}__H8.pt"),
                map_location="cuda"))
            model.eval()
            s_h = score(model, fs, np.arange(n))
            s_w = score(model, fs, derangement(n, seed))
            per = []
            for i, k in enumerate(test_keys):
                a = set(np.argsort(-s_h[i], kind="stable")[:BUDGET].tolist())
                b = set(np.argsort(-s_w[i], kind="stable")[:BUDGET].tolist())
                mh = head_metrics(s_h[i], orders[k])
                mw = head_metrics(s_w[i], orders[k])
                per.append(dict(
                    jaccard256=len(a & b) / len(a | b),
                    changed256=1.0 - len(a & b) / len(a | b),
                    head8_honest=mh["head_recall8"],
                    head8_wrong=mw["head_recall8"],
                    head8_delta=mh["head_recall8"] - mw["head_recall8"]))
            agg = {kk: float(np.mean([p[kk] for p in per])) for kk in per[0]}
            agg.update(seed=seed,
                       abs_score_diff_mean=float(np.abs(s_w - s_h).mean()),
                       abs_score_diff_max=float(np.abs(s_w - s_h).max()),
                       n_params=int(sum(p.numel() for p in model.parameters())))
            rows.append(agg)
            print(f"[{arm} s{seed}] changed256={agg['changed256']:.4f} "
                  f"jaccard={agg['jaccard256']:.4f} | head8 honest="
                  f"{agg['head8_honest']:.4f} wrong={agg['head8_wrong']:.4f} "
                  f"delta={agg['head8_delta']:+.4f} | "
                  f"|ds|mean={agg['abs_score_diff_mean']:.3e} "
                  f"max={agg['abs_score_diff_max']:.3e}", flush=True)
            del model
            torch.cuda.empty_cache()
        out["arms"][arm] = dict(
            per_seed=rows,
            mean={k: float(np.mean([r[k] for r in rows]))
                  for k in ("changed256", "jaccard256", "head8_honest",
                            "head8_wrong", "head8_delta", "abs_score_diff_mean",
                            "abs_score_diff_max")})
    dump(f"{args.tag}_query_control.json", out)


if __name__ == "__main__":
    main()
