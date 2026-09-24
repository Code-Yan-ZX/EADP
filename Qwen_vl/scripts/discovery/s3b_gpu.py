"""
S3-B GPU helpers: one scoring path shared by the base, sweep, ablation and
crop stages, so every arm in the stage is measured under identical conditions.

Everything runs through the frozen S3-A harness (S3AHarness): the retained set
is realised by splicing `vision_tower(image)[set]` in raster order between the
prompt prefix and suffix (gate S3A-G2 proved this bit-identical to the S2-C2
indicator-map delivery through the incumbent pruner), greedy decoding for the
official accuracy, and teacher-forced gold-answer NLL as the secondary signal.

Generation cap: S2-C2 and the M2 runs generated with the model default (2048
tokens), the S3-A harness used 64. S3-B uses 256 -- far above every legitimate
answer in these three benchmarks (gold answers are <= ~12 tokens), so no real
answer is truncated, while looped generations stay cheap. The cap is identical
for every arm of both banks, which is what within-stage comparisons need;
gates B2/B3 reconcile it against the frozen records.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                    # noqa: E402,F401
from common import eadp_model_name                               # noqa: E402,F401
from s1_audit import OUT                                         # noqa: E402,F401
from s3a_nll import S3AHarness                                   # noqa: E402,F401
from scoring import per_sample_hits                              # noqa: E402,F401
import s3b_common as B                                           # noqa: E402,F401

GEN_CAP = 256


def load_stack():
    model = common.load_model(eadp_model_name(B.BUDGET, 0.5, 2.0),
                              max_new_tokens=GEN_CAP)
    model.model.eval()
    torch.set_grad_enabled(False)
    return model, S3AHarness(model)


def instance_prep(h, model, datasets, rec):
    """message -> harness prep (vision embeddings are disk-cached) + gold ids."""
    ds_obj = datasets[rec["ds"]]
    model.set_dump_image(ds_obj.dump_image)
    row = ds_obj.data.iloc[rec["idx"]]
    msg = common.build_message(model, ds_obj, rec["ds"], row)
    prep = h.prep(msg, rec["ds"], rec["key"])
    golds = B.A.golds_of(row)
    return prep, h.answer_ids(golds), row, golds


def require_complete(results, planned, tag):
    """a stage that swallowed per-instance errors must not report success:
    every one of these stages writes exit=0 even when every instance failed."""
    done = len(results["done"])
    print(f"[{tag}] completed {done}/{planned} planned units")
    if done < planned:
        print(f"[{tag}] INCOMPLETE -- {planned - done} units missing")
        raise SystemExit(3)


def measure_floor(h, prep, ans, rec):
    """per-instance bf16 resolution floor: same set, batched vs per-gold."""
    _, per, _, _ = h.nll(prep, rec["S"], ans)
    singles = [h.nll(prep, rec["S"], [t])[0] for t in ans]
    return float(np.max(np.abs(np.array(per) - np.array(singles))))


def score_set(h, prep, ans, rec, adds, drops, row, kept=None):
    """one arm: teacher-forced NLL + greedy generation + official hit."""
    sel = B.apply_arms(rec, adds, drops) if kept is None else list(kept)
    assert len(sel) == B.BUDGET, len(sel)
    L, per_gold, first, P = h.nll(prep, sel, ans)
    ids, text = h.generate(prep, sel, max_new=GEN_CAP)
    pred = h.vlm._post_process_response(text)
    hit = float(per_sample_hits(rec["ds"], [row], [pred])[0])
    return dict(L=float(L), hit=hit, pred=pred, n_tokens=int(len(ids)),
                n_kept=int(len(sel)))
