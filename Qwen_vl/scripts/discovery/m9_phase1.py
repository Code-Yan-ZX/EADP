"""
M9 Phase 1 -- does early LLM interaction break the ranking wall?

For each held-out instance (val+test, 210): assemble the early sequence
S_early = sorted(B2-256 u reserve-32) = 288 visual tokens, run decoder layers
[0, 8) with the read-only capture, and store the text->boundary scores.
No pruning, no generation, no selection -- the teacher (frozen P1-G2 gradient
saliency) never enters the run.

Stored per instance (m9_phase1_scores.npz):
    scores  (n, NL=8 layers, 4 metrics, QMAX query rows, 48 boundary)  fp16
            metrics: 0 att_mean, 1 avn, 2 cmc_mn, 3 cmc_nm
    qlens   (n,)   valid query rows (suffix text rows)
    boundary/reserve/tail  (n, 48/32/16) original visual token ids
Layer index is 0-based (layer 0 = first decoder layer), consistent with
m2_gdep.LAYER = 4 meaning "output of layer 4".
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import eadp_model_name, load_model                       # noqa: E402
from m2_gdep import MODE_FULL, GDEPConfig, GDEPEngine                # noqa: E402
from m5_common import bank_items                                     # noqa: E402
from m9_capture import EarlyLayerCapture                             # noqa: E402
from m9_common import load_banks, instance_boundary, assemble, POOL, TAIL_MAX

N_LAYERS = 8
QMAX = 96
METRICS = ("att_mean", "avn", "cmc_mn", "cmc_nm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="m9_phase1_scores.npz")
    args = ap.parse_args()

    model = load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=8)
    model.model.eval()
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_FULL, tag="M9P1"))
    banks = load_banks()
    raw = {k: banks[k] for k in ("key", "ds", "idx", "split")}
    rows = np.flatnonzero(banks["split"] != "fit")
    items = bank_items(model, raw, splits=("val", "test"))
    assert len(items) == len(rows) == 210
    tok = model.processor.tokenizer
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    n = len(rows)
    out = dict(key=banks["key"][rows], ds=banks["ds"][rows],
               qlens=np.zeros(n, dtype=np.int64),
               nq=np.zeros(n, dtype=np.int64),
               boundary=np.zeros((n, TAIL_MAX + POOL), dtype=np.int64),
               reserve=np.zeros((n, POOL), dtype=np.int64),
               tail=np.zeros((n, TAIL_MAX), dtype=np.int64),
               scores=np.zeros((n, N_LAYERS, len(METRICS), QMAX,
                                TAIL_MAX + POOL), dtype=np.float16))
    t0 = time.time()
    for j, (row, it) in enumerate(zip(rows, items)):
        prep = eng.prepare(it["msg"], it["ds"])
        b = instance_boundary(banks, int(row))
        asm = assemble(prep, b["s_early"], im_end_id)
        bpos = np.searchsorted(b["s_early"], b["boundary"]) + int(asm["s"])
        assert (b["s_early"][bpos - int(asm["s"])] == b["boundary"]).all()
        qrows = asm["suffix_rows"]
        assert qrows.numel() <= QMAX, f"suffix {qrows.numel()} > QMAX"
        cache = None
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=eng.text.config)
        cp = torch.arange(asm["prompt"].shape[1], device=eng.dev)
        pos3 = cp.view(1, 1, -1).expand(3, 1, -1)
        h = asm["prompt"]
        with EarlyLayerCapture(eng.text, 0, N_LAYERS, qrows,
                               torch.as_tensor(bpos, dtype=torch.long)) as cap:
            pos_emb = eng.text.rotary_emb(h, pos3)
            for li in range(N_LAYERS):
                h = eng.layers[li](h, attention_mask=None,
                                   position_ids=pos3[0],
                                   past_key_values=cache,
                                   cache_position=cp,
                                   position_embeddings=pos_emb)
        Q = qrows.numel()
        for li in range(N_LAYERS):
            for mi, m in enumerate(METRICS):
                out["scores"][j, li, mi, :Q] = cap.layers[li][m].numpy() \
                    .astype(np.float16)
        out["qlens"][j] = Q
        # suffix layout: [0]=<|vision_end|>, [1..nq]=question, then scaffolding
        out["nq"][j] = int(asm["q_span"][1] - asm["q_span"][0])
        out["boundary"][j] = b["boundary"]
        out["reserve"][j] = b["reserve"]
        out["tail"][j] = b["tail"]
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{n}  {time.time()-t0:.0f}s", flush=True)

    path = os.path.join(common.OUTPUT_DIR, args.out)
    np.savez_compressed(path, **out)
    print(f"[saved] {path}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
