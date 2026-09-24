"""
M3-v2 step 1 -- cache the instruction-token INPUT EMBEDDINGS of the frozen 450.

The v2 auditor is query-conditioned, so it needs the question on the text side.
The brief allows exactly one source for it:

    "text embedding lookup 可以使用, L0/L1/... decoder hidden state 不能使用"

This script calls the wrapper's own `_get_instruction_sequence_embedding`, which
is `self.model.get_input_embeddings()(ids)` on the tokenised instruction text --
an embedding lookup, no decoder layer.  It is the SAME call `m2_gdep.
GDEPEngine._instruction_embeds` makes at serve time and the same tensor the
incumbent's own pruner consumes as `text_embeds_seq_llm`, so the live sequence
is the training sequence token for token (see the M3-v0 gate G-F note on why
this must be the instruction text alone and not the templated prompt).

One model load, one embedding lookup per instance, no vision forward, no LLM
layer, no generation.  Row order is `m3_features.bank_rows`'s, so the resulting
file is row-aligned with `m3_bank_v1.npz`; the two are asserted equal by key
whenever a bank is opened.

Usage
    python scripts/discovery/m3v2_text.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine              # noqa: E402
from m3_common import BASE_SELECTOR, BUDGET                          # noqa: E402
from m3_features import bank_rows                                    # noqa: E402

TAG = "m3v2_text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vis-bank", default="m3_bank_v1")
    args = ap.parse_args()

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=64)
    model.model.eval()
    torch.set_grad_enabled(False)
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M3V2TXT"))

    rows = bank_rows(model)
    if args.limit:
        rows = rows[: args.limit]

    keys, seqs = [], []
    t0 = time.time()
    for n_, it in enumerate(rows):
        prep = eng.prepare(it["msg"], it["ds"])
        _, text_seq = eng._instruction_embeds(prep)
        s = text_seq.reshape(-1, text_seq.shape[-1])       # (L, D)
        assert s.shape[0] >= 1 and torch.isfinite(s).all()
        keys.append(it["key"])
        seqs.append(s.detach().to(torch.float16).cpu().numpy())
        del prep, text_seq, s
        if (n_ + 1) % 100 == 0:
            print(f"  {n_+1}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    L_max = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    txt = np.zeros((len(seqs), L_max, D), dtype=np.float16)
    txt_len = np.zeros(len(seqs), dtype=np.int32)
    for i, s in enumerate(seqs):
        txt[i, : s.shape[0]] = s
        txt_len[i] = s.shape[0]

    # Row alignment with the vision bank is a hard precondition of every
    # downstream join, so it is checked here, where the mistake would be made.
    vpath = os.path.join(OUTPUT_DIR, f"{args.vis_bank}.npz")
    vkeys = [str(k) for k in np.load(vpath, allow_pickle=False)["key"]]
    assert vkeys == keys, "text rows are not aligned with the vision bank"

    path = os.path.join(OUTPUT_DIR, f"{args.tag}.npz")
    np.savez(path, key=np.array(keys), txt=txt, txt_len=txt_len)
    meta = dict(n=len(keys), L_max=int(L_max), L_mean=float(txt_len.mean()),
                L_p95=float(np.percentile(txt_len, 95)), d_txt=int(D),
                vis_bank=args.vis_bank, row_aligned=True,
                source="model.get_input_embeddings()(_tokenize_instruction(msg))",
                wall_seconds=float(time.time() - t0))
    with open(os.path.join(OUTPUT_DIR, f"{args.tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[saved] {path}  {meta}")


if __name__ == "__main__":
    main()
