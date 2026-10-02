"""Freeze full-split RTG banks (rtg scorer, K=256) for the three main-panel
datasets.  Same record schema as the DEV banks; per-sample RTG selection
(anchors are NOT inherited from the EADP bank).  Resumable.

Usage: python rtg_bank_full.py --datasets TextVQA_VAL,DocVQA_VAL,OCRBench
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time

import torch

import rtg_common as RC
import amp_common as AC


def bank_full_path(ds: str) -> str:
    return os.path.join(RC.OUT_DIR, "full", f"bank_full_rtg_{ds}.json.gz")


def _save(out_path: str, bank: dict) -> str:
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, out_path)
    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_full_bank(ds: str, eng, limit: int | None = None) -> None:
    out_path = bank_full_path(ds)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bank = {"meta": dict(scorer="rtg", split="full", ds=ds, K=RC.K,
                         params=RC.bank_config("rtg", ds)["params"],
                         base_commit=RC.git_commit()), "samples": {}}
    if os.path.exists(out_path):
        with gzip.open(out_path, "rt") as f:
            bank = json.load(f)
    dataset = AC.common.build_dataset(ds)
    eng.vlm.set_dump_image(dataset.dump_image)
    n_rows = len(dataset.data)
    items = range(n_rows) if limit is None else range(min(limit, n_rows))
    t0 = time.time()
    for n, i in enumerate(items):
        key = str(i)
        if key in bank["samples"]:
            continue
        row = dataset.data.iloc[i]
        msg = AC.common.build_message(eng.vlm, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        if prep["n_vis"] <= RC.K:
            bank["samples"][key] = dict(
                idx=int(i), n_vis=int(prep["n_vis"]),
                keep=list(range(prep["n_vis"])), gid=[], gsize=[],
                w_diag=dict(eff_tokens=float(prep["n_vis"]),
                            L=int(prep["n_vis"])))
            continue
        V, DS = eng.encode(prep)
        text_mean, text_seq = eng.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, DS=DS, K=RC.K, engine=eng,
                   text_mean=text_mean, text_seq=text_seq,
                   ds=ds, qid=int(i))
        keep, diags = RC.select_keep("rtg", ctx, RC.K, want_diag=True)
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        gid = gid.cpu()
        counts = torch.bincount(gid, minlength=int(keep.numel()))
        bank["samples"][key] = dict(
            idx=int(i), n_vis=int(prep["n_vis"]),
            keep=keep.cpu().tolist(), gid=gid.tolist(),
            gsize=counts.tolist(), w_diag=diags[0])
        if (n + 1) % 100 == 0:
            _save(out_path, bank)
        if (n + 1) % 100 == 0 or n + 1 == len(items):
            el = time.time() - t0
            print(f"[bank full rtg {ds}] {n+1}/{len(items)} "
                  f"({el/(n+1):.2f}s/q)", flush=True)
    sha = _save(out_path, bank)
    print(f"[saved] {out_path} ({len(bank['samples'])} records, "
          f"sha {sha[:12]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets",
                    default="TextVQA_VAL,DocVQA_VAL,OCRBench")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    for ds in args.datasets.split(","):
        build_full_bank(ds, eng, args.limit)


if __name__ == "__main__":
    main()
