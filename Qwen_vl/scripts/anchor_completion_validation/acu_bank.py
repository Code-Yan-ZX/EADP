"""Anchor Completion Validation — freeze full-split anchor banks (protocol §6).

For every row of a dataset: vision encode once, official EADP importance +
official facility-location anchors (selector b1, K=256) on the RAW main
features, cos-argmax assignment of dropped tokens.  One bank per dataset,
resumable, lean records (keep/gid/gsize/n_vis/image_key).  Downstream arms
ONLY read the bank — never reselect.

Usage: python acu_bank.py --datasets TextVQA_VAL,DocVQA_VAL,OCRBench
       python acu_bank.py --datasets OCRBench --limit 5
"""

from __future__ import annotations

import argparse
import gzip
import json
import time

import torch

import acu_common as AU


def build_bank(ds: str, limit: int | None, eng) -> None:
    out_path = AU.bank_path(ds)
    bank = {}
    try:
        with gzip.open(out_path, "rt") as f:
            bank = json.load(f)
    except FileNotFoundError:
        pass
    dataset = AU.common.build_dataset(ds)
    eng.vlm.set_dump_image(dataset.dump_image)
    n_rows = len(dataset.data)
    items = range(n_rows) if limit is None else range(min(limit, n_rows))
    t0 = time.time()
    for n, i in enumerate(items):
        key = str(i)
        if key in bank:
            continue
        row = dataset.data.iloc[i]
        msg = AU.common.build_message(eng.vlm, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        n_vis = int(prep["n_vis"])
        if n_vis <= AU.K:                     # degenerate: keep-all entry
            bank[key] = dict(idx=int(i), image_key=None, n_vis=n_vis,
                             keep=list(range(n_vis)), gid=[], gsize=[])
            bank[key]["image_key"] = _image_key(ds, row)
            continue
        V, DS = eng.encode(prep)
        text_mean, text_seq = eng.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, DS=DS, K=AU.K, engine=eng,
                   text_mean=text_mean, text_seq=text_seq)
        keep = AU.official_facility_keep(ctx, AU.K)
        split_sizes = AU.split_sizes_from_gthw(
            prep, eng.inner.visual.spatial_merge_size)
        dropped_idx, gid, _ = AU.compute_assignment_per_image(V, keep,
                                                              split_sizes)
        gid = gid.cpu()
        counts = torch.bincount(gid, minlength=int(keep.numel()))
        bank[key] = dict(idx=int(i), image_key=_image_key(ds, row),
                         n_vis=n_vis, keep=keep.cpu().tolist(),
                         gid=gid.tolist(), gsize=counts.tolist())
        if (n + 1) % 50 == 0 or n + 1 == len(items):
            el = time.time() - t0
            print(f"[bank {ds}] {n+1}/{len(items)} ({el/(n+1):.2f}s/q)",
                  flush=True)
        if (n + 1) % 500 == 0:
            _save(out_path, bank)
    sha = _save(out_path, bank)
    print(f"[saved] {out_path} ({len(bank)} records, sha {sha[:12]})",
          flush=True)


def _image_key(ds: str, row) -> str:
    from acu_exposure import image_key
    return image_key(ds, row)


def _save(out_path: str, bank: dict) -> str:
    import hashlib
    import os
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, out_path)
    with open(out_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    eng = AU.load_engine(max_new_tokens=64)
    for ds in args.datasets.split(","):
        build_bank(ds, args.limit, eng)


if __name__ == "__main__":
    main()
