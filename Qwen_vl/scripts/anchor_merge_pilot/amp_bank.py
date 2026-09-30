"""Anchor-Merge Pilot — freeze the anchor bank (protocol §3.3).

One pass per (split, dataset): for every manifest sample, run the vision tower
once, compute the official-facility anchor set S on the RAW main features and
the cos-argmax assignment of dropped tokens, and store both plus compact
per-group mechanism diagnostics.  Downstream arms ONLY read the bank — no arm
ever recomputes a selection (frozen rule; the bank file hash is recorded with
every result shard).

Bank record per sample:
  idx, image_key, keep[256], gid[Nd]
  gsize[256]                members per group
  rnorm_main[256]           ||m-a|| / ||a|| (uniform mean, main stream)
  cos_am_main[256]          cos(a_j, m_j)   (uniform mean, main stream)
  rnorm_ds[256]             mean over the 3 DS streams of ||m-a||/||a||
  cos_am_ds[256]            mean over the 3 DS streams of cos(a, m)

Usage: python amp_bank.py --split dev [--limit N]
       python amp_bank.py --split confirm
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import torch

import amp_common as AC


def build_for_split(split: str, limit: int | None):
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    K = AC.K

    for ds in AC.DS_LIST:
        items = manifest["datasets"][ds][split]
        if limit:
            items = items[:limit]
        out_path = AC.bank_path(split, ds)
        bank = {}
        if os.path.exists(out_path):
            with gzip.open(out_path, "rt") as f:
                bank = json.load(f)
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        t0 = time.time()
        for n, it in enumerate(items):
            key = str(it["idx"])
            if key in bank:
                continue
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            if prep["n_vis"] <= K:            # degenerate: keep-all bank entry
                bank[key] = dict(idx=int(it["idx"]),
                                 image_key=it["image_key"],
                                 keep=list(range(prep["n_vis"])), gid=[],
                                 gsize=[], rnorm_main=[], cos_am_main=[],
                                 rnorm_ds=[], cos_am_ds=[])
                continue
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            keep = AC.official_facility_keep(ctx, K)
            dropped_idx, gid, _ = AC.compute_assignment(V, keep)
            gid = gid.cpu()
            Kd = int(keep.numel())
            rec = dict(idx=int(it["idx"]), image_key=it["image_key"],
                       keep=keep.cpu().tolist(),
                       gid=gid.tolist())
            counts = torch.bincount(gid, minlength=Kd)
            rec["gsize"] = counts.tolist()
            # uniform-mean diagnostics on main + each DS stream
            def _diag(feat):
                a = feat[keep].float()
                mem = feat[dropped_idx].float()
                s = AC.group_sum(mem, gid.to(feat.device), Kd) + a
                m = s / (counts.to(feat.device).float().unsqueeze(1) + 1.0)
                na = a.norm(dim=1).clamp_min(1e-12)
                rnorm = ((m - a).norm(dim=1) / na)
                cos_am = torch.nn.functional.cosine_similarity(a, m, dim=1)
                return rnorm.cpu(), cos_am.cpu()
            r0, c0 = _diag(V)
            rec["rnorm_main"] = [round(float(x), 6) for x in r0]
            rec["cos_am_main"] = [round(float(x), 6) for x in c0]
            rds = torch.zeros(Kd)
            cds = torch.zeros(Kd)
            for dss in DS:
                r1, c1 = _diag(dss)
                rds += r1
                cds += c1
            rec["rnorm_ds"] = [round(float(x) / len(DS), 6) for x in rds]
            rec["cos_am_ds"] = [round(float(x) / len(DS), 6) for x in cds]
            bank[key] = rec
            if (n + 1) % 20 == 0 or n + 1 == len(items):
                el = time.time() - t0
                print(f"[bank {split} {ds}] {n+1}/{len(items)} "
                      f"({el/(n+1):.2f}s/q)", flush=True)
        tmp = out_path + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(bank, f)
        os.replace(tmp, out_path)
        print(f"[saved] {out_path} ({len(bank)} records, "
              f"sha {AC.bank_sha256(split, ds)[:12]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "confirm"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build_for_split(args.split, args.limit)


if __name__ == "__main__":
    main()
