"""
M3-v0 -- offline forecast of the learned arm's gain, before the GPU grid.

Given a frozen student, rebuild its rescue set on the VAL split (never test) and
report the teacher mass it recovers.  Two calibrations of "teacher mass ->
accuracy" are already on disk for these exact instances, so the forecast is not
free invention:

  mass      F(S) interpolated linearly between B2's set and the teacher's own
            Top-256 (S2-B measured both endpoints: 61.10 and 77.80);
  frontload S2-C2's measured cumulative-gap curve (39.6 % of the gap by token 6,
            52.9 % by 8, 77.0 % by 16) applied to the same endpoints.

The mass forecast is a LOWER bound: mass share is concave in rank depth while
the value is convex, and on S2-C2's own LIN_L4 curve the mass forecast
understated the measured result by +3.3/+4.9/+5.1 at k = 8/16/32.

Usage
    python scripts/discovery/m3_forecast.py --split val
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m3_analyze import S2B_PILOT, _frontload_fraction               # noqa: E402
from m3_common import (FEATURES, MissStudent, evict_t,               # noqa: E402
                       handcrafted_t, teacher_topk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="m3_bank")
    ap.add_argument("--student", default="m3_miss")
    ap.add_argument("--split", default="val")
    ap.add_argument("--rs", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--rule", default="lowimp")
    args = ap.parse_args()

    # NpzFile re-reads an uncompressed member on every __getitem__, so the
    # 3.8 GB vision member must be pulled out once, not once per instance.
    z = np.load(os.path.join(OUTPUT_DIR, f"{args.bank}.npz"), allow_pickle=False)
    z = {k: np.asarray(z[k]) for k in z.files}
    split = np.array([str(s) for s in z["split"]])
    rows = np.where(split == args.split)[0]
    ck = torch.load(os.path.join(OUTPUT_DIR, f"{args.student}.pt"),
                    map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MissStudent(ck["d_hand"], ck["d_vis"])
    m.load_state_dict(ck["state_dict"])
    m.eval().to(dev)
    mu_h = torch.from_numpy(ck["mu_hand"]).to(dev)
    sd_h = torch.from_numpy(ck["sd_hand"]).to(dev)
    mu_v = torch.from_numpy(ck["mu_v"] if "mu_v" in ck else ck["mu_vis"]).to(dev)
    sd_v = torch.from_numpy(ck["sd_v"] if "sd_v" in ck else ck["sd_vis"]).to(dev)
    per_z = bool(ck["per_instance_z"])

    out = dict(split=args.split, n=int(rows.size), student=ck["config_name"],
               rule=args.rule, results={})
    for r in args.rs:
        F_learn, F_or, ovl = [], [], []
        for i in rows:
            X = torch.from_numpy(z["X"][i]).to(dev)
            V = torch.from_numpy(z["vis"][i].astype(np.float32)).to(dev)
            s0 = torch.from_numpy(np.sort(z["s0"][i].astype(np.int64))).to(dev)
            g2 = torch.from_numpy(z["g2"][i]).to(dev)
            n = X.shape[0]
            keep = torch.zeros(n, dtype=torch.bool, device=dev)
            keep[s0] = True
            drop = (~keep).nonzero(as_tuple=True)[0]
            with torch.no_grad():
                zh = (X[drop] - mu_h) / sd_h
                zv = (V[drop] - mu_v) / sd_v
                if per_z:
                    zh = (zh - zh.mean(0, keepdim=True)) / zh.std(0, keepdim=True).clamp_min(1e-6)
                sc = m(zh, zv).float()
            res_l = drop[torch.argsort(-sc)[:r]]
            res_o = drop[torch.argsort(-g2[drop])[:r]]
            imp = torch.from_numpy(z["importance"][i] if "importance" in z.files
                                   else z["X"][i][:, 0]).to(dev).float()
            sim = None
            for tag, res in (("learn", res_l), ("oracle", res_o)):
                ev = _evict(z, i, s0, r, args.rule, dev)
                mask = torch.ones(n, dtype=torch.bool, device=dev)
                mask[ev] = False
                S = torch.cat([s0[mask[s0]], res])
                g = torch.from_numpy(z["g2"][i]).to(dev).float()
                denom = g[torch.argsort(-g)[:256]].sum()
                (F_learn if tag == "learn" else F_or).append(
                    float(g[S].sum() / denom))
            ovl.append(len(set(res_l.tolist()) & set(res_o.tolist())) / r)
        f0 = float(np.mean([
            _F0(z, i) for i in rows]))
        span = S2B_PILOT["teacher_facility"] - S2B_PILOT["official"]
        fl, fo = float(np.mean(F_learn)), float(np.mean(F_or))
        out["results"][r] = dict(
            F_learned=fl, F_oracle=fo, F_B2=f0,
            overlap_with_oracle=float(np.mean(ovl)),
            learned_mass_forecast=S2B_PILOT["official"] + (fl - f0) / (1 - f0) * span,
            oracle_mass_forecast=S2B_PILOT["official"] + (fo - f0) / (1 - f0) * span,
            learned_frontload_forecast=S2B_PILOT["official"]
            + _frontload_fraction(r) * (fl - f0) / (1 - f0) * span,
            oracle_frontload_forecast=S2B_PILOT["official"]
            + _frontload_fraction(r) * (fo - f0) / (1 - f0) * span)
        d = out["results"][r]
        print(f"r={r:2d}  F(B2)={f0:.3f}  F_learn={fl:.3f}  F_oracle={fo:.3f}  "
              f"ovl={d['overlap_with_oracle']:.3f}  |  forecast learned "
              f"{d['learned_mass_forecast']:.1f}  oracle {d['oracle_mass_forecast']:.1f}"
              f"   (B2 59.88, B1 61.10, teacher-map 77.80)")
    with open(os.path.join(OUTPUT_DIR, "m3_forecast.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[saved] m3_forecast.json")


def _F0(z, i):
    g2 = np.asarray(z["g2"][i], np.float64)
    return float(g2[np.asarray(z["s0"][i], np.int64)].sum()
                 / g2[np.argsort(-g2)[:256]].sum())


def _evict(z, i, s0, r, rule, dev):
    if rule == "lowimp":
        imp = torch.from_numpy(z["X"][i][:, 0]).to(dev).float()
        return s0[torch.argsort(imp[s0])[:r]]
    raise KeyError(rule)


if __name__ == "__main__":
    main()
