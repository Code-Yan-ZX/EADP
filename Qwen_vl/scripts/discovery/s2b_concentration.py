"""
S2-B spatial-concentration diagnostic.

"How many 8x8 blocks does a selection touch" is uninformative — 256 tokens is 25%
of the grid, so even a random draw touches essentially all 16 blocks. What matters
is how the budget is *distributed* across blocks, versus the official EADP map and
versus random. All statistics are computed per instance, then averaged.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_audit import BLOCK, OUT

GRID = 32
NB = (GRID // BLOCK) ** 2
rng = np.random.default_rng(0)


def block_hist(score):
    """Share of the top-256 tokens falling in each of the 16 blocks."""
    top = np.argsort(-score)[:256]
    b = (top // GRID // BLOCK) * (GRID // BLOCK) + (top % GRID) // BLOCK
    return np.bincount(b, minlength=NB) / len(top)


def stats(score):
    h = block_hist(score)
    srt = np.sort(h)[::-1]
    return dict(top1=float(srt[0]), top3=float(srt[:3].sum()),
                top1_over_uniform=float(srt[0] / (1.0 / NB)),
                blocks_for_half=int(np.searchsorted(np.cumsum(srt), 0.5) + 1))


def main():
    Z = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    O = np.load(os.path.join(OUT, "s2b_official_selection.npz"))
    meta = json.load(open(os.path.join(OUT, "s2b_gradient_scores_meta.json")))

    rows = {"gradient": [], "official": [], "random": []}
    for k in sorted(meta):
        rows["gradient"].append(stats(Z[k].astype(np.float64)))
        rows["official"].append(stats(O[f"imp__{k}"].astype(np.float64)))
        rows["random"].append(stats(rng.random(1024)))

    print(f"{'map':12s}{'top block share':>17s}{'top-3 share':>13s}"
          f"{'top/ uniform':>14s}{'blocks for 50%':>16s}")
    summary = {}
    for name, rs in rows.items():
        s = {f: float(np.mean([r[f] for r in rs])) for f in
             ("top1", "top3", "top1_over_uniform", "blocks_for_half")}
        summary[name] = s
        print(f"{name:12s}{s['top1']:17.4f}{s['top3']:13.4f}"
              f"{s['top1_over_uniform']:14.2f}{s['blocks_for_half']:16.2f}")
    print(f"\n(uniform reference: each block holds 1/16 = {1/NB:.4f} of the budget; "
          f"a random 256-token draw gives top block share ~{1/NB:.4f})")

    # ---- selection side: what the selector actually kept -------------------
    for tag in ("s2b_pilot.json", "s2b_full.json"):
        p = os.path.join(OUT, tag)
        if not os.path.exists(p):
            continue
        runs = json.load(open(p))["runs"]
        print(f"\nSELECTION-side concentration ({tag})")
        print(f"{'arm':26s}{'blocks used':>13s}{'top block':>11s}{'top-3':>9s}"
              f"{'blocks/half':>13s}")
        groups = {"official": []}
        for k, v in runs.items():
            groups.setdefault(f"{v['selector']}|{v['calibration']}", [])
        for k, v in runs.items():
            for si, i in zip(v["select_idx"], v["idx"]):
                if si is None:
                    continue
                key = f"sel__{v['dataset']}_{int(i)}"
                groups["official"].append(O[key]) if key in O.files else None
                groups[f"{v['selector']}|{v['calibration']}"].append(np.array(si))
        for name, sels in groups.items():
            if not sels:
                continue
            h = [np.bincount((np.array(s) // GRID // BLOCK) * (GRID // BLOCK)
                             + (np.array(s) % GRID) // BLOCK, minlength=NB) / len(s)
                 for s in sels]
            h = np.array(h)
            srt = np.sort(h, axis=1)[:, ::-1]
            print(f"{name:26s}{np.mean((h > 0).sum(axis=1)):13.2f}"
                  f"{srt[:, 0].mean():11.4f}{srt[:, :3].sum(axis=1).mean():9.4f}"
                  f"{np.mean([np.searchsorted(np.cumsum(r), 0.5) + 1 for r in srt]):13.2f}")

    json.dump(summary, open(os.path.join(OUT, "s2b_concentration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
