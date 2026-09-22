"""
S2-B step 2: run the gradient-scored generation arms on the frozen set.

The scoring map is swapped and nothing else: the selection operator and the
generation path are the Stage-1 code verbatim (`diag_selectors.generate_prediction`).
The precomputed P1-G2 map is injected through a `_score` override, so the
similarity kernel, the selector and `_build_pruned_inputs` are untouched.

Calibrations (the only ones tested, per the brief):
    C1  s = g2 / mean(g2)
    C2  s = (g2 - min) / (max - min)
    C3  s = C2 ** 2
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import build_dataset, build_message, eadp_model_name, ensure_out_dir, sample_indices
from diag_selectors import generate_prediction
from instrumented import TimedEADPPruner
from scoring import per_sample_hits
from s1_audit import OUT


def calibrate(g2: np.ndarray, name: str):
    """Return the importance map, or None meaning "use the official EADP score".

    ``official`` is the harness identity check: it must reproduce the Stage-1
    ``facility @256`` numbers exactly, which proves the runner itself is neutral.
    """
    if name == "official":
        return None
    if name == "raw":
        return g2
    if name == "C1":
        return g2 / (g2.mean() + 1e-12)
    lo, hi = g2.min(), g2.max()
    mm = (g2 - lo) / (hi - lo + 1e-12)
    if name == "C2":
        return mm
    if name == "C3":
        return mm ** 2
    raise KeyError(name)


class GradientPruner(TimedEADPPruner):
    """Instrumented pruner whose importance map is supplied externally."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.override = None

    def _score(self, image_features, text_embeds, text_embeds_seq, grid_h, grid_w, timer):
        if self.override is not None:
            return self.override
        return super()._score(image_features, text_embeds, text_embeds_seq,
                              grid_h, grid_w, timer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--cals", nargs="+", default=["C1", "C2", "C3"])
    ap.add_argument("--selectors", nargs="+", default=["facility"])
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--pilot", action="store_true",
                    help="use every 3rd index of the frozen 150 (=> 50 per benchmark)")
    ap.add_argument("--tag", default="s2b")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap instances per dataset")
    ap.add_argument("--shuffle-scores", type=int, default=0,
                    help="control: shift the score->instance assignment by N "
                         "(the gradient maps are correct but belong to other images)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    Z = np.load(os.path.join(OUT, "s2b_gradient_scores.npz"))
    have = set(Z.files)
    keys_all = sorted(have)

    model = common.load_model(eadp_model_name(args.budget, 0.5, 2.0),
                              max_new_tokens=args.max_new_tokens)
    pruner = GradientPruner(
        visual_token_num=args.budget, alpha=0.5, beta=2.0,
        visual_dim=model.pruner.visual_dim,
        spatial_merge_size=model.pruner.spatial_merge_size,
        selector=args.selectors[0], capture=True,
    ).to(next(model.model.parameters()).device)
    pruner.eval()
    model.pruner = pruner

    bank = {}
    for ds in args.datasets:
        dataset = build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        idx = sample_indices(len(dataset.data), 150, offset=0)
        if args.pilot:
            idx = idx[::3]
        if args.limit:
            idx = idx[: args.limit]
        bank[ds] = dict(dataset=dataset, idx=idx,
                        rows=[dataset.data.iloc[i] for i in idx],
                        msgs=[build_message(model, dataset, ds, dataset.data.iloc[i])
                              for i in idx])
        print(f"[bank] {ds}: {len(idx)} instances"
              f"{' (pilot: every 3rd of 150)' if args.pilot else ''}")

    out_path = os.path.join(ensure_out_dir(), f"{args.tag}.json")
    results = {"config": vars(args), "runs": {}}
    if os.path.exists(out_path):                       # merge, so arms can run in batches
        try:
            prev = json.load(open(out_path))
            results["runs"] = prev.get("runs", {})
            print(f"[merge] {len(results['runs'])} existing arm-records in {out_path}")
        except Exception:
            pass

    for cal in args.cals:
        for sel in args.selectors:
            pruner.selector_name = sel
            for ds in args.datasets:
                b = bank[ds]
                preds, meta = [], []
                for n, m in enumerate(tqdm(b["msgs"], desc=f"{cal}/{sel}/{ds}", leave=False)):
                    i = b["idx"][n]
                    key = f"{ds}_{i}"
                    if key not in have:
                        raise SystemExit(f"missing gradient score for {key}")
                    src = Z[keys_all[(keys_all.index(key) + args.shuffle_scores) % len(keys_all)]] \
                        if args.shuffle_scores else Z[key]
                    s = calibrate(src.astype(np.float64), cal)
                    pruner.override = (None if s is None else
                                       torch.from_numpy(s).float().unsqueeze(0).to(
                                           next(model.model.parameters()).device))
                    try:
                        r = generate_prediction(model, m, ds)
                    except Exception:
                        traceback.print_exc()
                        r = {"prediction": "", "n_kept": 0, "prune_ms": float("nan"),
                             "select_ms": float("nan")}
                    preds.append(r["prediction"])
                    sel_idx = pruner.last_capture.get("select_idx")
                    meta.append(dict(n_kept=r["n_kept"], prune_ms=r["prune_ms"],
                                     select_ms=r["select_ms"],
                                     select_idx=(sel_idx[0].cpu().numpy().tolist()
                                                 if sel_idx is not None else None)))
                hits = per_sample_hits(ds, b["rows"], preds)
                cal_tag = f"{cal}+shuf{args.shuffle_scores}" if args.shuffle_scores else cal
                rk = f"b{args.budget}|{sel}|{cal_tag}|{ds}"
                results["runs"][rk] = {
                    "budget": args.budget, "selector": sel, "calibration": cal_tag,
                    "dataset": ds, "pilot": bool(args.pilot),
                    "n": int(len(hits)), "acc_pct": float(np.mean(hits) * 100),
                    "sum_hits": float(np.sum(hits)),
                    "n_kept_mean": float(np.mean([x["n_kept"] for x in meta])),
                    "prune_ms_mean": float(np.nanmean([x["prune_ms"] for x in meta])),
                    "select_ms_mean": float(np.nanmean([x["select_ms"] for x in meta])),
                    "predictions": preds, "hits": [float(h) for h in hits],
                    "idx": [int(i) for i in b["idx"]],
                    "select_idx": [x["select_idx"] for x in meta],
                }
                print(f"{rk:40s} acc={results['runs'][rk]['acc_pct']:7.3f} "
                      f"select={results['runs'][rk]['select_ms_mean']:7.2f}ms")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
