"""
S2-C0 step 3: accuracy translation for the forward proxies on the S2-B pilot.

Same harness as S2-B (``diag_selectors.generate_prediction``, the EADP pruner
with an externally supplied importance map, the Stage-1 generation path), with
two variables deliberately removed:

  * calibration is the identity -- the score enters the selector raw. S2-B
    established that the tested calibrations are monotone rescalings and that
    the ranking carries everything; Top-K depends only on the ranking, so a
    calibration sweep here would be a no-op dressed up as a variable.
  * selector is Top-K only. S2-B showed Top-K costs the gradient teacher ~1.4
    points against facility location, so it is not the thing being tested and
    facility-location must not leak into the proxy comparison.
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import build_dataset, build_message, eadp_model_name, ensure_out_dir, sample_indices
from diag_selectors import generate_prediction
from s1_audit import OUT
from s2b_run import GradientPruner
from scoring import per_sample_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["TextVQA_VAL", "DocVQA_VAL", "OCRBench"])
    ap.add_argument("--scores", default="s2c0_forward_proxy.npz",
                    help="npz whose keys are '<arm>__<dataset>_<index>'")
    ap.add_argument("--teacher", default="s2b_gradient_scores.npz",
                    help="npz whose keys are '<dataset>_<index>' (the P1-G2 teacher); "
                         "included as the arm named 'P1G2'")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="score-map names, e.g. A_L2 B_L1; 'P1G2' = the teacher, "
                         "'official' = EADP's own importance score (no override)")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="control: rotate the score->instance assignment by N, i.e. the "
                         "map is real but belongs to another image (S2-B's control, here "
                         "under Top-K instead of facility location)")
    ap.add_argument("--selector", default="topk")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--pilot", action="store_true", default=True)
    ap.add_argument("--tag", default="s2c0_pilot")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    Z = np.load(os.path.join(OUT, args.scores))
    T = np.load(os.path.join(OUT, args.teacher))

    # Each source has its own key set (the teacher covers the frozen 450 the proxy
    # archives do not), so the shuffle pool must be built per source.
    pool_T = sorted(T.files)
    pool_Z = sorted({k.split("__", 1)[1] for k in Z.files if "__" in k})

    def lookup(arm, key):
        """Return the importance map for this instance, or None for the official score."""
        if arm == "official":
            return None
        if args.shuffle:
            src = pool_T if arm == "P1G2" else pool_Z
            key = src[(src.index(key) + args.shuffle) % len(src)]
        return (T[key] if arm == "P1G2" else Z[f"{arm}__{key}"]).astype(np.float64)

    model = common.load_model(eadp_model_name(args.budget, 0.5, 2.0),
                              max_new_tokens=args.max_new_tokens)
    pruner = GradientPruner(
        visual_token_num=args.budget, alpha=0.5, beta=2.0,
        visual_dim=model.pruner.visual_dim,
        spatial_merge_size=model.pruner.spatial_merge_size,
        selector=args.selector, capture=True,
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
        bank[ds] = dict(dataset=dataset, idx=idx,
                        rows=[dataset.data.iloc[i] for i in idx],
                        msgs=[build_message(model, dataset, ds, dataset.data.iloc[i])
                              for i in idx])
        print(f"[bank] {ds}: {len(idx)} instances (pilot: every 3rd of 150)")

    out_path = os.path.join(ensure_out_dir(), f"{args.tag}.json")
    results = {"config": vars(args), "runs": {}}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            results["runs"] = prev.get("runs", {})
            print(f"[merge] {len(results['runs'])} existing arm-records in {out_path}")
        except Exception:
            pass

    for arm in args.arms:
        for ds in args.datasets:
            b = bank[ds]
            preds, meta = [], []
            for n, m in enumerate(tqdm(b["msgs"], desc=f"{arm}/{ds}", leave=False)):
                i = b["idx"][n]
                s = lookup(arm, f"{ds}_{i}")
                pruner.override = None if s is None else torch_from(s, model)
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
            tag = f"{arm}+shuf{args.shuffle}" if args.shuffle else arm
            rk = f"b{args.budget}|{args.selector}|{tag}|{ds}"
            results["runs"][rk] = {
                "budget": args.budget, "selector": args.selector, "calibration": tag,
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
            print(f"{rk:44s} acc={results['runs'][rk]['acc_pct']:7.3f}")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
    print(f"\n[saved] {out_path}")


def torch_from(s, model):
    import torch
    return torch.from_numpy(np.asarray(s)).float().unsqueeze(0).to(
        next(model.model.parameters()).device)


if __name__ == "__main__":
    main()
