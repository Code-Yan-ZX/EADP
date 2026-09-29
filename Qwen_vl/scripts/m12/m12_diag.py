"""M12 mechanism diagnostics (explanatory only): do cheap pre-ViT scores
cover the groups the incumbent post-encoder selector (B2 = official EADP
scoring + block8 facility location) would keep?

Per DEV sample, per dataset:
  * B2 keep set at budget K (post-encoder, uses full ViT features + text);
  * for each pre-ViT score (event / graydog / sobel / variance):
      - top-K overlap recall: |topK(score) & B2K| / K
      - AUROC(score vs B2-kept labels)
  * the same for the Base+Event RetinaGate selection (recall of B2K).
These numbers explain FAILURES; they never override accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "m12")


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (Mann-Whitney U / pair count)."""
    s = scores.float()
    pos, neg = s[labels > 0], s[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).float().mean().item()
    eq = (pos[:, None] == neg[None, :]).float().mean().item()
    return gt + 0.5 * eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "m12_diag.json"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    plan = json.load(open(os.path.join(common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=32)
    from model.native_qwen3 import NativeEngine
    from model import retinagate as rg
    eng = NativeEngine(model)

    per_ds = {}
    for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench", "ChartQA_TEST"):
        rows = plan["datasets"][ds]["dev_rows"]
        step = max(1, len(rows) // args.n)
        picked = rows[::step][:args.n]
        dataset = common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        recs = []
        for idx in picked:
            row = dataset.data.iloc[idx]
            msg = common.build_message(model, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=args.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq, attn_list=None,
                       vz=None, seed=None)
            b2_keep = eng.select("b2", args.K, ctx).cpu()
            labels = torch.zeros(prep["n_vis"], dtype=torch.long)
            labels[b2_keep] = 1
            rgb = rg.pv_to_rgb_maps(prep["pv"], prep["gthw"],
                                    eng.inner.visual.spatial_merge_size,
                                    model.processor.image_processor.image_mean,
                                    model.processor.image_processor.image_std,
                                    tps=eng.inner.visual.patch_embed.temporal_patch_size)[0]
            scores = {}
            ev, _ = rg.retina_event_score(rgb, 2)
            scores["event"] = ev.reshape(-1)
            scores["graydog"] = rg._group_pool(rg.center_surround_energy(
                0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]),
                2).reshape(-1)
            scores["sobel"] = rg.sobel_score(rgb, 2).reshape(-1).cpu()
            scores["variance"] = rg.variance_score(rgb, 2).reshape(-1).cpu()
            r = dict(recall={}, auroc={})
            k = min(args.K, prep["n_vis"])
            for name, sc in scores.items():
                topk = torch.topk(sc, k).indices.cpu()
                r["recall"][name] = len(set(topk.tolist())
                                        & set(b2_keep.tolist())) / k
                r["auroc"][name] = auroc(sc, labels)
            kg, info = rg.select_groups("retinagate", args.K, prep, eng)
            r["recall"]["retinagate"] = len(set(kg.cpu().tolist())
                                            & set(b2_keep.tolist())) / k
            r["n_base"] = info.get("n_base", [None])[0]
            recs.append(r)
        agg = {}
        for name in ("event", "graydog", "sobel", "variance"):
            agg.setdefault("recall", {})[name] = \
                sum(x["recall"][name] for x in recs) / len(recs)
            agg.setdefault("auroc", {})[name] = \
                sum(x["auroc"][name] for x in recs) / len(recs)
        agg["recall"]["retinagate"] = \
            sum(x["recall"]["retinagate"] for x in recs) / len(recs)
        per_ds[ds] = dict(n=len(recs), agg=agg)
        print(f"[{ds}] n={len(recs)} "
              f"recall={ {k: round(v, 3) for k, v in agg['recall'].items()} } "
              f"auroc={ {k: round(v, 3) for k, v in agg['auroc'].items()} }",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(dict(K=args.K, per_ds=per_ds), f, indent=1)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
