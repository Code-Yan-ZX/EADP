"""
S3-B side check: is the crop probe's "whole frame" arm low because of the
pixels, or because of the position policy?

Every S3-B arm (and every pruned arm of the incumbent, see
``Qwen3VLChatFixedRes._build_pruned_inputs`` -> ``generate(inputs_embeds=...,
attention_mask=...)`` with no ``input_ids``/``position_ids``) decodes with
FLAT sequential positions: HF cannot run mrope without ``input_ids``, so the
2-D structure of the image block is gone. The frozen B0 baseline instead goes
through the ``input_ids`` path and keeps mrope positions. The crop probe's
``whole`` arm (all 1024 crop tokens, splice path) scored 0.525 where frozen B0
scores 0.787 on the same instances -- this separates the two explanations.

Three measurements per instance, same tokens, same decoder:
    splice1024      all 1024 merged tokens through the splice path, using the
                    stage's CACHED vision embeddings (what every S3-B arm uses)
    splice1024_png  the same frame re-encoded to PNG and reprocessed from
                    scratch -- the crop probe's exact input path
    eadp256_engine  the wrapper's own generate() -- which for this model name is
                    the INCUMBENT EADP-256 pruned decode (input_ids + grid_thw,
                    so mrope positions and EADP's own selection). It is NOT the
                    unpruned baseline B0; the label in the JSON is historical.

Result (n=10 panel instances): splice 0.80, splice+png 0.76, eadp256_engine
0.55; and the PNG re-encode reproduces the cached-prep generation in only
7/10 cases (max |dL| 5.7 nats), because the wrapper's own resize chain
(process_vision_info smart-resize -> expand2square(mean 127) -> 1024) is not
the same interpolation path as canonical_frame's. Conclusion recorded in
docs/scoring_search_s3b_bundles.md section 4.4: the crop probe is only valid
WITHIN its own path, and its absolute rates must not be compared to B0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                                    # noqa: E402
from s1_audit import OUT                                         # noqa: E402
import s3b_common as B                                           # noqa: E402
import s3b_gpu as G                                              # noqa: E402
import s3b_crop as K                                             # noqa: E402
from scoring import per_sample_hits                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    cases = json.load(open(os.path.join(OUT, B.CASES_JSON)))
    crop = json.load(open(os.path.join(OUT, "s3b_crop.json")))
    recs = {b: {r["key"]: r for r in cases["banks"][b]} for b in ("G", "L")}
    keys = sorted({(e["bank"], e["key"]) for e in crop["runs"].values()})
    rng = np.random.default_rng(7)
    pick = [keys[i] for i in
            rng.choice(len(keys), size=min(args.n, len(keys)), replace=False)]

    datasets = {ds: common.build_dataset(ds) for ds in B.DS_ALL}
    model, h = G.load_stack()
    rows = []
    for bank, key in pick:
        rec = recs[bank][key]
        ds_obj = datasets[rec["ds"]]
        model.set_dump_image(ds_obj.dump_image)
        row = ds_obj.data.iloc[rec["idx"]]
        msg = common.build_message(model, ds_obj, rec["ds"], row)
        golds = B.A.golds_of(row)
        ans = h.answer_ids(golds)
        prep = h.prep(msg, rec["ds"], rec["key"])            # cached vis
        keep = list(range(B.N_VIS))
        try:
            L1 = h.nll(prep, keep, ans)[0]
            _, t1 = h.generate(prep, keep, max_new=G.GEN_CAP)
            p1 = h.vlm._post_process_response(t1)
            # identical pixels, but re-encoded and re-loaded (crop path)
            frame = K.canonical_frame(datasets, rec)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                frame.save(tf.name)
                prep2 = K.prep_nocache(h, K.swap_image(msg, tf.name), rec["ds"])
            os.unlink(tf.name)
            L2 = h.nll(prep2, keep, ans)[0]
            _, t2 = h.generate(prep2, keep, max_new=G.GEN_CAP)
            p2 = h.vlm._post_process_response(t2)
            # the official unpruned engine path (mrope positions)
            p3 = model.generate(msg, dataset=rec["ds"])
            hits = [float(per_sample_hits(rec["ds"], [row], [x])[0])
                    for x in (p1, p2, p3)]
            rows.append(dict(key=key, bank=bank, ds=rec["ds"],
                             L_splice=L1, L_splice_png=L2,
                             hit_splice=hits[0], hit_splice_png=hits[1],
                             hit_engine=hits[2],
                             text_match_png=bool(p1 == p2),
                             pred_splice=p1[:44], pred_engine=p3[:44]))
            print(f"[{bank}|{key}] splice {hits[0]:.2f} (L {L1:.3f})  "
                  f"splice+png {hits[1]:.2f} (L {L2:.3f})  "
                  f"engine/mrope {hits[2]:.2f}   text_equal={rows[-1]['text_match_png']}")
            print(f"     splice: {p1[:52]!r}")
            print(f"     engine: {p3[:52]!r}")
        except Exception:
            traceback.print_exc()
            continue
    out = dict(
        n=len(rows),
        mean_hit_splice=float(np.mean([r["hit_splice"] for r in rows])),
        mean_hit_splice_png=float(np.mean([r["hit_splice_png"] for r in rows])),
        mean_hit_engine=float(np.mean([r["hit_engine"] for r in rows])),
        max_abs_dL_splice_vs_png=float(
            np.max(np.abs(np.array([r["L_splice"] for r in rows])
                          - np.array([r["L_splice_png"] for r in rows])))),
        png_text_equal=float(np.mean([r["text_match_png"] for r in rows])),
        rows=rows)
    path = os.path.join(OUT, "s3b_poscheck.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
