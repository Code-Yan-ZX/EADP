"""NeXT (llava-v1.6 anyres) per-question actual budget manifest (N06, audit
2026-10-08).  CPU-only.

For each SQA CQMI arm log we parse the tqdm fragments' per-question
``vtn=<n>`` postfix (the ACTUAL visual tokens kept, as reported by the model)
and pair it with the per-question crop count C and patch count N computed
offline from the image size + the model config's image_grid_pinpoints via
LLaVA's own select_best_resolution.

Outputs (audit_rescore_20261008/rescored/):
  next_budget_manifest_<arm>.json  {per_question: [...], summary: {...}}
"""
import argparse
import json
import os
import re
import sys

LL = "/media/disk2/YZX/research/EADP_amp/LLaVA"
IMGF = os.path.join(LL, "playground/data/eval/scienceqa/test")
QFILE = os.path.join(LL, "playground/data/eval/scienceqa/llava_test_CQM-I.json")
MODEL = "/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b"
OUT = "/media/disk2/YZX/research/audit_rescore_20261008/rescored"
LOGS = os.path.join(LL, "playground/data/eval/anchorzip_p3/next_sqa")

sys.path.insert(0, LL)
from llava.mm_utils import select_best_resolution  # noqa: E402
from PIL import Image  # noqa: E402


def parse_log_vtn(path):
    """Last vtn reading per tqdm iteration index (1-based)."""
    raw = open(path, errors="replace").read()
    frag = re.split(r"[\r\n]", raw)
    per_it = {}
    pat = re.compile(r"\|\s*(\d+)/(\d+)\s*\[.*?vtn=(\d+)")
    for f in frag:
        m = pat.search(f)
        if m:
            it, total, vtn = int(m.group(1)), int(m.group(2)), int(m.group(3))
            per_it[it] = vtn  # later fragments overwrite: last reading wins
    return per_it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="FULL_CQMI_vicuna,LRMAIN025_CQMI_vicuna,"
                                      "LRMAIN0125_CQMI_vicuna,LRMAIN00625_CQMI_vicuna")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    pinpoints = cfg["image_grid_pinpoints"]
    base = cfg.get("image_aspect_ratio", "anyres")
    print(f"grid pinpoints: {pinpoints}")

    q = json.load(open(QFILE))
    c_cache = {}
    for x in q:
        if "image" not in x:
            continue
        img_path = os.path.join(IMGF, x["image"])
        if img_path in c_cache:
            continue
        try:
            sz = Image.open(img_path).size  # (w, h)
        except Exception:
            c_cache[img_path] = None
            continue
        best = select_best_resolution(sz, pinpoints)  # (h, w) grid point
        if best is None:
            c_cache[img_path] = None
            continue
        bh, bw = best
        c = int((bh // 336) * (bw // 336))
        c_cache[img_path] = {"crops": c, "patches": c * 576}

    os.makedirs(OUT, exist_ok=True)
    for arm in args.arms.split(","):
        log = os.path.join(LOGS, f"{arm}.log")
        if not os.path.exists(log):
            print(f"[skip] no log for {arm}")
            continue
        vtn = parse_log_vtn(log)
        rows, missing_vtn = [], 0
        for i, x in enumerate(q):
            row = {"i": i, "id": x["id"], "has_image": "image" in x}
            if "image" in x:
                info = c_cache.get(os.path.join(IMGF, x["image"]))
                if info:
                    row.update(info)
            if (i + 1) in vtn:
                row["vtn_logged"] = vtn[i + 1]
            else:
                missing_vtn += 1
            rows.append(row)
        have = [r for r in rows if "vtn_logged" in r]
        img_rows = [r for r in have if r.get("crops")]
        crops_dist = {}
        for r in img_rows:
            crops_dist[r["crops"]] = crops_dist.get(r["crops"], 0) + 1
        vtn_vals = [r["vtn_logged"] for r in img_rows]
        summary = {
            "arm": arm,
            "n_questions": len(q),
            "n_with_image": sum(1 for x in q if "image" in x),
            "n_vtn_parsed": len(have),
            "n_vtn_missing": missing_vtn,
            "crops_distribution": dict(sorted(crops_dist.items())),
            "vtn_min": min(vtn_vals) if vtn_vals else None,
            "vtn_max": max(vtn_vals) if vtn_vals else None,
            "vtn_mean": round(sum(vtn_vals) / len(vtn_vals), 1) if vtn_vals else None,
            "note": "vtn_logged = model-reported visual tokens kept "
                    "(tqdm postfix, last reading per question); crops/patches "
                    "computed offline from image size + config grid pinpoints "
                    "(select_best_resolution).  Nominal K640 = 128/crop x 5.",
        }
        out = os.path.join(OUT, f"next_budget_manifest_{arm}.json")
        json.dump({"summary": summary, "per_question": rows},
                  open(out, "w"), indent=1)
        print(f"[manifest] {arm}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
