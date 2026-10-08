"""NeXT (llava-v1.6 anyres) per-question actual budget manifest (N06, audit
2026-10-08).  CPU-only.

Per-question input crop geometry is computed from the image size and the
model's image_grid_pinpoints. AnyRes adds ONE global image to the local
grid crops. This is vision-input geometry, not retained LLM sequence length.

The historical tqdm logs do not contain question IDs. Their refresh counters
can repeat/skip and set_postfix runs before iterator advancement. They cannot
support an aligned per-question retained-token manifest. Parsed display
postfixes are quarantined as UNALIGNED diagnostics; missing/zero/shifted
readings must never be assigned to an image question as its actual budget.

Default outputs (outputs/audit_followup_20261008/budget_manifests_v2/):
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
OUT = "/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/budget_manifests_v2"
LOGS = os.path.join(LL, "playground/data/eval/anchorzip_p3/next_sqa")

sys.path.insert(0, LL)
from llava.mm_utils import select_best_resolution  # noqa: E402
from PIL import Image  # noqa: E402


def parse_log_vtn(path):
    """UNALIGNED last postfix per display counter, not a question index."""
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


def runtime_trace(path, questions):
    """Only ID-aligned metadata written during generation can verify budgets."""
    if not os.path.exists(path):
        return {}
    predictions = [json.loads(line) for line in open(path)]
    ids = [str(row["question_id"]) for row in predictions]
    expected = {str(q["id"]): q for q in questions}
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"runtime trace question identity mismatch: {path}")
    trace = {}
    for row in predictions:
        qid = str(row["question_id"])
        meta = row.get("metadata", {})
        value = meta.get("actual_visual_tokens_retained")
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid retained-token count: {qid}: {value}")
        if "image" in expected[qid] and value == 0:
            raise ValueError(f"image question has zero retained tokens: {qid}")
        trace[qid] = value
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="FULL_CQMI_vicuna,LRMAIN025_CQMI_vicuna,"
                                      "LRMAIN0125_CQMI_vicuna,LRMAIN00625_CQMI_vicuna")
    ap.add_argument("--out-dir", default=OUT)
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
        local_crops = int((bh // 336) * (bw // 336))
        c_cache[img_path] = {"local_crops": local_crops,
                             "global_crops": 1,
                             "total_crops": local_crops + 1,
                             "vision_input_tokens": (local_crops + 1) * 576}

    os.makedirs(args.out_dir, exist_ok=True)
    for arm in args.arms.split(","):
        log = os.path.join(LOGS, f"{arm}.log")
        if not os.path.exists(log):
            print(f"[skip] no log for {arm}")
            continue
        vtn = parse_log_vtn(log)
        trace = runtime_trace(os.path.join(LOGS, f"{arm}.jsonl"), q)
        rows = []
        for i, x in enumerate(q):
            row = {"i": i, "id": x["id"], "has_image": "image" in x}
            if "image" in x:
                info = c_cache.get(os.path.join(IMGF, x["image"]))
                if info:
                    row.update(info)
            row["actual_retained_tokens"] = trace.get(str(x["id"]))
            row["retained_tokens_status"] = ("verified: question-ID runtime trace"
                                              if str(x["id"]) in trace else
                                              "unavailable: no question-ID runtime trace")
            rows.append(row)
        img_rows = [r for r in rows if r.get("total_crops")]
        crops_dist = {}
        for r in img_rows:
            crops_dist[r["total_crops"]] = crops_dist.get(r["total_crops"], 0) + 1
        summary = {
            "arm": arm,
            "n_questions": len(q),
            "n_with_image": sum(1 for x in q if "image" in x),
            "n_geometry_verified": len(img_rows),
            "total_crops_distribution": dict(sorted(crops_dist.items())),
            "n_actual_retained_tokens_verified": len(trace),
            "actual_budget_status": ("VERIFIED" if len(trace) == len(q) else
                                      "UNVERIFIED: requires complete runtime question-ID trace"),
            "note": "AnyRes input = 1 global crop + local grid crops. Nominal "
                    "K640 = 128 x 5; importance allocation changes per-crop quotas. "
                    "Input geometry does not establish retained LLM tokens.",
        }
        out = os.path.join(args.out_dir, f"next_budget_manifest_{arm}.json")
        json.dump({"summary": summary, "per_question": rows,
                   "unaligned_log_display_postfixes": vtn,
                   "log_display_status": "UNALIGNED; not per-question measurements"},
                  open(out, "w"), indent=1)
        print(f"[manifest] {arm}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
