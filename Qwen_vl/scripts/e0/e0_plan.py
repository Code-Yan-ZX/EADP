"""E0 M0 -- data partition plan (prereg docs/e0_native_baselines_prereg.md §2).

Builds the new image-disjoint DEV/CONFIRM split:
  1. candidate pool = every row of each of the 8 datasets;
  2. exclude (at IMAGE level) every image touched by any historical run:
       - every ``*_plan.json`` under outputs/discovery (any dict with ds+idx);
       - SAGE fit/val label files;
       - the evenly spaced 150-per-dataset bank (common.sample_indices(n, 150));
       - S2-A causal cases (s2a_gradient_viability.json);
  3. seeded 50/50 image-grouped DEV/CONFIRM split (E0_SEED=20260929);
  4. DEV capped at 300 questions per dataset, truncated in shuffled image order;
  5. CONFIRM is archived (indices + sha256) and never read beyond that.

Output: Qwen_vl/outputs/e0/e0_plan.json
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

DISC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/discovery"
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402  (env bootstrap, LMUData, HF offline, ...)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")
DISCOVERY_OUT = common.OUTPUT_DIR

DATASETS = [
    "TextVQA_VAL",
    "DocVQA_VAL",
    "OCRBench",
    "ChartQA_TEST",
    "MMBench_DEV_EN_V11",
    "MMStar",
    "RealWorldQA",
    "POPE",
]
OCR_DATASETS = ["TextVQA_VAL", "DocVQA_VAL", "OCRBench"]

E0_SEED = 20260929
DEV_CAP = 300

# Datasets that ever hosted a historical sample. Only these can have
# image-level exclusions; the others are fully fresh.
HISTORICAL_DATASETS = set(OCR_DATASETS)


def sha256_of(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_used_keys() -> set:
    """Every historically used (ds, row_idx) question key from the frozen list."""
    used = set()

    def add(ds, idx):
        used.add((str(ds), int(idx)))

    def walk(o):
        if isinstance(o, dict):
            if "ds" in o and "idx" in o and isinstance(o.get("idx"), int):
                add(o["ds"], o["idx"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    # 1. every *_plan.json under outputs/discovery
    plan_files = sorted(glob.glob(os.path.join(DISCOVERY_OUT, "*plan*.json")))
    for pf in plan_files:
        with open(pf) as f:
            walk(json.load(f))

    # 2. SAGE fit/val label files (s0_hits keys are "<DS>_<idx>")
    for lf in ("sage_labels_fit.json", "sage_labels_val.json"):
        p = os.path.join(DISCOVERY_OUT, lf)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            lab = json.load(f)
        for hits in (lab.get("s0_hits") or {}):
            ds, idx = hits.rsplit("_", 1)
            add(ds, idx)

    # 3. the evenly spaced 150-per-dataset bank
    for ds in HISTORICAL_DATASETS:
        n_total = bank_sizes.get(ds)
        if not n_total:
            continue
        for i in common.sample_indices(n_total, 150):
            add(ds, i)

    # 4. S2-A causal cases
    s2a = os.path.join(DISCOVERY_OUT, "s2a_gradient_viability.json")
    if os.path.exists(s2a):
        with open(s2a) as f:
            for key in json.load(f)["cases"]:
                ds, idx = key.rsplit("_", 1)
                add(ds, idx)

    return used


def image_key(ds_name: str, row) -> str:
    """Stable identity of the image behind one dataset row."""
    ip = row.get("image_path", None)
    if ip is not None and isinstance(ip, str) and ip.strip():
        return f"path:{os.path.basename(ip.strip())}"
    b64 = row.get("image", None)
    if isinstance(b64, str) and b64.startswith("/9j"):
        return "md5:" + hashlib.md5(b64.encode("ascii")).hexdigest()
    # text-only row (e.g. MMBench questions without an image): its own group
    return f"noimage:{ds_name}:{row.name if row.name is not None else id(row)}"


def main():
    from vlmeval.dataset import build_dataset as vlmeval_build

    global bank_sizes
    bank_sizes = {}
    for ds in HISTORICAL_DATASETS:
        probe = vlmeval_build(ds)
        bank_sizes[ds] = len(probe.data)
        del probe

    used = collect_used_keys()
    used_by_ds = {}
    for ds, idx in used:
        used_by_ds.setdefault(ds, set()).add(idx)

    plan = {
        "seed": E0_SEED,
        "dev_cap": DEV_CAP,
        "datasets": {},
        "exclusion_sources": {
            "plans": sorted(os.path.basename(p) for p in
                            glob.glob(os.path.join(DISCOVERY_OUT, "*plan*.json"))),
            "sage_labels": ["sage_labels_fit.json", "sage_labels_val.json"],
            "bank": "common.sample_indices(n, 150) on " + ", ".join(HISTORICAL_DATASETS),
            "s2a_causal": "s2a_gradient_viability.json:cases",
        },
        "env": {
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": __import__("torch").version.cuda,
        },
    }

    for ds in DATASETS:
        dataset = vlmeval_build(ds)
        n = len(dataset.data)

        # image-level exclusion
        excluded_images = set()
        n_excluded_rows = 0
        hist_idx = used_by_ds.get(ds, set()) & set(range(n))
        keys = []
        for i in range(n):
            keys.append(image_key(ds, dataset.data.iloc[i]))
        for i in hist_idx:
            excluded_images.add(keys[i])
        kept_rows = []
        n_excluded_rows = sum(1 for k in keys if k in excluded_images)
        for i in range(n):
            if keys[i] not in excluded_images:
                kept_rows.append((i, keys[i]))

        # image-grouped 50/50 split
        img_order, img_to_rows = [], {}
        for i, k in kept_rows:
            if k not in img_to_rows:
                img_to_rows[k] = []
                img_order.append(k)
            img_to_rows[k].append(i)
        rng = np.random.default_rng(E0_SEED + abs(hash(ds)) % 100000)
        rng.shuffle(img_order)

        half = (len(img_order) + 1) // 2
        dev_imgs, conf_imgs = img_order[:half], img_order[half:]

        dev_rows, dev_used_imgs = [], []
        for k in dev_imgs:
            if len(dev_rows) >= DEV_CAP:
                break
            rows = img_to_rows[k]
            take = rows[: max(0, DEV_CAP - len(dev_rows))]
            dev_rows.extend(take)
            dev_used_imgs.append(k)
        dev_dropped_imgs = len(dev_imgs) - len(dev_used_imgs)
        conf_rows = sorted(i for k in conf_imgs for i in img_to_rows[k])
        dev_rows = sorted(dev_rows)

        keyify = lambda rows: [f"{ds}_{i}" for i in rows]
        dev_keys, conf_keys = keyify(dev_rows), keyify(conf_rows)

        plan["datasets"][ds] = {
            "total_rows": n,
            "n_images": len(img_order) + len(excluded_images),
            "excluded_questions": len(hist_idx),
            "excluded_images": len(excluded_images),
            "excluded_rows_total": n_excluded_rows,
            "pool_rows": len(kept_rows),
            "pool_images": len(img_order),
            "dev_rows": dev_rows,
            "dev_count": len(dev_rows),
            "dev_images": len(dev_used_imgs),
            "dev_dropped_for_cap": dev_dropped_imgs,
            "confirm_rows": conf_rows,
            "confirm_count": len(conf_rows),
            "dev_sha256": sha256_of(dev_keys),
            "confirm_sha256": sha256_of(conf_keys),
        }
        print(f"[plan] {ds}: total={n} excl_imgs={len(excluded_images)} "
              f"pool={len(kept_rows)} rows / {len(img_order)} imgs -> "
              f"DEV {len(dev_rows)} (imgs {len(dev_used_imgs)}, dropped {dev_dropped_imgs}), "
              f"CONFIRM {len(conf_rows)}", flush=True)
        del dataset

    plan["global_confirm_sha256"] = sha256_of(
        {ds: plan["datasets"][ds]["confirm_sha256"] for ds in DATASETS})
    plan["global_dev_sha256"] = sha256_of(
        {ds: plan["datasets"][ds]["dev_sha256"] for ds in DATASETS})

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "e0_plan.json")
    with open(out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"[saved] {out}")
    print(f"global dev sha256 {plan['global_dev_sha256']}")
    print(f"global confirm sha256 {plan['global_confirm_sha256']}")


if __name__ == "__main__":
    main()
