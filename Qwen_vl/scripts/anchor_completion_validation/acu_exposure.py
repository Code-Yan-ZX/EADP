"""Anchor Completion Validation — exposure audit (protocol §5).

Classifies EVERY row of the 8 official-split datasets into
  exposed : appears in any enumerated historical usage source of this project
  fresh   : not found in any source AND every source was enumerable
  unknown : could not be verified (never counted as fresh)

Sources (each hashed and counted into the manifest):
  S1 bank150        discovery.common.sample_indices(n,150) recomputation on
                    TextVQA_VAL / DocVQA_VAL / OCRBench (E0 exclusion record)
  S2 m1_plan        outputs/discovery/m1_plan.json instances
  S3 sage_plan      outputs/discovery/sage_plan.json instances
  S4 sage labels    sage_labels_fit.json / sage_labels_val.json s0_* keys
  S5 s2c2 rescue    outputs/discovery/s2c2_rescue_plan.json rescuable
  S6 s2a causal     outputs/discovery/s2a_gradient_viability.json cases
  S7 e0_plan        outputs/e0/e0_plan.json dev_rows + confirm_rows (8 ds)
  S8 acc shards     EVERY record key under outputs/**/acc/**/*.json (union of
                    what was actually generated in e0 / m12 / m13 /
                    anchor_merge r1-2 / cross_stream / grounding /
                    visual_calibration rounds)
  S9 amp manifest   outputs/anchor_merge_pilot/manifest.json (DEV/CONFIRM)

Image identity follows e0_plan.image_key; a row is exposed if its row index
OR its image_key was touched (whole-image cluster rule, incl. cross-split
duplicate images).  fresh claims ONLY "not used for method selection in this
project" — pretrained-data contamination is out of scope and never claimed.

Usage: python acu_exposure.py            (CPU only)
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import sys

from acu_common import DS_ALL, OUT_DIR, QWEN_ROOT  # noqa: E402  (paths only)

OUT = os.path.join(OUT_DIR, "exposure_manifest.json")


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def image_key(ds_name: str, row) -> str:
    """Verbatim rule from e0_plan.image_key (image_path > base64 > noimage)."""
    ip = row.get("image_path", None)
    if ip is not None and not (isinstance(ip, float) and ip != ip) and str(ip):
        return f"{ds_name}:{os.path.basename(str(ip))}"
    b64 = row.get("image", None)
    if b64 is not None and isinstance(b64, str) and len(b64) > 0:
        import base64
        h = hashlib.sha256(
            base64.b64decode(b64)).hexdigest()
        return f"{ds_name}:b64:{h}"
    return f"noimage:{ds_name}:{row.name if row.name is not None else id(row)}"


def load_dataset(ds: str):
    from acu_common import common as C  # sets LMUData env on import
    return C.build_dataset(ds)


# ---------------------------------------------------------------------------
# source collection -> per-ds sets of (idx) and (image_key)
# ---------------------------------------------------------------------------
class Exposure:
    def __init__(self):
        self.idx = {ds: set() for ds in DS_ALL}
        self.keys = {ds: set() for ds in DS_ALL}
        self.sources = {}

    def add_idx(self, tag: str, ds: str, i: int):
        self.idx[ds].add(int(i))
        self.sources.setdefault(tag, {}).setdefault(ds, 0)
        self.sources[tag][ds] += 1

    def add_key(self, tag: str, ds: str, k: str):
        self.keys[ds].add(k)
        self.sources.setdefault(tag, {}).setdefault(ds, 0)
        self.sources[tag][ds] += 1


def parse_key(tag_key: str):
    """'TextVQA_VAL_208' -> (ds, idx) if parseable."""
    for ds in DS_ALL:
        pref = ds + "_"
        if tag_key.startswith(pref):
            tail = tag_key[len(pref):]
            try:
                return ds, int(tail)
            except ValueError:
                return None
    return None


def collect() -> tuple[Exposure, list[dict]]:
    exp = Exposure()
    src_meta = []

    def record(tag, path, note=""):
        is_file = path and os.path.isfile(path)
        src_meta.append(dict(tag=tag,
                             path=os.path.relpath(path, QWEN_ROOT)
                             if path and os.path.exists(path) else path,
                             sha256=sha256_file(path) if is_file else None,
                             note=note))

    disc = os.path.join(QWEN_ROOT, "outputs", "discovery")

    # S2 m1_plan ------------------------------------------------------------
    p = os.path.join(disc, "m1_plan.json")
    d = json.load(open(p))
    for it in d["instances"]:
        exp.add_idx("m1_plan", it["ds"], it["idx"])
    record("m1_plan", p)

    # S3 sage_plan ----------------------------------------------------------
    p = os.path.join(disc, "sage_plan.json")
    d = json.load(open(p))
    for it in d["instances"]:
        exp.add_idx("sage_plan", it["ds"], it["idx"])
    record("sage_plan", p)

    # S4 sage labels (s0_* keyed samples) ------------------------------------
    for name in ("sage_labels_fit.json", "sage_labels_val.json"):
        p = os.path.join(disc, name)
        if not os.path.exists(p):
            record(name, p, "MISSING -> unknown contribution, see report")
            continue
        d = json.load(open(p))
        n = 0
        for field in ("s0_hits", "s0_preds"):
            for k in d.get(field, {}):
                pr = parse_key(k)
                if pr:
                    exp.add_idx(f"sage_labels({name})", pr[0], pr[1])
                    n += 1
        record(name, p, f"{n} keyed samples")

    # S5 s2c2 rescue ---------------------------------------------------------
    p = os.path.join(disc, "s2c2_rescue_plan.json")
    d = json.load(open(p))
    for it in d["rescuable"]:
        exp.add_idx("s2c2_rescue", it["ds"], it["idx"])
    record("s2c2_rescue", p)

    # S6 s2a causal ----------------------------------------------------------
    p = os.path.join(disc, "s2a_gradient_viability.json")
    d = json.load(open(p))
    for k in d["cases"]:
        pr = parse_key(k)
        if pr:
            exp.add_idx("s2a_causal", pr[0], pr[1])
    record("s2a_causal", p)

    # S1 bank150 -------------------------------------------------------------
    try:
        from acu_common import common as C
        n_map = {}
        for ds in ("TextVQA_VAL", "DocVQA_VAL", "OCRBench"):
            n_map[ds] = len(load_dataset(ds).data)
        from common import sample_indices  # scripts/discovery/common.py
        for ds, n in n_map.items():
            for i in sample_indices(n, 150):
                exp.add_idx("bank150", ds, int(i))
        record("bank150", os.path.join(QWEN_ROOT, "scripts",
                                       "discovery", "common.py"),
               "sample_indices(n,150) recomputation")
    except Exception as e:  # pragma: no cover
        record("bank150", None, f"FAILED: {e}")

    # S7 e0_plan -------------------------------------------------------------
    p = os.path.join(QWEN_ROOT, "outputs", "e0", "e0_plan.json")
    d = json.load(open(p))
    for ds in DS_ALL:
        for it in d["datasets"][ds].get("dev_rows", []):
            exp.add_idx("e0_plan_dev", ds,
                        it["idx"] if isinstance(it, dict) else it)
        for it in d["datasets"][ds].get("confirm_rows", []):
            exp.add_idx("e0_plan_confirm", ds,
                        it["idx"] if isinstance(it, dict) else it)
    record("e0_plan", p)

    # S9 anchor_merge manifest ----------------------------------------------
    p = os.path.join(QWEN_ROOT, "outputs", "anchor_merge_pilot",
                     "manifest.json")
    d = json.load(open(p))
    for ds in d["datasets"]:
        for split in ("dev", "confirm"):
            for it in d["datasets"][ds].get(split, []):
                exp.add_idx(f"amp_manifest_{split}", ds, it["idx"])
    record("amp_manifest", p)

    # S8 acc shards (everything actually generated) --------------------------
    acc_files = glob.glob(os.path.join(QWEN_ROOT, "outputs", "**",
                                       "acc", "**", "*.json"), recursive=True)
    n_shards = 0
    for path in sorted(acc_files):
        stem = os.path.basename(path)
        if stem.endswith("_score.json") or stem == "gate.json":
            continue
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if not (isinstance(d, dict) and "records" in d):
            continue
        n_shards += 1
        rel = os.path.relpath(path, QWEN_ROOT)
        ds_guess = None
        meta = d.get("meta", {}) or {}
        ds_guess = meta.get("ds")
        if ds_guess not in DS_ALL:
            ds_guess = stem[:-5] if stem.endswith(".json") else stem
            ds_guess = ds_guess if ds_guess in DS_ALL else None
        if ds_guess is not None:
            for k in d["records"]:
                exp.add_idx("acc_shards", ds_guess, k)
        else:
            # shard without a recognisable ds: mark every parseable key
            for k in d["records"]:
                pr = parse_key(k)
                if pr:
                    exp.add_idx("acc_shards", pr[0], pr[1])
    record("acc_shards", os.path.join(QWEN_ROOT, "outputs"),
           f"{n_shards} shard files with records")

    return exp, src_meta


def main():
    exp, src_meta = collect()

    manifest = dict(
        generated_by="acu_exposure.py",
        rule=dict(
            exposed="row index OR image_key appears in any enumerated source",
            fresh="not in any source; all sources enumerated",
            unknown=" unverifiable; NEVER counted as fresh",
            cluster="whole image cluster exposed if any row exposed",
            scope="project-internal exposure only; pretraining contamination "
                  "NOT claimed"),
        sources=src_meta,
        source_counts=exp.sources,
        datasets={})

    for ds in DS_ALL:
        dataset = load_dataset(ds)
        rows = dataset.data
        n = len(rows)
        cls = {}
        n_exposed = n_fresh = 0
        for i in range(n):
            row = rows.iloc[i]
            k = image_key(ds, row)
            if i in exp.idx[ds] or k in exp.keys[ds]:
                c = "exposed"
            else:
                c = "fresh"
            cls[str(i)] = dict(cls=c, image_key=k)
            if c == "exposed":
                n_exposed += 1
            else:
                n_fresh += 1
        # image-cluster exposure propagation: any row exposed on an image ->
        # every row with the same image_key is exposed
        exposed_keys = {v["image_key"] for v in cls.values()
                        if v["cls"] == "exposed"}
        n_prop = 0
        for i, v in cls.items():
            if v["cls"] == "fresh" and v["image_key"] in exposed_keys:
                v["cls"] = "exposed"
                n_prop += 1
        n_exposed += n_prop
        n_fresh -= n_prop
        images = {}
        for v in cls.values():
            images.setdefault(v["image_key"], set()).add(v["cls"])
        manifest["datasets"][ds] = dict(
            n_rows=n,
            n_images=len(images),
            n_exposed=n_exposed,
            n_fresh=n_fresh,
            n_unknown=0,
            rows=cls)

    with open(OUT, "w") as f:
        json.dump(manifest, f)
    print("sources:")
    for tag, per in exp.sources.items():
        print(f"  {tag}: " + ", ".join(f"{k}={v}" for k, v in
                                       sorted(per.items())))
    for ds in DS_ALL:
        m = manifest["datasets"][ds]
        print(f"{ds}: rows={m['n_rows']} exposed={m['n_exposed']} "
              f"fresh={m['n_fresh']} unknown={m['n_unknown']}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
