"""Anchor Completion Validation — fresh control panel (protocol §6.3).

Per main-panel task: take the FRESH rows of the exposure manifest, group them
by image cluster (image_key), shuffle the cluster order with seed=20261002
(independently per dataset), and admit WHOLE clusters in order until at least
200 questions are included.  Fewer than 200 fresh -> all fresh; zero -> the
task is marked "无法独立比较".  Same-image questions are never trimmed.

The manifest (with SHA256) is frozen BEFORE any ablation generation.  The
ablation arms (MAIN100 / MAIN_SIM025) generate ONLY these rows; BASE/MAIN025
predictions are taken from the verified full-panel run of the same rows.

Usage: python acu_fresh.py
"""

from __future__ import annotations

import hashlib
import json
import random

import acu_common as AU

OUT = AU.OUT_DIR + "/fresh_panel_manifest.json"


def build_for_ds(ds: str, rows: dict) -> dict:
    fresh = sorted((int(i) for i, v in rows.items() if v["cls"] == "fresh"),
                   key=int)
    if not fresh:
        return dict(dataset=ds, clusters=[], n_questions=0, n_images=0,
                    status="no_fresh_unable_to_compare")
    clusters: dict[str, list[int]] = {}
    for i in fresh:
        clusters.setdefault(rows[str(i)]["image_key"], []).append(i)
    keys = sorted(clusters)
    rng = random.Random(AU.FRESH_SEED)          # independent per dataset
    rng.shuffle(keys)
    chosen, nq = [], 0
    for k in keys:
        chosen.append(k)
        nq += len(clusters[k])
        if nq >= AU.FRESH_MIN_Q:
            break
    idxs = sorted(i for k in chosen for i in clusters[k])
    return dict(dataset=ds, clusters=chosen, rows=idxs, n_questions=len(idxs),
                n_images=len(chosen),
                n_fresh_total=len(fresh), status="ok")


def main():
    man = json.load(open(AU.OUT_DIR + "/exposure_manifest.json"))
    out = dict(seed=AU.FRESH_SEED, min_questions=AU.FRESH_MIN_Q,
               rule="whole image clusters, shuffled per dataset, stop at "
                    ">=200 questions; fresh only; frozen before any "
                    "ablation generation",
               datasets={})
    for ds in AU.DS_MAIN:
        out["datasets"][ds] = build_for_ds(ds, man["datasets"][ds]["rows"])
        d = out["datasets"][ds]
        print(f"{ds}: status={d['status']} n_questions={d['n_questions']} "
              f"n_images={d['n_images']} "
              f"(fresh_total={d.get('n_fresh_total', 0)})", flush=True)
    blob = json.dumps(out, sort_keys=True).encode()
    out["sha256"] = hashlib.sha256(blob).hexdigest()
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"written: {OUT}  sha256={out['sha256'][:16]}…")


if __name__ == "__main__":
    main()
