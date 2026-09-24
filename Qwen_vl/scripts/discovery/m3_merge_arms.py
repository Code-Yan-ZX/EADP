"""Copy the student-independent arms of one M3 grid into another tag.

MG0, OR-* and RND-* never read the miss student -- the oracle uses the teacher
cache, the control uses a seeded permutation, and MG0 is the incumbent's own
code path.  Re-running them under a second student would be three minutes of GPU
per arm spent to reproduce a bit-identical result, so they are copied, with the
provenance recorded in the record itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUTPUT_DIR                                        # noqa: E402

INDEPENDENT = ("MG0",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="m3_accuracy")
    ap.add_argument("--dst", default="m3_accuracy_v1")
    args = ap.parse_args()
    src = json.load(open(os.path.join(OUTPUT_DIR, f"{args.src}.json")))
    dst_path = os.path.join(OUTPUT_DIR, f"{args.dst}.json")
    dst = json.load(open(dst_path)) if os.path.exists(dst_path) else dict(src)
    dst["arms"] = dst.get("arms", {})
    n = 0
    for k, v in src["arms"].items():
        if "error" in v or k in dst["arms"]:
            continue
        if k in INDEPENDENT or k.startswith(("OR-", "RND-")):
            v = dict(v)
            v["copied_from"] = f"{args.src}.json"
            v["copy_note"] = ("student-independent arm: the oracle reads the "
                              "teacher cache, the control a seeded permutation, "
                              "MG0 the incumbent's own path")
            dst["arms"][k] = v
            n += 1
    dst.setdefault("config", src.get("config", {}))
    dst.setdefault("keys", src["keys"])
    dst.setdefault("ds_order", src["ds_order"])
    dst.setdefault("primary_arm", src.get("primary_arm"))
    dst["n_instances"] = src["n_instances"]
    dst["merged_from"] = args.src + ".json"
    with open(dst_path, "w") as f:
        json.dump(dst, f, indent=1)
    print(f"[merge] {n} student-independent arms -> {args.dst}.json")


if __name__ == "__main__":
    main()
