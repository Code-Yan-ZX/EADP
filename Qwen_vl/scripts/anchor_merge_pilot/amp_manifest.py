"""Anchor-Merge Pilot — frozen data manifests (protocol §4).

DEV  : first 100 rows of e0_plan dev_rows per dataset (prefix rule, frozen
       before any of this round's scores).
CONF : e0_plan confirm_rows prefix up to 200; OCRBench's confirm pool has only
       164 rows, so it is topped up to 200 with rows from the UNUSED dev suffix
       (dev_rows[100:]), first-come in plan order, enforcing image-disjointness
       against this round's DEV-100 by the same image-key rule as e0_plan.

Writes outputs/anchor_merge_pilot/manifest.json (+ sha256 per list).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import amp_common as AC


def image_key(dataset, i: int) -> str:
    """Same image-identity rule as e0_analyze.image_keys / e0_plan."""
    row = dataset.data.iloc[int(i)]
    ip = row.get("image_path", None)
    if ip is not None and isinstance(ip, str) and ip.strip():
        return os.path.basename(ip.strip())
    b = row.get("image", None)
    if isinstance(b, str) and b.startswith("/9j"):
        return hashlib.md5(b.encode("ascii")).hexdigest()
    return f"n{int(i)}"


def sha256_of(keys) -> str:
    return hashlib.sha256(
        json.dumps(sorted(map(str, keys)), sort_keys=True).encode()).hexdigest()


def main():
    plan = json.load(open(os.path.join(AC.common.QWEN_ROOT, "outputs", "e0",
                                       "e0_plan.json")))
    manifest = dict(seed=plan["seed"], dev_n=100, confirm_n=200,
                    base_commit=repo_commit(), datasets={})

    for ds in AC.DS_LIST:
        d = plan["datasets"][ds]
        dataset = AC.common.build_dataset(ds)

        dev_rows = list(d["dev_rows"][:manifest["dev_n"]])
        dev = [dict(idx=int(i), image_key=image_key(dataset, i))
               for i in dev_rows]

        confirm_rows = list(d["confirm_rows"])
        confirm = [dict(idx=int(i), image_key=image_key(dataset, i))
                   for i in confirm_rows[:manifest["confirm_n"]]]
        dev_keys = {r["image_key"] for r in dev}
        topup = []
        if len(confirm) < manifest["confirm_n"]:
            # protocol §4: fill from the UNUSED dev suffix, image-disjoint
            # against this round's DEV-100, plan order, frozen before scores
            for i in d["dev_rows"][manifest["dev_n"]:]:
                k = image_key(dataset, i)
                if k in dev_keys:
                    continue
                topup.append(dict(idx=int(i), image_key=k, source="dev_unused"))
                if len(confirm) + len(topup) >= manifest["confirm_n"]:
                    break
        confirm += topup

        # image-disjointness between this round's DEV and CONF (hard assert)
        conf_keys = [r["image_key"] for r in confirm]
        overlap = dev_keys & set(conf_keys)
        assert not overlap, f"{ds}: DEV/CONF image overlap {len(overlap)}"

        manifest["datasets"][ds] = dict(
            dev=dev, confirm=confirm,
            dev_sha256=sha256_of([r["image_key"] for r in dev]),
            confirm_sha256=sha256_of(conf_keys),
            confirm_topup_from_dev=len(topup),
            pool_confirm_available=len(d["confirm_rows"]),
            note=("OCRBench confirm pool < 200; topped from unused dev suffix"
                  if topup else ""))

    out = os.path.join(AC.OUT_DIR, "manifest.json")
    with open(out + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(out + ".tmp", out)
    for ds, m in manifest["datasets"].items():
        print(f"[{ds}] dev={len(m['dev'])} confirm={len(m['confirm'])} "
              f"(topup {m['confirm_topup_from_dev']}) "
              f"dev_sha={m['dev_sha256'][:12]} conf_sha={m['confirm_sha256'][:12]}")
    print(f"[saved] {out}")


def repo_commit() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=AC.common.QWEN_ROOT,
                          capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    main()
