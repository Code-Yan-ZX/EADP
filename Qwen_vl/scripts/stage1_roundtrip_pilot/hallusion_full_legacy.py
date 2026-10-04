"""FULL keep-all (K=1024) on HallusionBench under the official-legacy config.

Diagnostic for the HallBench gap question (user, 2026-10-04): our
L_R_MAIN025 K=256 aAcc (64.46) already exceeds the paper's FULL row
(51.2) while the official Qwen3-VL-8B card quotes aAcc 61.1 — a FULL
run from OUR pipeline tells which level our numbers should be compared
against and whether the K=256 result is anomalous.

Usage: python hallusion_full_legacy.py   (single A40, ~15 min)
Output: outputs/stage1_roundtrip_pilot/legacy_full/acc/FULL_HALLB/
        HallusionBench.json + _pred.tsv + _score.json
"""

from __future__ import annotations

import json
import os
import time

import rtg_common as RC
import amp_common as AC
from acu_common import common as C

OUT = os.path.join(RC.OUT_DIR, "legacy_full", "acc", "FULL_HALLB_NATIVE")
os.makedirs(OUT, exist_ok=True)
SHARD = os.path.join(OUT, "HallusionBench.json")
LEGACY = dict(deepstack=True, pos="mrope3d")


def main():
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    ds = "HallusionBench"
    dataset = AC.common.build_dataset(ds)
    eng.vlm.set_dump_image(dataset.dump_image)
    if os.path.exists(SHARD):
        shard = json.load(open(SHARD))
    else:
        shard = {"meta": dict(arm="FULL_KEEPALL_NATIVE", pipeline="official_legacy",
                              ds=ds, K=1024, legacy=LEGACY,
                              base_commit=RC.git_commit(),
                              max_new_tokens=2048), "records": {}}
    n_rows = len(dataset.data)
    t0 = time.time()
    for n, i in enumerate(range(n_rows)):
        if str(i) in shard["records"]:
            continue
        row = dataset.data.iloc[i]
        msg = C.build_message(eng.vlm, dataset, ds, row)
        timings = {}
        out = eng.generate(msg, ds, K=1024, selector="identity",
                           max_new_tokens=2048, timings=timings, **LEGACY)
        shard["records"][str(i)] = dict(
            prediction=out["text"],
            truncated=len(out["gen_ids"]) >= 2048,
            n_vis_full=int(out["meta"]["n_vis_full"]),
            ttft_ms=timings.get("ttft_ms"))
        if len(shard["records"]) % 25 == 0:
            json.dump(shard, open(SHARD + ".tmp", "w"), indent=1)
            os.replace(SHARD + ".tmp", SHARD)
        if n % 50 == 0:
            print(f"[FULL HallB] {n}/{n_rows} "
                  f"({(time.time()-t0)/max(1,n+1):.2f}s/q)", flush=True)
    json.dump(shard, open(SHARD, "w"), indent=1)

    # official scoring
    from vlmeval.dataset import build_dataset as vlmeval_build
    vds = vlmeval_build(ds)
    data = vds.data
    sub = data.copy()
    sub = sub.drop(columns=[c for c in ("image",) if c in sub.columns])
    sub["prediction"] = [shard["records"][str(i)]["prediction"]
                         for i in range(n_rows)]
    sub["truncated"] = [shard["records"][str(i)]["truncated"]
                        for i in range(n_rows)]
    tsv = SHARD.replace(".json", "_pred.tsv")
    sub.to_csv(tsv, sep="\t", index=False)
    res = vds.evaluate(tsv)
    if hasattr(res, "to_dict"):
        res = res.to_dict()
    json.dump(dict(arm="FULL_KEEPALL_NATIVE", pipeline="official_legacy", ds=ds,
                   n=n_rows, official=res, base_commit=RC.git_commit()),
              open(SHARD.replace(".json", "_score.json"), "w"),
              indent=1, default=str)
    print(f"[scored] FULL HallB: {json.dumps(res, default=str)[:200]}",
          flush=True)


if __name__ == "__main__":
    main()
