"""L_E_GATHER full-panel generation + scoring (official-legacy scenario).

User-directed round (2026-10-04, P1 of the AnchorZip paper round): the
EADP comparison arm E_GATHER (official EADP hard pruning: official
importance + official facility keep, NO Completion, lam=0) run under the
SAME official-legacy scenario as L_R_MAIN025 — DeepStack OFF, 1-D
positions, fixed-res 1024, greedy — for the 7 Table-4 datasets whose old
official predictions were NOT recovered (ChartQA_TEST, AI2D_TEST,
HallusionBench, MME, MMBench_DEV_EN_V11, MMBench_DEV_CN_V11,
InfoVQA_VAL).  TextVQA/DocVQA/OCRBench reuse the audit-recovered official
predictions (audit_base_gap_20261003) and are NOT regenerated here.

Deltas vs rtg_accuracy_legacy.py (frozen file, not modified):
  * cfg = acu_common.arm_cfg("BASE")  (kind="base", lam=0.0 — no merge)
  * bank = ACV b1 banks (outputs/anchor_completion_validation/
    bank_full_<ds>.json.gz; built by acu_bank.py, official EADP
    importance + official facility keep).  Bank building touches only
    prepare/encode (ViT), so records are engine-mode independent
    (same argument as the RTG banks; bitwise record re-check in
    rtg_legacy_checks.py C1).
  * output root  outputs/stage1_roundtrip_pilot/legacy_full/acc/L_E_GATHER/
  * arm/pipeline tags "L_E_GATHER" / "official_legacy".

Scoring: mirrors rtg_accuracy_legacy._score_one for the 5 classic
datasets and reuses rtg_score_new5.score_one for the 5 specialised
datasets by pinning rtg_accuracy_legacy.ARM/FULL BEFORE importing
rtg_score_new5 (its copied ARM string is re-pinned after import).

Usage: python egather_legacy.py --mode gen  [--datasets ...] [--limit N]
       python egather_legacy.py --mode score [--datasets ...]
"""

from __future__ import annotations

import argparse
import os
import time

import rtg_common as RC            # first: installs ACV dir on sys.path
import acu_common as AU
import amp_common as AC
import rtg_accuracy_legacy as RAL
from acu_common import common as C
from acu_common import degeneracy, run_one as acu_run_one

ARM = "L_E_GATHER"
PIPELINE = "official_legacy"
LEGACY_FLAGS = dict(deepstack=False, pos="1d")

# Table-4 datasets needing E_GATHER generation (old official predictions
# not recovered).  Banks for the last 5 are built by acu_bank.py first.
EGATHER_DS = ["ChartQA_TEST", "AI2D_TEST", "HallusionBench", "MME",
              "MMBench_DEV_EN_V11", "MMBench_DEV_CN_V11", "InfoVQA_VAL"]

# P4 ablation panel (pre-fixed): the 4 accuracy-panel datasets
ABLATION_DS = ["TextVQA_VAL", "ChartQA_TEST", "DocVQA_VAL", "OCRBench"]

# arm spec: name -> (cfg_name, bank_kind, datasets)
#   L_E_GATHER   EADP b1 keep, no Completion (this file's original arm)
#   L_E_MAIN025  EADP b1 keep + Completion 0.25 (P4 ablation)
#   L_R_GATHER   RTG keep, no Completion (P4 ablation)
#   L_R_MAIN025  RTG keep + Completion 0.25 (frozen method; other script)
ARM_SPECS = {
    "L_E_GATHER": ("BASE", "acv", EGATHER_DS),
    "L_E_MAIN025": ("MAIN025", "acv", ABLATION_DS),
    "L_R_GATHER": ("BASE", "rtg", ABLATION_DS),
}

# ACV bank files live under the ACV output root, not RC.OUT_DIR.
ACV_ROOT = os.path.join(os.path.dirname(RC.OUT_DIR),
                        "anchor_completion_validation")

FULL = os.path.join(RC.OUT_DIR, "legacy_full", "acc", ARM)
os.makedirs(FULL, exist_ok=True)
FAIL_LOG = os.path.join(RC.OUT_DIR, "legacy_full",
                        "score_failures_egather.jsonl")


def set_arm(arm: str) -> None:
    """Retarget module-level path/arm constants (called pre-import of the
    scoring modules for the P4 ablation arms)."""
    global ARM, FULL, FAIL_LOG
    ARM = arm
    FULL = os.path.join(RC.OUT_DIR, "legacy_full", "acc", arm)
    os.makedirs(FULL, exist_ok=True)
    FAIL_LOG = os.path.join(RC.OUT_DIR, "legacy_full",
                            f"score_failures_{arm}.jsonl")
    RAL.ARM = ARM
    RAL.FULL = FULL
    RAL.FAIL_LOG = FAIL_LOG

# Pin the frozen scoring modules onto this arm's paths/arm name BEFORE
# rtg_score_new5 imports (it copies the ARM string; shard_path/fail/
# load_shard resolve FULL/ARM at call time).
RAL.ARM = ARM
RAL.FULL = FULL
RAL.FAIL_LOG = FAIL_LOG


def _fail(ds, kind, detail):
    import json
    entry = dict(arm=ARM, ds=ds, kind=kind, detail=str(detail))
    with open(FAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[score FAIL] {ARM} {ds}: {kind}: {detail}", flush=True)
    return False


def shard_path(ds: str) -> str:
    return os.path.join(FULL, f"{ds}.json")


def load_shard(path: str) -> dict:
    import json
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"records": {}}


def save_shard(path: str, shard: dict) -> None:
    import json
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=1)
    os.replace(tmp, path)


def run_gen(datasets=None, limit=None, arm=None):
    if arm is not None and arm != ARM:
        set_arm(arm)
    cfg_name, bank_kind, default_ds = ARM_SPECS[ARM]
    cfg = AU.arm_cfg(cfg_name)
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    assert (cfg["kind"] == "base") == (ARM.endswith("GATHER"))
    for ds in (datasets or default_ds):
        if bank_kind == "acv":
            bank = AU.load_bank(ds)
            bank_sha = AU.bank_sha256(ds)
            bank_from = "anchor_completion_validation/bank_full"
        else:
            import gzip
            import json
            bp = os.path.join(RC.OUT_DIR, "full",
                              f"bank_full_rtg_{ds}.json.gz")
            with gzip.open(bp, "rt") as f:
                loaded = json.load(f)
            bank = loaded["samples"] if "samples" in loaded else loaded
            bank_sha = RC.sha256_file(bp)
            bank_from = "stage1_roundtrip_pilot/full/bank_full_rtg"
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        path = shard_path(ds)
        shard = load_shard(path)
        if "meta" not in shard:
            shard["meta"] = dict(
                arm=ARM, pipeline=PIPELINE, panel="full", ds=ds,
                K=RC.K, legacy=LEGACY_FLAGS,
                bank_sha256=bank_sha, bank_reused_from=bank_from,
                cfg=dict(kind=cfg["kind"], lam=cfg.get("lam", 0.0),
                         selector=cfg.get("selector"), K=cfg.get("K")),
                base_commit=RC.git_commit(), max_new_tokens=2048)
            save_shard(path, shard)
        n_rows = len(dataset.data)
        if limit is not None:
            n_rows = min(n_rows, limit)
        t0 = time.time()
        n_done = 0
        for n, i in enumerate(range(n_rows)):
            key = str(i)
            if key in shard["records"]:
                n_done += 1
                continue
            if key not in bank:
                raise RuntimeError(
                    f"{ARM} {ds}: bank record missing for key {key} "
                    f"(build it with acu_bank.py first)")
            row = dataset.data.iloc[i]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            out = acu_run_one(eng, msg, ds, bank[key], cfg,
                              max_new_tokens=2048, timings=timings,
                              **LEGACY_FLAGS)
            n_kept = int(out["state"].keep_idx.numel())
            shard["records"][key] = dict(
                prediction=out["text"],
                truncated=len(out["gen_ids"]) >= 2048,
                n_vis_kept=n_kept,
                ttft_ms=timings.get("ttft_ms"),
                merge_ms=timings.get("merge_ms"),
                layer_calls_ok=bool(out["meta"]["layer_calls_ok"]),
                degeneracy=degeneracy(out["text"]))
            n_done += 1
            if n_done % 25 == 0:
                save_shard(path, shard)
            if n % 50 == 0:
                el = time.time() - t0
                print(f"[{ARM} full {ds}] {n}/{n_rows} "
                      f"({el/max(1, n_done):.2f}s/q)", flush=True)
        save_shard(path, shard)
        print(f"[done] {ARM} full {ds}: {len(shard['records'])} records",
              flush=True)


def run_score(datasets=None, arm=None):
    if arm is not None and arm != ARM:
        set_arm(arm)
    ok = True
    dsets = datasets or ARM_SPECS[ARM][2]
    classic = [d for d in dsets if d not in
               ("AI2D_TEST", "HallusionBench", "MME",
                "MMBench_DEV_CN_V11", "InfoVQA_VAL")]
    new5 = [d for d in dsets if d in
            ("AI2D_TEST", "HallusionBench", "MME",
             "MMBench_DEV_CN_V11", "InfoVQA_VAL")]
    for ds in classic:
        ok &= RAL._score_one(ds)
    if new5:
        import rtg_score_new5 as N5
        N5.ARM = ARM          # re-pin the copied arm string
        for ds in new5:
            ok &= N5.score_one(ds)
    if not ok:
        raise SystemExit("E_GATHER legacy scoring failures recorded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["gen", "score",
                                                       "both"])
    ap.add_argument("--arm", default="L_E_GATHER",
                    choices=sorted(ARM_SPECS),
                    help="L_E_GATHER / L_E_MAIN025 / L_R_GATHER")
    ap.add_argument("--datasets", default=None,
                    help="comma list; default = arm's dataset set")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke: cap rows per dataset (gen only)")
    args = ap.parse_args()
    dsets = args.datasets.split(",") if args.datasets else None
    if args.mode in ("gen", "both"):
        run_gen(dsets, args.limit, arm=args.arm)
    if args.mode in ("score", "both"):
        run_score(dsets, arm=args.arm)


if __name__ == "__main__":
    main()
