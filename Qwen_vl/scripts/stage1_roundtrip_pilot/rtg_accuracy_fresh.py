"""P1: RTG arms (R_GATHER / F_MAIN025 / S_MAIN025) on the frozen fresh
panel (TextVQA + DocVQA, 200 rows each; OCRBench has no fresh rows).
R_MAIN025 fresh-row results are taken from the full panel (dispatch P1).

Banks for fresh rows are computed on the fly (RTG re-selects its own
anchors) and cached resumably.  Scoring = same frozen path.

Usage: python rtg_accuracy_fresh.py --mode both
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

import rtg_common as RC
import amp_common as AC
from acu_common import common as C
from acu_common import run_one as acu_run_one, degeneracy
from acu_accuracy import _headline100, _per_q_generic

FRESH = os.path.join(RC.OUT_DIR, "fresh_acc")
os.makedirs(FRESH, exist_ok=True)
ARMS = ["R_GATHER", "F_MAIN025", "S_MAIN025"]
DS_FRESH = ["TextVQA_VAL", "DocVQA_VAL"]   # OCRBench: zero fresh rows


def fresh_rows(ds: str):
    man = json.load(open(os.path.join(RC.OUT_DIR, "..",
                                      "anchor_completion_validation",
                                      "fresh_panel_manifest.json")))
    d = man["datasets"][ds]
    return d["rows"] if d["status"] == "ok" else []


def bank_path(ds: str) -> str:
    return os.path.join(FRESH, f"bank_fresh_rtg_{ds}.json.gz")


def load_or_build_bank(ds: str, eng, scorer: str = "rtg"):
    """Bank for the fresh rows for ONE scorer (per-arm anchors; cached)."""
    import gzip
    path = bank_path(ds).replace("rtg", scorer)
    bank = {"meta": dict(scorer=scorer, split="fresh", ds=ds, K=RC.K,
                         base_commit=RC.git_commit()), "samples": {}}
    if os.path.exists(path):
        with gzip.open(path, "rt") as f:
            bank = json.load(f)
    dataset = AC.common.build_dataset(ds)
    eng.vlm.set_dump_image(dataset.dump_image)
    todo = [i for i in fresh_rows(ds) if str(i) not in bank["samples"]]
    t0 = time.time()
    for n, i in enumerate(todo):
        row = dataset.data.iloc[i]
        msg = AC.common.build_message(eng.vlm, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        if prep["n_vis"] <= RC.K:
            bank["samples"][str(i)] = dict(
                idx=int(i), n_vis=int(prep["n_vis"]),
                keep=list(range(prep["n_vis"])), gid=[], gsize=[])
            continue
        V, DS = eng.encode(prep)
        text_mean, text_seq = eng.instruction_embeds(msg, ds)
        ctx = dict(prep=prep, V=V, DS=DS, K=RC.K, engine=eng,
                   text_mean=text_mean, text_seq=text_seq,
                   ds=ds, qid=int(i))
        keep, _ = RC.select_keep(scorer, ctx, RC.K)
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)
        gid = gid.cpu()
        counts = torch.bincount(gid, minlength=int(keep.numel()))
        bank["samples"][str(i)] = dict(
            idx=int(i), n_vis=int(prep["n_vis"]),
            keep=keep.cpu().tolist(), gid=gid.tolist(),
            gsize=counts.tolist())
        if (n + 1) % 50 == 0 or n + 1 == len(todo):
            el = time.time() - t0
            print(f"[fresh bank rtg {ds}] {n+1}/{len(todo)} "
                  f"({el/(n+1):.2f}s/q)", flush=True)
            tmp = path + ".tmp"
            with gzip.open(tmp, "wt") as f:
                json.dump(bank, f)
            os.replace(tmp, path)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(bank, f)
    os.replace(tmp, path)
    print(f"[fresh bank saved] {path} ({len(bank['samples'])})",
          flush=True)
    return bank, dataset


def shard_path(arm: str, ds: str) -> str:
    return os.path.join(FRESH, arm, f"{ds}.json")


def run_gen():
    model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                 max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    for ds in DS_FRESH:
        eng.vlm.set_dump_image(dataset_dump := None) if False else None
        dataset = AC.common.build_dataset(ds)
        for arm in ARMS:
            scorer = RC.ARMS[arm]["scorer"]
            bank, dataset = load_or_build_bank(ds, eng, scorer=scorer)
            eng.vlm.set_dump_image(dataset.dump_image)
            cfg = RC.arm_cfg(arm)
            path = shard_path(arm, ds)
            shard = RC.load_shard(path)
            if "meta" not in shard:
                shard["meta"] = dict(arm=arm, panel="fresh", ds=ds, K=RC.K,
                                     base_commit=RC.git_commit())
            for n, i in enumerate(fresh_rows(ds)):
                key = str(i)
                if key in shard["records"] or key not in bank["samples"]:
                    continue
                row = dataset.data.iloc[i]
                msg = C.build_message(eng.vlm, dataset, ds, row)
                out = acu_run_one(eng, msg, ds, bank["samples"][key], cfg,
                                  max_new_tokens=2048)
                shard["records"][key] = dict(
                    prediction=out["text"],
                    truncated=len(out["gen_ids"]) >= 2048,
                    n_vis_kept=int(out["meta"]["n_vis_kept"]),
                    layer_calls_ok=bool(out["meta"]["layer_calls_ok"]),
                    degeneracy=degeneracy(out["text"]))
                if len(shard["records"]) % 20 == 0:
                    RC.save_shard(path, shard)
            RC.save_shard(path, shard)
            print(f"[done] fresh/{arm} {ds}: {len(shard['records'])}",
                  flush=True)


def run_score():
    from vlmeval.dataset import build_dataset as vlmeval_build
    ok = True
    for ds in DS_FRESH:
        for arm in ARMS:
            path = shard_path(arm, ds)
            shard = RC.load_shard(path)
            done = sorted(int(k) for k in shard["records"])
            if not done:
                continue
            dataset = vlmeval_build(ds)
            sub = dataset.data.iloc[done].copy()
            if "image" in sub.columns:
                sub = sub.drop(columns=["image"])
            sub["prediction"] = [shard["records"][str(i)]["prediction"]
                                 for i in done]
            sub["truncated"] = [shard["records"][str(i)]["truncated"]
                                for i in done]
            tsv = path.replace(".json", "_pred.tsv")
            sub.to_csv(tsv, sep="\t", index=False)
            res = dataset.evaluate(tsv)
            if hasattr(res, "to_dict"):
                res = res.to_dict()
            per_q = (_per_q_generic(tsv, ds, shard)
                     if ds != "OCRBench" else None)
            h = _headline100(res, ds)
            diff = None
            if per_q and h is not None:
                m = 100.0 * sum(per_q.values()) / len(per_q)
                diff = abs(m - h)
                if diff > 0.05:
                    ok = False
                    print(f"[score FAIL] {arm} {ds}: perq {m:.4f} vs "
                          f"headline {h:.4f}", flush=True)
                    continue
            with open(path.replace(".json", "_score.json"), "w") as f:
                json.dump(dict(arm=arm, panel="fresh", ds=ds, n=len(done),
                               official=res, per_question=per_q,
                               perq_vs_headline_diff=diff,
                               base_commit=RC.git_commit()), f, indent=1)
            print(f"[scored] fresh/{arm} {ds}: n={len(done)} h={h}",
                  flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both",
                    choices=["gen", "score", "both"])
    args = ap.parse_args()
    if args.mode in ("gen", "both"):
        run_gen()
    if args.mode in ("score", "both"):
        run_score()


if __name__ == "__main__":
    main()
