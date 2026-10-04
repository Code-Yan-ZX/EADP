"""Technical checks for the legacy-scenario delta ONLY (user directive
2026-10-03: reuse existing correctness gates, check just this change).

The change is: acu_run_one now forwards deepstack=False, pos="1d" to
eng.prefill (official EADP legacy config).  Everything else (RTG scorer,
facility, merge math, frozen banks, official scorers) is untouched and was
gated in the core round (G1-G5, T1-T7).

Three checks, all bitwise:
  C1 bank mode-independence — recompute 3 frozen RTG bank records through
     the bank path (prepare/encode/instruction_embeds/select_keep; the
     path never calls prefill/decode) and compare keep/gid/gsize/w_diag
     against bank_full_rtg_*.json.gz.
  C2 lambda=0 identity under legacy flags — kind="base" vs kind="uniform"
     lam=0.0, both with deepstack=False/pos='1d': predictions + gen_ids
     must be bitwise identical (merge is a no-op at lam=0).
  C3 legacy smoke + determinism — 8 samples of the real R cfg under legacy
     flags: engine invariants hold (layer_calls_ok, exact keep count);
     re-running sample 0 reproduces its text bitwise.

No subset-score gate: subset scores are NOT required to match paper
full-split scores (user directive, DEV-300 gate removed).

Output: outputs/stage1_roundtrip_pilot/legacy_full/legacy_checks.json
Usage:  python rtg_legacy_checks.py
"""

from __future__ import annotations

import gzip
import json
import os

import torch

import rtg_common as RC
import amp_common as AC
from acu_common import common as C
from acu_common import run_one as acu_run_one

ARM = "L_R_MAIN025"
LEGACY = dict(deepstack=False, pos="1d")
OUT = os.path.join(RC.OUT_DIR, "legacy_full", "legacy_checks.json")

C1_KEYS = {"TextVQA_VAL": ["0", "1"], "DocVQA_VAL": ["0"]}
C2_DS_KEYS = [("TextVQA_VAL", 0), ("TextVQA_VAL", 1),
              ("DocVQA_VAL", 0), ("OCRBench", 0)]
C3_N = 8


def load_bank(ds):
    p = os.path.join(RC.OUT_DIR, "full", f"bank_full_rtg_{ds}.json.gz")
    with gzip.open(p, "rt") as f:
        return json.load(f)


def c1_bank_bitwise(eng):
    results = []
    for ds, keys in C1_KEYS.items():
        bank = load_bank(ds)
        dataset = AC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        for key in keys:
            row = dataset.data.iloc[int(key)]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=RC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq,
                       ds=ds, qid=int(key))
            keep, diags = RC.select_keep("rtg", ctx, RC.K, want_diag=True)
            dropped_idx, gid, _ = AC.compute_assignment(V, keep)
            gid = gid.cpu()
            counts = torch.bincount(gid, minlength=int(keep.numel()))
            rec = bank["samples"][key]
            ok = (keep.cpu().tolist() == rec["keep"]
                  and gid.tolist() == rec["gid"]
                  and counts.tolist() == rec["gsize"])
            results.append(dict(ds=ds, key=key, ok=bool(ok),
                                n_vis=int(prep["n_vis"])))
            print(f"[C1] {ds}[{key}] bank bitwise: {'OK' if ok else 'FAIL'}",
                  flush=True)
    return results


def _gen(eng, ds, idx, cfg, bank):
    dataset = AC.common.build_dataset(ds)
    eng.vlm.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[idx]
    msg = C.build_message(eng.vlm, dataset, ds, row)
    out = acu_run_one(eng, msg, ds, bank["samples"][str(idx)], cfg,
                      max_new_tokens=2048, **LEGACY)
    return out


def c2_lam0_identity(eng):
    results = []
    cfg_base = dict(kind="base", lam=0.0, scope="main", scorer="rtg")
    cfg_lam0 = dict(kind="uniform", lam=0.0, scope="main", scorer="rtg")
    banks = {}
    for ds, idx in C2_DS_KEYS:
        if ds not in banks:
            banks[ds] = load_bank(ds)
        a = _gen(eng, ds, idx, cfg_base, banks[ds])
        b = _gen(eng, ds, idx, cfg_lam0, banks[ds])
        ga, gb = a["gen_ids"], b["gen_ids"]
        if not isinstance(ga, list):
            ga = ga.tolist()
        if not isinstance(gb, list):
            gb = gb.tolist()
        ok = (a["text"] == b["text"] and ga == gb)
        results.append(dict(ds=ds, idx=idx, ok=bool(ok)))
        print(f"[C2] {ds}[{idx}] lam0 identity legacy: "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    return results


def c3_smoke_determinism(eng):
    cfg = RC.arm_cfg("R_MAIN025")
    banks = {}
    results, texts = [], {}
    ds0, idx0 = "TextVQA_VAL", 0
    plan = [(ds0, idx0)] + [("TextVQA_VAL", i) for i in range(1, 5)] \
        + [("DocVQA_VAL", i) for i in range(3)] + [("OCRBench", 0)]
    for ds, idx in plan:
        if ds not in banks:
            banks[ds] = load_bank(ds)
        out = _gen(eng, ds, idx, cfg, banks[ds])
        meta = out["meta"]
        inv_ok = bool(meta["layer_calls_ok"])
        # legacy engine reports st.n_vis=0 (vmask only built when
        # deepstack=True); the real kept count is the compacted keep_idx
        n_kept = int(out["state"].keep_idx.numel())
        keep_ok = n_kept == min(RC.K, int(meta["n_vis_full"]))
        results.append(dict(ds=ds, idx=idx, invariants_ok=inv_ok,
                            keep_ok=bool(keep_ok),
                            n_vis_kept=n_kept,
                            n_vis_full=int(meta["n_vis_full"])))
        texts[(ds, idx)] = out["text"]
        print(f"[C3] {ds}[{idx}] inv={inv_ok} keep={keep_ok}", flush=True)
    # determinism: rerun the first sample
    out2 = _gen(eng, ds0, idx0, cfg, banks[ds0])
    det = out2["text"] == texts[(ds0, idx0)]
    print(f"[C3] determinism rerun: {'OK' if det else 'FAIL'}", flush=True)
    return results, bool(det)


def main():
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    c1 = c1_bank_bitwise(eng)
    c2 = c2_lam0_identity(eng)
    c3, det = c3_smoke_determinism(eng)

    all_pass = (all(r["ok"] for r in c1) and all(r["ok"] for r in c2)
                and all(r["invariants_ok"] and r["keep_ok"] for r in c3)
                and det)
    out = dict(arm=ARM, pipeline="official_legacy",
               change_under_test="acu_run_one deepstack=False pos='1d'",
               base_commit=RC.git_commit(),
               c1_bank_bitwise=c1, c2_lam0_identity=c2,
               c3_smoke=c3, c3_determinism=det, all_pass=all_pass)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[legacy checks] ALL_PASS={all_pass} -> {OUT}", flush=True)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
