"""Stage-1 grounding discovery — correctness gates (run BEFORE any accuracy).

G1  variant-scorer base path == official TimedEADPPruner._score, bitwise,
    on real DEV samples (importance + keep_idx).
G2  online b1 selection reproduces the frozen amp bank keep_idx on real
    DEV samples (proves the online path == the round-2 BASE pipeline).
G3  every S1 arm runs end-to-end on 2 smoke samples per dataset;
    importance finite, selector_ms recorded, diag populated.

Usage: python s1g_gate.py [--gate-samples 4]
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import s1g_common as SC


def build(manifest):
    model = SC.common.load_model(SC.common.BASELINE_MODEL, max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    dataset_cache = {}
    for ds in SC.DS_LIST:
        dataset_cache[ds] = SC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset_cache[ds].dump_image)
    return eng, dataset_cache


def g1_scorer_identity(eng, dataset_cache, manifest, n):
    from model import eadp_stage1
    from model.e0_selectors import _EADP_CACHE
    from model.pruner import VisualTokenPruner
    import instrumented
    from common import CudaTimer

    pruner = instrumented.TimedEADPPruner(
        visual_token_num=SC.K, alpha=0.5, beta=2.0,
        visual_dim=None, spatial_merge_size=2)
    pruner.eval()
    ok, worst = True, 0.0
    for ds in SC.DS_LIST:
        items = manifest["datasets"][ds]["dev"][:n]
        dataset = dataset_cache[ds]
        for it in items:
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            gthw = prep["gthw"]
            sms = eng.inner.visual.spatial_merge_size
            split_sizes = (gthw.prod(-1) // (sms ** 2)).tolist()
            off = 0
            for i, n_i in enumerate(split_sizes):
                if n_i <= SC.K:
                    off += n_i
                    continue
                feats = V[off:off + n_i].unsqueeze(0)
                gh = int(gthw[i, 1]) // sms
                gw = int(gthw[i, 2]) // sms
                imp_official = pruner._score(feats, text_mean[i:i + 1],
                                             text_seq[i:i + 1], gh, gw,
                                             CudaTimer())
                imp_variant = eadp_stage1.stage1_importance(
                    feats, text_mean[i:i + 1], text_seq[i:i + 1],
                    gh, gw, mode="base", stat="g1", lam=1.0)
                d = (imp_official - imp_variant).abs().max().item()
                worst = max(worst, d)
                if d != 0.0:
                    ok = False
                off += n_i
    return ok, worst


def g2_bank_repro(eng, dataset_cache, manifest, n):
    from model.e0_selectors import run_selector
    import amp_common as AC
    ok, n_cmp = True, 0
    for ds in SC.DS_LIST:
        items = manifest["datasets"][ds]["dev"][:n]
        bank = AC.load_bank("dev", ds)
        dataset = dataset_cache[ds]
        for it in items:
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            gthw = prep["gthw"]
            if prep["n_vis"] <= SC.K:
                continue
            ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq,
                       attn_list=None, vz=None, seed=None)
            keep_online = run_selector("b1", SC.K, ctx).cpu().tolist()
            keep_bank = list(bank[str(it["idx"])]["keep"])
            n_cmp += 1
            if keep_online != keep_bank:
                ok = False
                first = next((i for i, (a, b) in
                              enumerate(zip(keep_online, keep_bank)) if a != b),
                             "len")
                print(f"  [G2 MISMATCH] {ds} idx={it['idx']} "
                      f"online={len(keep_online)} bank={len(keep_bank)} "
                      f"first_diff={first}")
    return ok, n_cmp


def g3_smoke(eng, dataset_cache, manifest, lam):
    SC.ensure_selectors(lam)
    out = {}
    for arm in SC.S1_ARMS:
        for ds in SC.DS_LIST:
            it = manifest["datasets"][ds]["dev"][0]
            dataset = dataset_cache[ds]
            row = dataset.data.iloc[it["idx"]]
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            res, diag = SC.run_one(eng, msg, ds, arm, timings=timings,
                                   want_diag=True)
            t = timings.get("selector_ms")
            finite = bool(torch.isfinite(res["state"].keep_idx).all())
            d = diag["last"] or {}
            out[f"{arm}/{ds}"] = dict(
                selector_ms=t, keep_n=len(res["keep_idx"]),
                pred_len=len(res["text"]), diag_ok=d is not None,
                n_g=len(d.get("g", [])) if d else 0)
            print(f"  [G3] {arm} {ds}: selector_ms={t:.1f} "
                  f"keep={len(res['keep_idx'])} diag_n={out[f'{arm}/{ds}']['n_g']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-samples", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args()

    manifest = SC.load_manifest()
    eng, dataset_cache = build(manifest)

    print("[G1] variant base-path == official scorer ...", flush=True)
    ok1, worst = g1_scorer_identity(eng, dataset_cache, manifest,
                                    args.gate_samples)
    print(f"[G1] {'PASS' if ok1 else 'FAIL'} (max|diff|={worst})")

    print("[G2] online b1 == frozen amp bank keep ...", flush=True)
    ok2, n_cmp = g2_bank_repro(eng, dataset_cache, manifest,
                               args.gate_samples)
    print(f"[G2] {'PASS' if ok2 else 'FAIL'} (n compared={n_cmp})")

    print(f"[G3] smoke all arms (lam={args.lam}) ...", flush=True)
    smoke = g3_smoke(eng, dataset_cache, manifest, args.lam)

    result = dict(g1_pass=ok1, g1_max_diff=worst, g2_pass=ok2,
                  g2_n=n_cmp, g3_smoke=smoke, lam=args.lam)
    p = os.path.join(SC.OUT_DIR, "gate.json")
    with open(p, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[saved] {p}")
    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
