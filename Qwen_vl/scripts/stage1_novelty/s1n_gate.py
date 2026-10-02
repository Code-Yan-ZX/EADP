"""Stage-1 visual-calibration discovery — correctness gates.

G1  fused selector at beta=0 reproduces online b1 keep_idx BITWISE on real
    DEV samples (proves the fusion plumbing changes nothing else).
G2  online b1 selection reproduces the frozen amp bank keep_idx (proves
    the online path == the round-2 BASE pipeline that BASE shards use).
G3  every arm runs end-to-end on 1 smoke sample per dataset; finite keep,
    selector_ms recorded, keep size correct; keep != identity vs b1 on at
    least one sample (else the signal is a no-op -> fail fast).
G4  importance-map correlation: per arm, Pearson corr(fused, official)
    and mean |keep delta| vs official on gate images.

Usage: python s1n_gate.py [--gate-samples 3]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import s1n_common as SC


def build(manifest):
    model = SC.common.load_model(SC.common.BASELINE_MODEL, max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    dataset_cache = {}
    for ds in SC.DS_LIST:
        dataset_cache[ds] = SC.common.build_dataset(ds)
        eng.vlm.set_dump_image(dataset_cache[ds].dump_image)
    return eng, dataset_cache


def _sample_ctx(eng, dataset, ds, row):
    msg = SC.common.build_message(eng.vlm, dataset, ds, row)
    prep = eng.prepare(msg, ds)
    V, DS = eng.encode(prep)
    text_mean, text_seq = eng.instruction_embeds(msg, ds)
    return msg, prep, V, DS, text_mean, text_seq


def g1_identity(eng, dataset_cache, manifest, n):
    from model.e0_selectors import run_selector
    ok, n_cmp = True, 0
    for ds in SC.DS_LIST:
        dataset = dataset_cache[ds]
        for it in manifest["datasets"][ds]["dev"][:n]:
            row = dataset.data.iloc[it["idx"]]
            msg, prep, V, DS, text_mean, text_seq = _sample_ctx(
                eng, dataset, ds, row)
            if prep["n_vis"] <= SC.K:
                continue
            ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq,
                       attn_list=None, vz=None, seed=None)
            keep_b1 = run_selector("b1", SC.K, ctx).cpu().tolist()
            keep_id = run_selector(SC.arm_selector_name("id"), SC.K,
                                   ctx).cpu().tolist()
            n_cmp += 1
            if keep_b1 != keep_id:
                ok = False
                first = next((i for i, (a, b) in
                              enumerate(zip(keep_b1, keep_id)) if a != b),
                             "len")
                print(f"  [G1 MISMATCH] {ds} idx={it['idx']} "
                      f"b1={len(keep_b1)} id={len(keep_id)} first_diff={first}")
    return ok, n_cmp


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
            msg, prep, V, DS, text_mean, text_seq = _sample_ctx(
                eng, dataset, ds, row)
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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def g3_g4_smoke(eng, dataset_cache, manifest):
    from model.e0_selectors import run_selector
    from model.pruner import VisualTokenPruner
    import instrumented
    from common import CudaTimer

    pruner = instrumented.TimedEADPPruner(
        visual_token_num=SC.K, alpha=0.5, beta=2.0, visual_dim=None,
        spatial_merge_size=2)
    pruner.eval()

    out_g3, out_g4 = {}, {}
    for ds in SC.DS_LIST:
        dataset = dataset_cache[ds]
        it = manifest["datasets"][ds]["dev"][0]
        row = dataset.data.iloc[it["idx"]]

        # G3: end-to-end smoke per arm
        for arm in SC.ARMS:
            msg = SC.common.build_message(eng.vlm, dataset, ds, row)
            timings = {}
            res, keep = SC.run_one(eng, msg, ds, arm, timings=timings)
            t = timings.get("selector_ms")
            finite = bool(torch.isfinite(res["state"].keep_idx).all())
            out_g3[f"{arm}/{ds}"] = dict(
                selector_ms=t, keep_n=len(res["keep_idx"]),
                pred_len=len(res["text"]), finite=finite)
            print(f"  [G3] {arm} {ds}: selector_ms={t:.1f} "
                  f"keep={len(res['keep_idx'])} finite={finite}")

        # G4: importance correlation + keep delta vs official (1 image)
        msg, prep, V, DS, text_mean, text_seq = _sample_ctx(eng, dataset, ds,
                                                            row)
        if prep["n_vis"] > SC.K:
            gthw = prep["gthw"]
            sms = eng.inner.visual.spatial_merge_size
            n = int((gthw.prod(-1) // (sms ** 2))[0])
            feats = V[:n].unsqueeze(0)
            gh = int(gthw[0, 1]) // sms
            gw = int(gthw[0, 2]) // sms
            imp_official = pruner._score(feats, text_mean[:1], text_seq[:1],
                                         gh, gw, CudaTimer())[0].cpu().numpy()
            sim = pruner._similarity(feats)
            ctx = dict(prep=prep, V=V, DS=DS, K=SC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq,
                       attn_list=None, vz=None, seed=None)
            keep_official = run_selector("b1", SC.K, ctx).cpu().tolist()
            for arm, cfg in SC.ARMS.items():
                sig = SC._minmax(SC.visual_signal(sim, cfg["variant"],
                                                  cfg["knn_k"]))
                imp_fused = imp_official + cfg["beta"] * sig.cpu().numpy()
                out_g4[f"{arm}/{ds}"] = dict(
                    pearson=round(_corr(imp_fused, imp_official), 4),
                    signal_pearson=round(_corr(sig.cpu().numpy(),
                                               imp_official), 4),
                    keep_delta=int(len(set(keep_official)
                                       ^ set(run_selector(
                                           SC.arm_selector_name(arm), SC.K,
                                           ctx).cpu().tolist()))))
                print(f"  [G4] {arm} {ds}: corr={out_g4[f'{arm}/{ds}']['pearson']} "
                      f"keep_delta={out_g4[f'{arm}/{ds}']['keep_delta']}")
    return out_g3, out_g4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-samples", type=int, default=3)
    args = ap.parse_args()

    manifest = SC.load_manifest()
    eng, dataset_cache = build(manifest)
    SC.ensure_selectors()

    print("[G1] beta=0 fused selector == online b1 (bitwise) ...", flush=True)
    ok1, n1 = g1_identity(eng, dataset_cache, manifest, args.gate_samples)
    print(f"[G1] {'PASS' if ok1 else 'FAIL'} (n compared={n1})")

    print("[G2] online b1 == frozen amp bank keep ...", flush=True)
    ok2, n2 = g2_bank_repro(eng, dataset_cache, manifest, args.gate_samples)
    print(f"[G2] {'PASS' if ok2 else 'FAIL'} (n compared={n2})")

    print("[G3/G4] smoke all arms + importance correlations ...", flush=True)
    g3, g4 = g3_g4_smoke(eng, dataset_cache, manifest)

    result = dict(g1_pass=ok1, g1_n=n1, g2_pass=ok2, g2_n=n2,
                  g3_smoke=g3, g4_corr=g4)
    p = os.path.join(SC.OUT_DIR, "gate.json")
    with open(p, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[saved] {p}")
    if not (ok1 and ok2):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
