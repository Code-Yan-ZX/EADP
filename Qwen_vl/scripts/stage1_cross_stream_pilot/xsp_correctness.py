"""Stage-1 Cross-Stream Pilot — correctness gates G1-G7 (protocol §5).

No formal accuracy run is allowed until every gate passes.  Requires the four
scorer banks to exist (run xsp_bank.py first).  Results ->
outputs/stage1_cross_stream_pilot/correctness.json

Usage: /home/dell/miniconda3/envs/qwen3vl_clean/bin/python xsp_correctness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.nn.functional as F

import xsp_common as XC
import amp_common as AC


N_SAMPLES_PER_DS = 3
DECODE_TOKENS = 2048        # full greedy output for the E-reproduction gate


def feat_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


# ------------------------------------------------------------------ G1 ----
def reference_stage1(scorer: str, V: torch.Tensor, DS_list, sim: torch.Tensor,
                     m: int = XC.M_NB, eps: float = XC.EPS,
                     floor: float = XC.FLOOR) -> torch.Tensor:
    """Explicit per-point Python reference (slow, unambiguous)."""
    N = V.shape[0]
    m = min(m, N - 1)
    if scorer == "uniform":
        return torch.ones(N, dtype=torch.float64)
    w = torch.full((N,), floor, dtype=torch.float64)
    streams = [V] if scorer == "main_residual" else DS_list
    for st in streams:
        H = (st.double() / st.double().norm(dim=-1, keepdim=True)
             .clamp_min(eps))
        d = torch.zeros(N, dtype=torch.float64)
        for i in range(N):
            # neighbours: top-m of sim row excluding self; ties -> smaller idx
            cand = sorted(((float(sim[i, j].double()), -j)
                           for j in range(N) if j != i), reverse=True)
            nb = [-c[1] for c in cand[:m]]
            mu = H[nb].mean(dim=0)
            d[i] = ((H[i] - mu) ** 2).sum()
        w = w + d / (d.mean() + eps)
    return w.float()


def gate_g1(res):
    torch.manual_seed(7)
    N, D = 48, 16
    V = torch.randn(N, D)
    V[5] = V[4]          # duplicate rows (repeated-token behaviour)
    V[6] = V[4]
    V[10] = 0.0          # zero main row
    DS_list = [torch.randn(N, D) for _ in range(3)]
    DS_list[2] = torch.zeros(N, D)                  # all-zero stream
    Vn = F.normalize(V, dim=1)
    sim = Vn @ Vn.t()
    ok = True
    for scorer in ("cross_stream", "main_residual", "uniform"):
        w_vec, diag = XC.stage1_importance(scorer, V, DS_list, sim)
        w_ref = reference_stage1(scorer, V, DS_list, sim)
        rel = float((w_vec.double() - w_ref.double()).abs().max()
                    / w_ref.double().abs().max().clamp_min(1e-6))
        finite = bool(torch.isfinite(w_vec).all()) and bool((w_vec >= 0).all())
        w2, _ = XC.stage1_importance(scorer, V, DS_list, sim)
        det = bool(torch.equal(w_vec, w2))
        ok_s = rel < 1e-5 and finite and det
        if scorer == "cross_stream":
            # the all-zero third stream must be counted and contribute zero:
            # w with 3 streams (one zero) == reference over the 2 real streams
            ref2 = reference_stage1("cross_stream", V, DS_list[:2], sim)
            ok_s = ok_s and diag["zero_norm"] == [0, 0, N] \
                and float((w_vec.double() - ref2.double()).abs().max()) < 1e-5
        res["gates"]["G1"][scorer] = dict(rel=rel, finite=finite,
                                          deterministic=det,
                                          zero_norm=diag["zero_norm"],
                                          ok=bool(ok_s))
        ok = ok and ok_s
        print(f"[G1] {scorer} rel={rel:.2e} finite={finite} det={det} "
              f"zero_norm={diag['zero_norm']} ok={ok_s}", flush=True)

    # neighbour properties: self excluded, ties -> smaller index,
    # caller's matrix untouched
    sim_e = torch.zeros(4, 4)
    sim_e[0, 1] = 0.8
    sim_e[0, 2] = 0.8            # exact tie: col 1 must precede col 2
    sim_e[0, 3] = 0.5
    nb_e = XC.neighbor_index(sim_e, m=3)
    tie_order_ok = int(nb_e[0, 0]) == 1 and int(nb_e[0, 1]) == 2
    sim_orig = torch.rand(16, 16)
    sim_orig.fill_diagonal_(1.0)
    sim_copy = sim_orig.clone()
    nbr_full = XC.neighbor_index(sim_orig, m=15)
    untouched = bool(torch.equal(sim_orig, sim_copy))
    no_self = all(i not in nbr_full[i].tolist() for i in range(16))
    diag_ok = not bool((nbr_full == torch.arange(16).unsqueeze(1)).any())
    res["gates"]["G1"]["neighbors"] = dict(tie_order_ok=bool(tie_order_ok),
                                           matrix_untouched=untouched,
                                           no_self=bool(no_self),
                                           diag_never_selected=bool(diag_ok))
    ok = ok and tie_order_ok and untouched and no_self and diag_ok
    print(f"[G1] neighbors tie_order={tie_order_ok} untouched={untouched} "
          f"no_self={no_self} diag_sel={not diag_ok}", flush=True)
    res["gates"]["G1"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G2 ----
class _FakeVis:
    spatial_merge_size = 2


class _FakeEng:
    inner = type("I", (), {"visual": _FakeVis()})()


def gate_g2(res):
    """Multi-image: no cross-image neighbourhood / grouping."""
    torch.manual_seed(11)
    N, D = 64, 8
    V1, V2 = torch.randn(N, D), torch.randn(N, D)
    DS1 = [torch.randn(N, D) for _ in range(3)]
    DS2 = [torch.randn(N, D) for _ in range(3)]
    V_cat = torch.cat([V1, V2])
    DS_cat = [torch.cat([a, b]) for a, b in zip(DS1, DS2)]
    ctx_cat = dict(prep=dict(gthw=torch.tensor([[1, 16, 16], [1, 16, 16]])),
                   V=V_cat, DS=DS_cat, engine=_FakeEng())
    ok = True
    for scorer in ("cross_stream", "main_residual", "uniform"):
        keep_cat, _ = XC.select_keep(scorer, ctx_cat, XC.K)
        k1, _ = XC.select_keep(
            scorer, dict(prep=dict(gthw=torch.tensor([[1, 16, 16]])),
                         V=V1, DS=DS1, engine=_FakeEng()), XC.K)
        k2, _ = XC.select_keep(
            scorer, dict(prep=dict(gthw=torch.tensor([[1, 16, 16]])),
                         V=V2, DS=DS2, engine=_FakeEng()), XC.K)
        eq = bool(torch.equal(keep_cat, torch.cat([k1, k2 + N])))
        res["gates"]["G2"][scorer] = dict(concat_eq_independent=eq, ok=eq)
        ok = ok and eq
        print(f"[G2] {scorer} concat_eq_independent={eq}", flush=True)
    res["gates"]["G2"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G3 ----
def gate_g3(res):
    """Degenerate cases: N<=K keep-all; zero stream; determinism."""
    ok = True
    torch.manual_seed(3)
    N, D = 32, 8
    V = torch.randn(N, D)
    DS_list = [torch.randn(N, D) for _ in range(3)]
    sim = F.normalize(V, dim=1) @ F.normalize(V, dim=1).t()
    # zero third stream contributes exactly zero: w(3 streams, one zero) ==
    # reference over the two real streams; the stream is still counted
    DS_zero = [DS_list[0], DS_list[1], torch.zeros(N, D)]
    wz, dz = XC.stage1_importance("cross_stream", V, DS_zero, sim)
    ref2 = reference_stage1("cross_stream", V, DS_list[:2], sim)
    zero_contrib = (dz["d_mean"][2] == 0.0
                    and float((wz.double() - ref2.double()).abs().max())
                    < 1e-5)
    # N <= K -> keep-all, no scoring
    keep, _ = XC.select_keep("cross_stream",
                             dict(prep=dict(gthw=torch.tensor([[1, 8, 16]])),
                                  V=V, DS=DS_list, engine=_FakeEng()), N + 16)
    keepall = bool(torch.equal(keep, torch.arange(N)))
    # duplicate-token determinism (two identical rows in every stream)
    V2 = V.clone()
    V2[7] = V2[3]
    DS2 = [d.clone() for d in DS_list]
    for d in DS2:
        d[7] = d[3]
    sim2 = F.normalize(V2, dim=1) @ F.normalize(V2, dim=1).t()
    w_a, _ = XC.stage1_importance("cross_stream", V2, DS2, sim2)
    w_b, _ = XC.stage1_importance("cross_stream", V2, DS2, sim2)
    dup_det = bool(torch.equal(w_a, w_b))
    res["gates"]["G3"] = dict(zero_stream_contributes_zero=zero_contrib,
                              n_le_k_keep_all=keepall,
                              duplicate_deterministic=dup_det,
                              ok=bool(zero_contrib and keepall and dup_det))
    ok = zero_contrib and keepall and dup_det
    print(f"[G3] zero_stream={zero_contrib} keep_all={keepall} "
          f"dup_det={dup_det}", flush=True)
    res["gates"]["G3"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G4 ----
def gate_g4(res, manifest, model, eng):
    """E arms reproduce the round-1/2 BASE / MAIN025 DEV predictions
    bitwise (anchors + greedy output; the old shards carry no logits —
    recorded as such in the report)."""
    ok = True
    for ds in XC.DS_LIST:
        old_base = json.load(open(os.path.join(
            AC.OUT_DIR, "acc", "dev", "BASE", "K256", f"{ds}.json")))
        old_main = json.load(open(os.path.join(
            AC.OUT_DIR, "acc", "dev", "MAIN025", "K256", f"{ds}.json")))
        legacy = XC.load_legacy_bank(ds)          # flat {idx: rec}
        xbank = XC.load_bank("eadp", ds)
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in manifest["datasets"][ds]["dev"][:args.samples_per_ds]:
            key = str(it["idx"])
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            keep_new = xbank["samples"][key]["keep"]
            keep_old = legacy[key]["keep"]
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=XC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            keep_live = AC.official_facility_keep(ctx, XC.K).cpu().tolist()
            anchors_ok = keep_new == keep_old == keep_live
            out_g = AC.run_one(eng, msg, ds, {"keep": keep_new},
                               XC.arm_run_cfg("E_GATHER"),
                               max_new_tokens=DECODE_TOKENS)
            out_m = AC.run_one(eng, msg, ds, {"keep": keep_new},
                               XC.arm_run_cfg("E_MAIN025"),
                               max_new_tokens=DECODE_TOKENS)
            pred_ok = (out_g["text"] == old_base["records"][key]["prediction"]
                       and out_m["text"]
                       == old_main["records"][key]["prediction"])
            sample_ok = anchors_ok and pred_ok
            ok = ok and sample_ok
            print(f"[G4] {ds}_{key} anchors={anchors_ok} pred={pred_ok}",
                  flush=True)
    res["gates"]["G4"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G5 ----
def gate_g5(res, manifest, model, eng):
    """lam=0 path == plain gather (features + prefill logits bitwise) for
    the eadp and cross_stream scorers' own anchors."""
    ok = True
    for ds in XC.DS_LIST:
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        it = manifest["datasets"][ds]["dev"][0]
        key = str(it["idx"])
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        V, DS = eng.encode(prep)
        for scorer in ("eadp", "cross_stream"):
            keep = torch.as_tensor(
                XC.load_bank(scorer, ds)["samples"][key]["keep"],
                dtype=torch.long, device=V.device)
            dropped_idx, gid, _ = AC.compute_assignment(V, keep)
            y0 = AC.merge_stream(V, keep, dropped_idx, gid, "lam0", 0.0)
            ys0 = [AC.merge_stream(d, keep, dropped_idx, gid, "lam0", 0.0)
                   for d in DS]
            feat_same = (torch.equal(y0, V[keep])
                         and all(torch.equal(y0_, d[keep])
                                 for y0_, d in zip(ys0, DS)))
            st_g = eng.prefill(prep, V, DS, keep)
            st_l = eng.prefill(prep, V, DS, keep, V_sel=y0, DS_sel=ys0)
            d_logit = float((st_g.logits[0] - st_l.logits[0]).abs().max()
                            .item())
            ok_s = feat_same and d_logit == 0.0
            ok = ok and ok_s
            print(f"[G5] {ds}_{key} {scorer} feat={feat_same} "
                  f"d_logit={d_logit:.2e}", flush=True)
    res["gates"]["G5"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G6 ----
def gate_g6(res, manifest, model, eng):
    """Frozen banks == live recomputation (keep + gid) for all 4 scorers."""
    ok = True
    for ds in XC.DS_LIST:
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in manifest["datasets"][ds]["dev"][:args.samples_per_ds]:
            key = str(it["idx"])
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=XC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            for scorer in XC.SCORERS:
                rec = XC.load_bank(scorer, ds)["samples"][key]
                keep_live, _ = XC.select_keep(scorer, ctx, XC.K)
                keep_ok = keep_live.cpu().tolist() == rec["keep"]
                dropped_idx, gid, _ = AC.compute_assignment(V, keep_live)
                gid_ok = gid.cpu().tolist() == rec["gid"]
                ok = ok and keep_ok and gid_ok
                if not (keep_ok and gid_ok):
                    print(f"[G6] MISMATCH {ds}_{key} {scorer} "
                          f"keep={keep_ok} gid={gid_ok}", flush=True)
            print(f"[G6] {ds}_{key} bank==live (all scorers)", flush=True)
    res["gates"]["G6"]["ok"] = bool(ok)


# ------------------------------------------------------------------ G7 ----
def gate_g7(res, manifest, model, eng):
    """Invariants for all six arms + DS-integrity (DS_sel == DS[keep])."""
    ok = True
    for ds in XC.DS_LIST:
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        it = manifest["datasets"][ds]["dev"][0]
        key = str(it["idx"])
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        V, DS = eng.encode(prep)
        for arm in XC.ARMS:
            rec = XC.load_bank(XC.ARMS[arm]["scorer"], ds)["samples"][key]
            out = AC.run_one(eng, msg, ds, {"keep": rec["keep"]},
                             XC.arm_run_cfg(arm), max_new_tokens=16)
            meta = out["meta"]
            inv_ok = (meta["n_vis_kept"] == XC.K and meta["no_dup"]
                      and meta["in_range"] and meta["ascending"]
                      and meta.get("ds_lengths") == [XC.K] * len(DS)
                      and meta["layer_calls_ok"] and meta["cache_ok"]
                      and len(out["keep_idx"]) == XC.K)
            ok = ok and inv_ok
            if not inv_ok:
                print(f"[G7] INVARIANT FAIL {arm} {ds}_{key}: {meta}",
                      flush=True)
        # DS integrity: explicit DS[keep] vs the engine's internal gather
        rec = XC.load_bank("cross_stream", ds)["samples"][key]
        keep = torch.as_tensor(rec["keep"], dtype=torch.long, device=V.device)
        DS_k = [d[keep] for d in DS]
        st_a = eng.prefill(prep, V, DS, keep, DS_sel=None)
        st_b = eng.prefill(prep, V, DS, keep, DS_sel=DS_k)
        ds_ok = float((st_a.logits[0] - st_b.logits[0]).abs().max()
                      .item()) == 0.0
        ok = ok and ds_ok
        print(f"[G7] {ds} six-arm invariants + DS integrity={ds_ok}",
              flush=True)
    res["gates"]["G7"]["ok"] = bool(ok)


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-ds", type=int, default=N_SAMPLES_PER_DS)
    args = ap.parse_args()

    out_path = os.path.join(XC.OUT_DIR, "correctness.json")
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)

    res: dict = dict(base_commit=XC.git_commit(), gates={})
    for fn in (gate_g1, gate_g2, gate_g3, gate_g4, gate_g5, gate_g6, gate_g7):
        res["gates"][fn.__name__.replace("gate_", "").upper()] = {"ok": False}
    gate_g1(res)
    gate_g2(res)
    gate_g3(res)
    gate_g4(res, manifest, model, eng)
    gate_g5(res, manifest, model, eng)
    gate_g6(res, manifest, model, eng)
    gate_g7(res, manifest, model, eng)

    res["all_passed"] = all(g["ok"] for g in res["gates"].values())
    res["env"] = XC.env_meta()
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"\n[{'PASS' if res['all_passed'] else 'FAIL'}] "
          f"{ {k: v['ok'] for k, v in res['gates'].items()} } -> {out_path}",
          flush=True)
    raise SystemExit(0 if res["all_passed"] else 1)


if __name__ == "__main__":
    main()
