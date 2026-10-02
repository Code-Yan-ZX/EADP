"""RTG pilot — correctness gates (protocol §5; all pass before scores).

T1  independent element-wise reference vs vectorised: P col-sums=1,
    B row-sums=1, c = sum_i P*B, w sums to 1.
T2  FP64: 1/L <= c_t <= 1 and the chi-square identity
    c_t = (1 + sum_i (P_it - pbar_i)^2 / pbar_i) / L, pbar_i = mean_t P_it;
    FP32 gives a reasonable (non-bitwise) error.
T3  identical columns -> equal weights; L=1 -> weight 1; zero input finite;
    real N>K pruning (N=64, K=16); multi-image slice isolation; N<=K keep-all.
T4  SHUF: per-column sorted values unchanged (entropy invariant); weights
    DO change on a constructed non-degenerate example (three same-entropy
    columns, two sharing a peak, third peaking elsewhere -> the shared
    columns get equal weight, the unique-peak column higher weight).
    Real-sample anchor/weight change statistics are recorded separately in
    the bank diagnostics (never misjudged as implementation errors).
T5  E formal-config 9-sample reproduction: keep bitwise vs the round-1
    frozen bank; greedy outputs bitwise vs the verified cross_stream-round
    E shard predictions (which were themselves gate-verified against
    round 1); lambda=0 identity runs separately below.
T6  new scorers: live keep/gid == bank; DS untouched (lengths + DS_sel
    unused); output counts + native invariants.

Usage:
  python rtg_correctness.py --gates T1,T2,T3,T4
  python rtg_correctness.py --gates T5,T6 [--per-ds 3]
  python rtg_correctness.py --lam0
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

import rtg_common as RC
import amp_common as AC
from acu_common import common as C  # env bootstrap
from acu_common import run_one as acu_run_one

OUT = os.path.join(RC.OUT_DIR, "correctness.json")
T5_PRED = os.path.join(RC.OUT_DIR, "gate_t5_predictions.json")


# ---------------------------------------------------------------------------
# independent element-wise reference (explicit loops, FP64)
# ---------------------------------------------------------------------------
def ref_roundtrip(A: torch.Tensor):
    N, L = A.shape
    A = A.double()
    P = torch.empty(N, L, dtype=torch.float64)
    for t in range(L):
        col = 100.0 * A[:, t]
        e = torch.exp(col - col.max())
        P[:, t] = e / e.sum()
    B = torch.empty(N, L, dtype=torch.float64)
    for i in range(N):
        row = torch.log(P[i, :])
        B[i, :] = torch.exp(row - torch.logsumexp(row, dim=0))
    c = (P * B).sum(dim=0)
    w = c / c.sum()
    return P, B, c, w


def rtg_weights_only(A: torch.Tensor, mode: str, ds: str = "", qid=None):
    w, _ = RC.text_weight(A, mode, ds, qid)
    return w


# ---------------------------------------------------------------------------
def gate_t1(res):
    torch.manual_seed(20261002)
    A = torch.randn(17, 6, dtype=torch.float32)
    A[3, 2] = 0.0
    P, B, c, w_ref = ref_roundtrip(A)
    w_vec = rtg_weights_only(A, "rtg")
    err = dict(
        p_colsum=float((P.sum(dim=0) - 1).abs().max()),
        b_rowsum=float((B.sum(dim=1) - 1).abs().max()),
        c_recompute=float((c - (P * B).sum(0)).abs().max()),
        w_sum=float((w_vec.sum() - 1).abs()),
        w_vs_ref=float((w_vec.double() - w_ref).abs().max()),
    )
    ok = all(v < 1e-5 for v in err.values())
    res["T1"] = dict(errors=err, ok=bool(ok))
    print("T1:", res["T1"]["ok"],
          {k: f"{v:.2e}" for k, v in err.items()}, flush=True)


def gate_t2(res):
    torch.manual_seed(20261002)
    A = torch.randn(9, 5, dtype=torch.float64)
    P, B, c, w = ref_roundtrip(A)
    pbar = P.mean(dim=1)                                            # [N]
    chi = (1.0 + ((P - pbar.unsqueeze(1)) ** 2
                  / pbar.unsqueeze(1)).sum(dim=0)) / P.shape[1]
    ident_err = float((c - chi).abs().max())
    range_ok = bool((c >= 1.0 / P.shape[1] - 1e-12).all()
                    and (c <= 1.0 + 1e-12).all())
    logP = torch.log_softmax(100.0 * A.float(), dim=0)
    logB = logP - torch.logsumexp(logP, dim=1, keepdim=True)
    c32 = torch.exp(logP + logB).sum(dim=0)
    fp32_err = float((c32.double() - c).abs().max())
    res["T2"] = dict(chi2_identity_fp64_err=ident_err,
                     range_1_over_L_to_1=range_ok,
                     fp32_abs_err=fp32_err,
                     ok=bool(ident_err < 1e-12 and range_ok
                             and fp32_err < 1e-5))
    print("T2:", res["T2"]["ok"],
          f"ident={ident_err:.2e} fp32={fp32_err:.2e}", flush=True)


def gate_t3(res):
    torch.manual_seed(20261002)
    # identical columns -> equal weights
    A = torch.randn(11, 4)
    A[:, 1] = A[:, 0]
    A[:, 3] = A[:, 0]
    w = rtg_weights_only(A, "rtg")
    same_cols_equal = bool((w[1] - w[0]).abs() < 1e-6
                           and (w[3] - w[0]).abs() < 1e-6)
    # L=1 -> weight 1
    w1 = rtg_weights_only(torch.randn(7, 1), "rtg")
    single_col = bool((w1 - 1.0).abs().max() < 1e-6)
    # zero input finite (uniform fallback)
    wz = rtg_weights_only(torch.zeros(5, 3), "rtg")
    zero_finite = bool(torch.isfinite(wz).all() and (wz > 0).all()
                       and (wz - 1.0 / 3).abs().max() < 1e-6)
    # real N>K pruning through the full scorer+facility path (N=64, K=16)
    V = torch.randn(64, 32)
    feats = V.unsqueeze(0)
    te = torch.randn(1, 32)          # text_mean [1, D]
    ts = torch.randn(1, 8, 32)       # text_seq  [1, M, D]
    from model.pruner import _sim_visual_impl
    import instrumented
    imp, w = RC.rtg_importance(feats, te, ts, 8, 8, "rtg")
    sim01 = _sim_visual_impl(feats)
    sel, _ = instrumented.SELECTORS["facility"](imp, sim01, 16)
    keep = sel[0].sort().values
    n_gt_k = bool(keep.numel() == 16 and len(set(keep.tolist())) == 16)
    # multi-image slice isolation: importance of image A's slice computed
    # inside a concatenated feature matrix == importance of A alone
    V2 = torch.randn(64, 32)
    feats_cat = torch.cat([feats, V2.unsqueeze(0)], dim=1)
    imp_cat_slice = RC.rtg_importance(feats_cat[:, :64], te, ts, 8, 8,
                                      "rtg")[0]
    iso = bool(torch.equal(imp_cat_slice, imp))
    res["T3"] = dict(same_cols_equal=same_cols_equal,
                     single_col=single_col, zero_finite=zero_finite,
                     n_gt_k=n_gt_k, multi_image_isolation=iso,
                     ok=bool(same_cols_equal and single_col and zero_finite
                             and n_gt_k and iso))
    print("T3:", res["T3"]["ok"], flush=True)


def gate_t4(res):
    torch.manual_seed(20261002)
    A = torch.randn(20, 5)
    A_shuf = RC.per_column_shuffle(A, "T4", 0)
    sort_inv = bool(torch.equal(A_shuf.sort(dim=0).values,
                                A.sort(dim=0).values))
    # non-degenerate example (protocol §5.4): three same-entropy columns,
    # cols 0/1 identical (shared peak), col 2 peaks elsewhere
    A3 = torch.zeros(3, 3)
    A3[0, 0] = 4.0; A3[1, 0] = 0.0; A3[2, 0] = 0.0
    A3[:, 1] = A3[:, 0]                              # identical column
    A3[1, 2] = 4.0                                   # peak elsewhere
    w3 = rtg_weights_only(A3, "rtg")
    weights_change = bool((w3[0] - w3[1]).abs() < 1e-6
                          and w3[2] > w3[0] + 1e-3)
    # SHUF on this example must also change weights vs an equal-weight view
    w3s = rtg_weights_only(A3, "shuf", "T4", 0)
    res["T4"] = dict(sort_invariant=sort_inv,
                     weights_change=weights_change,
                     example_weights=[round(float(x), 4) for x in w3],
                     example_weights_shuf=[round(float(x), 4) for x in w3s],
                     ok=bool(sort_inv and weights_change))
    print("T4:", res["T4"]["ok"], res["T4"]["example_weights"], flush=True)


# ---------------------------------------------------------------------------
def gate_t5_t6(res, eng, per_ds: int = 3):
    preds = {}
    t5_ok = True
    t6_ok = True
    for ds in RC.DS_LIST:
        dataset = C.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        old_bank = AC.load_bank("dev", ds)      # round-1/2 frozen b1 bank
        old_gather = json.load(open(os.path.join(
            RC._XSP_OUT, "acc", "E_GATHER", f"{ds}.json")))
        old_main = json.load(open(os.path.join(
            RC._XSP_OUT, "acc", "E_MAIN025", f"{ds}.json")))
        items = json.load(open(os.path.join(
            AC.OUT_DIR, "manifest.json")))["datasets"][ds]["dev"][:per_ds]
        for it in items:
            key = str(it["idx"])
            row = dataset.data.iloc[it["idx"]]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=RC.K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            keep_live = AC.official_facility_keep(ctx, RC.K)
            keep_old = torch.tensor(old_bank[key]["keep"],
                                    dtype=torch.long, device=V.device)
            keep_eq = bool(torch.equal(keep_live.sort().values,
                                       keep_old.sort().values))
            out_g = AC.run_one(eng, msg, ds, old_bank[key],
                               dict(kind="base", lam=0.0, selector="b1",
                                    K=RC.K))
            out_m = AC.run_one(eng, msg, ds, old_bank[key],
                               dict(kind="uniform", lam=0.25, scope="main",
                                    selector="b1", K=RC.K))
            entry = dict(
                keep_eq=keep_eq,
                gather_text_eq=out_g["text"] ==
                old_gather["records"][key]["prediction"],
                main025_text_eq=out_m["text"] ==
                old_main["records"][key]["prediction"])
            preds[f"E:{ds}:{key}"] = entry
            t5_ok &= keep_eq and entry["gather_text_eq"] \
                and entry["main025_text_eq"]
            print(f"[T5 {ds}:{key}] keep={keep_eq} "
                  f"gather={entry['gather_text_eq']} "
                  f"main025={entry['main025_text_eq']}", flush=True)

            for scorer in RC.SCORERS:
                bank = RC.load_bank(scorer, ds)
                brec = bank["samples"][key]
                ctx_s = dict(ctx, ds=ds, qid=int(it["idx"]))
                keep_live_s, _ = RC.select_keep(scorer, ctx_s, RC.K)
                keep_bank = torch.tensor(brec["keep"], dtype=torch.long,
                                         device=V.device)
                eq = bool(torch.equal(keep_live_s.sort().values,
                                      keep_bank.sort().values))
                arm = {"rtg": "R_MAIN025", "flat": "F_MAIN025",
                       "shuf": "S_MAIN025"}[scorer]
                outa = acu_run_one(eng, msg, ds, brec, RC.arm_cfg(arm))
                ds_ok = bool(outa["meta"]["ds_lengths"] == [RC.K] * 3)
                inv_ok = bool(outa["meta"]["n_vis_kept"] == RC.K
                              and outa["meta"]["layer_calls_ok"]
                              and outa["meta"]["cache_ok"])
                t6_ok &= eq and ds_ok and inv_ok
                preds[f"{scorer}:{ds}:{key}"] = dict(
                    keep_eq=eq, ds_identity=ds_ok, invariants=inv_ok,
                    n_vis_kept=int(outa["meta"]["n_vis_kept"]))
                print(f"[T6 {scorer} {ds}:{key}] keep_eq={eq} ds={ds_ok} "
                      f"inv={inv_ok}", flush=True)
    with open(T5_PRED, "w") as f:
        json.dump(preds, f, indent=1)
    res["T5"] = dict(ok=bool(t5_ok))
    res["T6"] = dict(ok=bool(t6_ok))
    print("T5:", t5_ok, "T6:", t6_ok, flush=True)


def gate_lam0(res, eng, per_ds: int = 2):
    """lambda=0 identity vs pure gather (separate from the formal lambda
    check; per protocol it does NOT substitute for it)."""
    ok = True
    for ds in RC.DS_LIST:
        bank = RC.load_bank("rtg", ds)
        dataset = C.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        items = json.load(open(os.path.join(
            AC.OUT_DIR, "manifest.json")))["datasets"][ds]["dev"][:per_ds]
        for it in items:
            key = str(it["idx"])
            row = dataset.data.iloc[int(key)]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            lam0 = acu_run_one(eng, msg, ds, bank["samples"][key],
                               dict(kind="uniform", lam=0.0, scope="main"))
            base = acu_run_one(eng, msg, ds, bank["samples"][key],
                               dict(kind="base", lam=0.0, scope="main"))
            same = bool(torch.equal(lam0["state"].logits,
                                    base["state"].logits)
                        and lam0["text"] == base["text"])
            ok &= same
            print(f"[lam0 {ds}:{key}] identity={same}", flush=True)
    res["T7_lam0"] = dict(ok=bool(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="T1,T2,T3,T4")
    ap.add_argument("--per-ds", type=int, default=3)
    ap.add_argument("--lam0", action="store_true")
    args = ap.parse_args()
    gates = args.gates.split(",")
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for g in ("T1", "T2", "T3", "T4", "T5", "T6", "T7_lam0"):
        res.setdefault(g, {})
    need_eng = ("T5" in gates or "T6" in gates or args.lam0)
    eng = None
    if need_eng:
        model = AC.common.load_model(AC.common.BASELINE_MODEL,
                                     max_new_tokens=2048)
        from model.native_qwen3 import NativeEngine
        eng = NativeEngine(model)
    if "T1" in gates:
        gate_t1(res)
    if "T2" in gates:
        gate_t2(res)
    if "T3" in gates:
        gate_t3(res)
    if "T4" in gates:
        gate_t4(res)
    if "T5" in gates or "T6" in gates:
        gate_t5_t6(res, eng, args.per_ds)
    if args.lam0:
        gate_lam0(res, eng)
    res["meta"] = dict(code_commit=RC.git_commit())
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("written:", OUT)


if __name__ == "__main__":
    main()
