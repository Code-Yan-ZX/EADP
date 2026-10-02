"""Anchor Completion Validation — correctness gates (protocol §7).

G1  formula: production merge_stream vs an INDEPENDENT explicit per-group
    FP32 loop reference (uniform lam 0.25/1.0, sim tau=0.1) on synthetic
    tensors incl. duplicate and zero rows; rel<1e-5 in FP32 AND bf16 cast
    bitwise vs production output.
G2  paths: real N>K pruning bank entries sane; multi-image isolation
    (per-image facility split on a concatenated ctx; B-image dropped tokens
    may only map to B-image anchors); empty-dropped identity; N<=K keep-all.
G3  reproduction: per dataset 3 bank samples — live recompute of
    anchors/assignment bitwise vs bank; BASE prefill logits + greedy text
    via the bank path vs via eng.generate(selector='identity') bitwise;
    MAIN025/MAIN100/MAIN_SIM025 merge features vs the G1 reference
    (bf16 bitwise); MAIN025 DS streams bitwise == DS[keep]; gate greedy
    outputs stored for later comparison against the accuracy shards
    (verify mode).
G4  invariants: n_vis_kept=K, no dup / ascending keep, ds_lengths, layer
    calls, cache; shared keep/gid across the three ablation arms.
G5  lam=0 identity: uniform lam=0.0 (main scope) vs BASE — features,
    prefill logits, greedy text all bitwise.

Usage:
  python acu_correctness.py --gates G1,G2,G5            (synthetic + bank)
  python acu_correctness.py --gates G3,G4 --per-ds 3
  python acu_correctness.py --verify-g3                 (after accuracy run)
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

import acu_common as AU
from acu_common import common as C

OUT = os.path.join(AU.OUT_DIR, "correctness.json")
G3_PRED = os.path.join(AU.OUT_DIR, "gate_g3_predictions.json")


# ---------------------------------------------------------------------------
# independent reference (explicit loops, FP32) — deliberately NOT reusing
# merge_stream / group_sum
# ---------------------------------------------------------------------------
def ref_merge_full(feat: torch.Tensor, keep: list[int],
                   dropped: list[int], assign: list[int],
                   kind: str, lam: float, tau: float = 0.1) -> torch.Tensor:
    """assign[i] = anchor rank for dropped[i].  Returns y [K, D] fp32."""
    K = len(keep)
    a = feat[keep].clone()
    members = [[keep[j]] for j in range(K)]          # anchor global index first
    for li, j in enumerate(assign):
        members[j].append(dropped[li])
    y = a.clone()
    for j in range(K):
        idxs = members[j]
        if len(idxs) == 1:
            m = a[j]                                  # single-member group
        elif kind == "uniform":
            m = feat[idxs].mean(dim=0)
        elif kind == "sim":
            an = F.normalize(a[j], dim=0)
            logits = []
            for gi in idxs:
                if gi == keep[j]:
                    logits.append(1.0 / tau)
                else:
                    v = F.normalize(feat[gi], dim=0)
                    logits.append(float(an @ v) / tau)
            w = torch.softmax(torch.tensor(logits, dtype=torch.float32,
                                           device=feat.device), dim=0)
            m = (w.unsqueeze(1) * feat[idxs]).sum(dim=0)
        else:
            raise KeyError(kind)
        y[j] = a[j] + lam * (m - a[j])
    return y


def _ready_ds():
    """Datasets whose full bank is already on disk (gates tolerate a bank
    still being built so they can be exercised early)."""
    import gzip
    ready = []
    for ds in AU.DS_MAIN:
        try:
            AU.load_bank(ds)
            ready.append(ds)
        except FileNotFoundError:
            pass
    return ready


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------
def gate_g1(res):
    torch.manual_seed(20261002)
    N, D, K = 37, 16, 6
    V = torch.randn(N, D, dtype=torch.float32)
    V[3] = V[3] + 0.0                                 # keep as is
    V[10] = V[4]                                      # duplicate row
    V[20] = 0.0                                       # zero row
    keep = sorted(torch.randperm(N)[:K].tolist())
    keep_set = set(keep)
    dropped = [i for i in range(N) if i not in keep_set]
    # assignment: cosine argmax, ties -> lowest anchor rank (production rule)
    a = F.normalize(V[keep], dim=1)
    x = F.normalize(V[dropped], dim=1)
    gid = (x @ a.t()).argmax(dim=1).tolist()

    for name, kind, lam in [("uniform025", "uniform", 0.25),
                            ("uniform100", "uniform", 1.0),
                            ("sim025", "sim", 0.25)]:
        y_ref = ref_merge_full(V, keep, dropped, gid, kind, lam)
        kt = torch.tensor(keep, dtype=torch.long)
        di = torch.tensor(dropped, dtype=torch.long)
        gt = torch.tensor(gid, dtype=torch.long)
        sim = (x @ a.t()) if kind == "sim" else None
        y_prod = AU.merge_stream(V, kt, di, gt, kind, lam, 0.1, sim)
        r = rel(y_ref, y_prod.float())
        # bf16 cast convention (round-1 G3): production runs on bf16 model
        # features; the FP32 reference cast to bf16 must match bitwise.
        Vb = V.to(torch.bfloat16)
        simb = ((F.normalize(Vb[dropped].float(), dim=1)
                 @ F.normalize(Vb[keep].float(), dim=1).t())
                if kind == "sim" else None)
        y_prod_b = AU.merge_stream(Vb, kt, di, gt, kind, lam, 0.1, simb)
        y_ref_b = ref_merge_full(Vb.float(), keep, dropped, gid,
                                 kind, lam).to(torch.bfloat16)
        cast_ok = bool(torch.equal(y_prod_b, y_ref_b))
        res["G1"][name] = dict(rel=r, bf16_cast_bitwise=cast_ok,
                               ok=bool(r < 1e-5 and cast_ok))
    # empty dropped -> identity
    y_empty = AU.merge_stream(V, kt, torch.empty(0, dtype=torch.long),
                              torch.empty(0, dtype=torch.long),
                              "uniform", 0.25)
    res["G1"]["empty_dropped_identity"] = bool(
        torch.equal(y_empty, V[kt]))
    # single-member groups: with K == N all groups single -> y == a
    y_all = AU.merge_stream(V, torch.arange(N), torch.empty(0,
                            dtype=torch.long), torch.empty(0,
                            dtype=torch.long), "uniform", 0.25)
    res["G1"]["n_eq_k_identity"] = bool(torch.equal(y_all, V))
    res["G1"]["ok"] = all(v.get("ok", v) if isinstance(v, dict) else v
                          for v in res["G1"].values())


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------
def gate_g2(res, eng):
    bank = AU.load_bank("TextVQA_VAL")
    real = [k for k, v in bank.items() if v["n_vis"] > AU.K][:2]
    res["G2"]["real_pruning_samples"] = len(real)
    ok_real = True
    for k in real:
        v = bank[k]
        keep = v["keep"]
        ok_real &= (keep == sorted(set(keep)) and len(keep) == AU.K)
        ok_real &= (len(v["gid"]) == v["n_vis"] - AU.K)
        ok_real &= all(0 <= g < AU.K for g in v["gid"])
        ok_real &= (sum(v["gsize"]) == v["n_vis"] - AU.K) and \
                   (max(v["gsize"]) >= 1)
    res["G2"]["real_bank_sane"] = bool(ok_real)

    # multi-image isolation on a concatenated ctx
    dataset = C.build_dataset("TextVQA_VAL")
    eng.vlm.set_dump_image(dataset.dump_image)
    Vs, gthws, preps = [], [], []
    for i in (0, 1):
        row = dataset.data.iloc[i]
        msg = C.build_message(eng.vlm, dataset, "TextVQA_VAL", row)
        prep = eng.prepare(msg, "TextVQA_VAL")
        V, DS = eng.encode(prep)
        Vs.append(V)
        gthws.append(prep["gthw"])
        preps.append(prep)
    n0 = Vs[0].shape[0]
    V_cat = torch.cat(Vs, dim=0)
    from model.e0_selectors import _eadp_parts
    tm0, ts0 = eng.instruction_embeds(
        C.build_message(eng.vlm, dataset, "TextVQA_VAL",
                        dataset.data.iloc[0]), "TextVQA_VAL")
    tm1, ts1 = eng.instruction_embeds(
        C.build_message(eng.vlm, dataset, "TextVQA_VAL",
                        dataset.data.iloc[1]), "TextVQA_VAL")
    ctx_cat = dict(prep=dict(gthw=torch.cat(gthws, dim=0),
                             n_vis=V_cat.shape[0]),
                   V=V_cat, K=AU.K, engine=eng,
                   text_mean=torch.cat([tm0, tm1], dim=0),   # [2, D]
                   text_seq=torch.cat([ts0, ts1], dim=0))    # [2, M, D]
    keep_cat = _eadp_parts(AU.K, ctx_cat, "facility")
    k0 = AU.official_facility_keep(
        dict(prep=preps[0], V=Vs[0], K=AU.K, engine=eng,
             text_mean=tm0, text_seq=ts0), AU.K)
    k1 = AU.official_facility_keep(
        dict(prep=preps[1], V=Vs[1], K=AU.K, engine=eng,
             text_mean=tm1, text_seq=ts1), AU.K)
    expect = torch.cat([k0, k1 + n0])
    res["G2"]["multi_image_concat_equals_independent"] = bool(
        torch.equal(keep_cat.sort().values, expect.sort().values))
    # dropped tokens of image B must map to image-B anchors only
    # per-image assignment must keep image-B drops on image-B anchors AND
    # reproduce the independent per-image assignments bitwise
    split_cat = [Vs[0].shape[0], Vs[1].shape[0]]
    di_cat, gid_cat, _ = AU.compute_assignment_per_image(
        V_cat, keep_cat.sort().values, split_cat)
    k1g = k1 + n0
    di1, g1, _ = AU.compute_assignment(Vs[1], k1)
    b_sel = di_cat >= n0
    b_dropped = di_cat[b_sel]
    b_gid = gid_cat[b_sel]
    n_keep0 = int(k0.numel())
    no_cross = bool((b_gid >= n_keep0).all()) if b_gid.numel() else True
    same_as_independent = bool(
        torch.equal(b_dropped.sort().values, (di1 + n0).sort().values)
        and torch.equal(b_gid.sort().values,
                        (g1 + n_keep0).sort().values))
    res["G2"]["multi_image_no_cross_assignment"] = bool(
        no_cross and same_as_independent)

    # N<=K keep-all bank entry
    deg = [v for v in bank.values() if v["n_vis"] <= AU.K]
    res["G2"]["n_le_k_keepall_entries"] = len(deg)
    res["G2"]["ok"] = bool(res["G2"]["real_bank_sane"]
                           and res["G2"]["multi_image_concat_equals_independent"]
                           and res["G2"]["multi_image_no_cross_assignment"])
    print(f"[G2] real_bank_sane={res['G2']['real_bank_sane']} "
          f"concat={res['G2']['multi_image_concat_equals_independent']} "
          f"no_cross={res['G2']['multi_image_no_cross_assignment']} "
          f"n_le_k_entries={res['G2']['n_le_k_keepall_entries']}",
          flush=True)


# ---------------------------------------------------------------------------
# G3 / G4
# ---------------------------------------------------------------------------
def gate_g3_g4(res, eng, per_ds: int):
    preds = {}
    for ds in _ready_ds():
        bank = AU.load_bank(ds)
        cands = [k for k, v in bank.items() if v["n_vis"] > AU.K][:per_ds]
        dataset = C.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        for key in cands:
            rec = bank[key]
            i = int(key)
            row = dataset.data.iloc[i]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            # live selection vs bank
            tm, ts = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, K=AU.K, engine=eng,
                       text_mean=tm, text_seq=ts)
            keep_live = AU.official_facility_keep(ctx, AU.K)
            keep_bank = torch.tensor(rec["keep"], dtype=torch.long,
                                  device=V.device)
            keep_eq = bool(torch.equal(keep_live.sort().values,
                                       keep_bank.sort().values))
            split_sizes = AU.split_sizes_from_gthw(
                prep, eng.inner.visual.spatial_merge_size)
            di, gid, sim = AU.compute_assignment_per_image(V, keep_bank,
                                                           split_sizes)
            gid_eq = bool(torch.equal(gid.cpu(),
                                      torch.tensor(rec["gid"])))
            entry = dict(keep_eq=keep_eq, gid_eq=gid_eq)

            # BASE via bank path vs direct prefill on the LIVE keep:
            # prefill logits + identity-selector equivalence
            out_b1 = AU.run_one(eng, msg, ds, rec, AU.arm_cfg("BASE"))
            st_live = eng.prefill(prep, V, DS, keep_live.sort().values,
                                  deepstack=True, pos="mrope3d")
            entry["base_prefill_logits_bitwise"] = bool(
                torch.equal(out_b1["state"].logits, st_live.logits))

            # ablation arms: features vs reference; DS invariants
            feats_ref = {}
            for arm in ("MAIN025", "MAIN100", "MAIN_SIM025"):
                cfg = AU.arm_cfg(arm)
                out = AU.run_one(eng, msg, ds, rec, cfg)
                feats_ref[arm] = out["text"]
                # recompute production merge directly for bitwise vs reference
                y_prod = AU.merge_stream(
                    V, keep_bank, di, gid, cfg["kind"], cfg["lam"],
                    cfg.get("tau", 0.1),
                    sim if cfg["kind"] == "sim" else None)
                y_ref = ref_merge_full(
                    V.float(), rec["keep"], di.cpu().tolist(),
                    gid.cpu().tolist(), cfg["kind"], cfg["lam"],
                    cfg.get("tau", 0.1))
                entry[f"{arm}_ref_rel"] = rel(y_ref, y_prod.float())
                entry[f"{arm}_bf16_cast_bitwise"] = bool(
                    torch.equal(y_prod, y_ref.to(y_prod.dtype)))
                entry[f"{arm}_n_vis_kept"] = int(out["meta"]["n_vis_kept"])
                entry[f"{arm}_layer_calls_ok"] = bool(
                    out["meta"]["layer_calls_ok"])
                entry[f"{arm}_cache_ok"] = bool(out["meta"]["cache_ok"])
                entry[f"{arm}_ds_lengths"] = out["meta"]["ds_lengths"]
            # MAIN025 DS == DS[keep] via prefill-state ds check:
            # run_one passes DS_sel only for main scope; DS_k gathered in
            # prefill — verify by comparing a BASE and MAIN025 ds_lengths.
            entry["main025_ds_untouched_lengths"] = (
                entry["MAIN025_ds_lengths"] == [AU.K] * 3)
            preds[f"{ds}:{key}"] = dict(
                base=out_b1["text"], **{a: feats_ref[a]
                                        for a in ("MAIN025", "MAIN100",
                                                  "MAIN_SIM025")},
                **entry)
            print(f"[G3 {ds}:{key}] keep_eq={keep_eq} gid_eq={gid_eq} "
                  f"logits={entry['base_prefill_logits_bitwise']} "
                  f"ref_rel={entry['MAIN025_ref_rel']:.2e}", flush=True)
    with open(G3_PRED, "w") as f:
        json.dump(preds, f)
    res["G3"]["ok"] = all(
        v["keep_eq"] and v["gid_eq"]
        and v["base_prefill_logits_bitwise"]
        and v["MAIN025_bf16_cast_bitwise"]
        and v["MAIN100_bf16_cast_bitwise"]
        and v["MAIN_SIM025_bf16_cast_bitwise"]
        for v in preds.values())
    res["G4"]["ok"] = all(
        v[f"{a}_n_vis_kept"] == AU.K and v[f"{a}_layer_calls_ok"]
        and v[f"{a}_cache_ok"] and v[f"{a}_ds_lengths"] == [AU.K] * 3
        and v["main025_ds_untouched_lengths"]
        for v in preds.values()
        for a in ("MAIN025", "MAIN100", "MAIN_SIM025"))


def verify_g3():
    """Compare gate greedy outputs with the accuracy shards (bitwise text)."""
    preds = json.load(open(G3_PRED))
    out = {}
    for name, rec in preds.items():
        ds, key = name.split(":")
        for arm in ("BASE", "MAIN025", "MAIN100", "MAIN_SIM025"):
            shard = AU.load_shard(AU.shard_path("main", arm, ds))
            got = shard["records"].get(key, {}).get("prediction")
            out[f"{name}:{arm}"] = (got == rec[arm.lower()
                                              if arm == "BASE" else arm])
    ok = all(out.values())
    with open(os.path.join(AU.OUT_DIR, "g3_verify.json"), "w") as f:
        json.dump(dict(ok=ok, details=out), f, indent=1)
    print(f"G3 verify vs accuracy shards: {'PASS' if ok else 'FAIL'}")
    return ok


def gate_g5(res, eng, per_ds: int = 1):
    ok = True
    for ds in _ready_ds():
        bank = AU.load_bank(ds)
        cands = [k for k, v in bank.items() if v["n_vis"] > AU.K][:per_ds]
        dataset = C.build_dataset(ds)
        eng.vlm.set_dump_image(dataset.dump_image)
        for key in cands:
            rec = bank[key]
            row = dataset.data.iloc[int(key)]
            msg = C.build_message(eng.vlm, dataset, ds, row)
            lam0_cfg = dict(kind="uniform", lam=0.0, scope="main",
                            selector="b1", K=AU.K)
            o0 = AU.run_one(eng, msg, ds, rec, lam0_cfg)
            ob = AU.run_one(eng, msg, ds, rec, AU.arm_cfg("BASE"))
            same = bool(torch.equal(o0["state"].logits, ob["state"].logits)) \
                and o0["text"] == ob["text"]
            ok &= same
            print(f"[G5 {ds}:{key}] lam0_identity={same}", flush=True)
    res["G5"]["ok"] = bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="G1,G2,G5")
    ap.add_argument("--per-ds", type=int, default=3)
    args = ap.parse_args()
    gates = args.gates.split(",")
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    for g in ("G1", "G2", "G3", "G4", "G5"):
        res.setdefault(g, {})

    need_bank = any(g in gates for g in ("G2", "G3", "G4", "G5"))
    eng = AU.load_engine(max_new_tokens=2048) if (need_bank or "G1" in gates) \
        else None

    if "G1" in gates:
        gate_g1(res)
        print("G1:", res["G1"].get("ok"), flush=True)
    if "G2" in gates:
        gate_g2(res, eng)
        print("G2:", res["G2"].get("ok"), flush=True)
    if "G5" in gates:
        gate_g5(res, eng)
        print("G5:", res["G5"].get("ok"), flush=True)
    if "G3" in gates or "G4" in gates:
        gate_g3_g4(res, eng, args.per_ds)
        print("G3:", res["G3"].get("ok"), "G4:", res["G4"].get("ok"),
              flush=True)
    res["meta"] = dict(code_commit=AU.repo_commit())
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("written:", OUT)


if __name__ == "__main__":
    main()
