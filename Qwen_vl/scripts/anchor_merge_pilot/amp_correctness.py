"""Anchor-Merge Pilot — correctness gates G1-G5 (protocol §5).

No formal accuracy run is allowed until every gate passes.  Results ->
outputs/anchor_merge_pilot/correctness.json.  Re-running the script compares
merge-feature hashes / predictions with the previous invocation (cross-process
reproducibility, G5).

Usage: /home/dell/miniconda3/envs/qwen3vl_clean/bin/python amp_correctness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

import amp_common as AC

N_SAMPLES_PER_DS = 4
DECODE_STEPS = 32


def stock_reference(engine, prep, n_steps):
    """Stock prefill + n_steps greedy decode logits through the untouched
    ForConditionalGeneration.forward (verbatim from e0_gates)."""
    hf = engine.model
    inputs = prep["inputs"]
    out = hf(**inputs)
    ref_prefill = out.logits[0, -1].clone()
    past = out.past_key_values
    L = int(prep["ids"].shape[1])
    tok = int(ref_prefill.argmax().item())
    steps = []
    for n in range(n_steps):
        out = hf(input_ids=torch.tensor([[tok]], device=ref_prefill.device),
                 attention_mask=torch.ones(1, L + n + 1, dtype=torch.long,
                                           device=ref_prefill.device),
                 past_key_values=past,
                 cache_position=torch.tensor([L + n], device=ref_prefill.device))
        lg = out.logits[0, -1].clone()
        steps.append(lg)
        tok = int(lg.argmax().item())
    return ref_prefill, steps


def feat_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                          ).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-ds", type=int, default=N_SAMPLES_PER_DS)
    args = ap.parse_args()

    out_path = os.path.join(AC.OUT_DIR, "correctness.json")
    prev = json.load(open(out_path)) if os.path.exists(out_path) else None

    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=64)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    K = AC.K

    res: dict = dict(base_commit=manifest["base_commit"], gates={})

    # ---------------------------------------------------------------- G1 --
    g1 = dict(desc="keep-all native vs stock (prefill + 32 decode logits)",
              max_d_prefill=0.0, max_d_decode=0.0, token_mismatch=0, ok=True)
    for ds in AC.DS_LIST:
        rows = manifest["datasets"][ds]["dev"][:args.samples_per_ds]
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in rows:
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            ref_pre, ref_steps = stock_reference(eng, prep, DECODE_STEPS)
            ref_toks = [int(s.argmax().item()) for s in ref_steps]
            V, DS = eng.encode(prep)
            st = eng.prefill(prep, V, DS, None, full_lm_head=True)
            gen_ids, _ = eng.decode(st, DECODE_STEPS + 1, ignore_eos=True)
            d_pre = float((st.logits[0] - ref_pre).abs().max().item())
            d_dec = max((float((lg[0] - rlg).abs().max().item())
                         for lg, rlg in zip(st.decode_logits, ref_steps)),
                        default=0.0)
            mism = int(ref_pre.argmax().item() != gen_ids[0]) + \
                sum(int(gen_ids[1 + n] != ref_toks[n])
                    for n in range(DECODE_STEPS))
            ok = d_pre <= 1e-3 and d_dec <= 1e-3 and mism == 0
            g1["max_d_prefill"] = max(g1["max_d_prefill"], d_pre)
            g1["max_d_decode"] = max(g1["max_d_decode"], d_dec)
            g1["token_mismatch"] += mism
            g1["ok"] = bool(g1["ok"] and ok)
            print(f"[G1] {ds}_{it['idx']} d_pre={d_pre:.2e} d_dec={d_dec:.2e} "
                  f"mism={mism} ok={ok}", flush=True)
    res["gates"]["G1"] = g1

    # ------------------------------------------------- G2/G4/G5 samples ----
    samples = []
    for ds in AC.DS_LIST:
        rows = manifest["datasets"][ds]["dev"][:args.samples_per_ds]
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        for it in rows:
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            samples.append(dict(ds=ds, idx=it["idx"], message=msg))

    # ---------------------------------------------------------------- G2 --
    g2 = dict(desc="merge-code lam0 == b1 plain gather (features, logits, "
                   "prediction); bank == live recomputation",
              feat_bitexact=0, feat_total=0, logit_max_diff=0.0,
              pred_mismatch=0, bank_mismatch=0, ok=True)
    for s in samples:
        prep = eng.prepare(s["message"], s["ds"])
        V, DS = eng.encode(prep)
        text_mean, text_seq = eng.instruction_embeds(s["message"], s["ds"])
        ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                   text_mean=text_mean, text_seq=text_seq)
        keep = AC.official_facility_keep(ctx, K)
        dropped_idx, gid, _ = AC.compute_assignment(V, keep)

        # bank == live (bank built in a separate process)
        bank = AC.load_bank("dev", s["ds"])
        rec = bank[str(s["idx"])]
        if rec["keep"] != keep.cpu().tolist() or rec["gid"] != gid.cpu().tolist():
            g2["bank_mismatch"] += 1
            g2["ok"] = False

        # lambda-0 through the merge code vs plain gather
        y0 = AC.merge_stream(V, keep, dropped_idx, gid, "lam0", 0.0)
        ys0 = [AC.merge_stream(d, keep, dropped_idx, gid, "lam0", 0.0)
               for d in DS]
        g2["feat_total"] += 1 + len(DS)
        same = (torch.equal(y0, V[keep])
                and all(torch.equal(y0_, d[keep])
                        for y0_, d in zip(ys0, DS)))
        g2["feat_bitexact"] += int(same)

        st_gather = eng.prefill(prep, V, DS, keep)
        st_lam0 = eng.prefill(prep, V, DS, keep, V_sel=y0, DS_sel=ys0)
        d = float((st_gather.logits[0] - st_lam0.logits[0]).abs().max().item())
        g2["logit_max_diff"] = max(g2["logit_max_diff"], d)
        _, t_gather = eng.decode(st_gather, 16)
        _, t_lam0 = eng.decode(st_lam0, 16)
        if t_gather != t_lam0:
            g2["pred_mismatch"] += 1
            g2["ok"] = False
        if not same or d > 0:
            g2["ok"] = False
        print(f"[G2] {s['ds']}_{s['idx']} feat_bitexact={same} d_logit={d:.2e} "
              f"pred_eq={t_gather == t_lam0}", flush=True)
    res["gates"]["G2"] = g2

    # ---------------------------------------------------------------- G3 --
    g3 = dict(desc="group_sum/softmax merge vs naive per-group reference "
                   "(synthetic fp32 + one real sample)",
              max_rel_syn=0.0, max_rel_real=0.0, ok=True)
    torch.manual_seed(0)
    Ns, Ks, D = 64, 16, 32
    X = torch.randn(Ns, D)
    keep_s = torch.randperm(Ns)[:Ks].sort().values
    didx, gids, _ = AC.compute_assignment(X, keep_s)
    # naive uniform
    ref = torch.zeros(Ks, D)
    for j in range(Ks):
        members = [X[int(keep_s[j])]] + \
            [X[int(didx[i])] for i in range(didx.numel()) if int(gids[i]) == j]
        ref[j] = torch.stack(members).mean(0)
    y = AC.merge_stream(X, keep_s, didx, gids, "uniform", 0.25)
    a_s = X[keep_s]
    y_ref = a_s + 0.25 * (ref - a_s)
    rel = float((y.double() - y_ref.double()).abs().max()
                / y_ref.abs().max().clamp_min(1e-6))
    g3["max_rel_syn"] = rel
    g3["ok"] = bool(g3["ok"] and rel < 1e-5)
    # naive sim-softmax
    tau = 0.1
    sim = AC.compute_assignment(X, keep_s)[2]
    ref2 = torch.zeros(Ks, D)
    for j in range(Ks):
        idxs = [i for i in range(didx.numel()) if int(gids[i]) == j]
        logits = [1.0 / tau] + [float(sim[i, j]) / tau for i in idxs]
        xs = [X[int(keep_s[j])]] + [X[int(didx[i])] for i in idxs]
        lg = torch.tensor(logits)
        w = torch.softmax(lg, dim=0)
        ref2[j] = (torch.stack(xs) * w.unsqueeze(1)).sum(0)
    y2 = AC.merge_stream(X, keep_s, didx, gids, "sim", 0.25, tau, sim)
    y2ref = X[keep_s] + 0.25 * (ref2 - X[keep_s])
    rel2 = float((y2.double() - y2ref.double()).abs().max()
                 / y2ref.abs().max().clamp_min(1e-6))
    g3["max_rel_syn"] = max(g3["max_rel_syn"], rel2)
    g3["ok"] = bool(g3["ok"] and rel2 < 1e-5)

    # real sample, uniform merge vs naive loop
    s = samples[0]
    prep = eng.prepare(s["message"], s["ds"])
    V, DS = eng.encode(prep)
    text_mean, text_seq = eng.instruction_embeds(s["message"], s["ds"])
    ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
               text_mean=text_mean, text_seq=text_seq)
    keep = AC.official_facility_keep(ctx, K).cpu()
    Vc = V.cpu().float()                 # FP32 domain: tests the MATH; the
    didx, gids, _ = AC.compute_assignment(Vc, keep)   # bf16 output cast is
    j0 = int(gids[0])                    # checked separately below
    members = [Vc[int(keep[j0])]] + \
        [Vc[int(didx[i])] for i in range(didx.numel())
         if int(gids[i]) == j0]
    ref_m = torch.stack(members).float().mean(0)
    yv = AC.merge_stream(Vc, keep, didx, gids, "uniform", 0.25)[j0]
    a_real = Vc[int(keep[j0])]
    y_ref_real = a_real + 0.25 * (ref_m - a_real)
    relr = float((yv.double() - y_ref_real.double()).abs().max()
                 / y_ref_real.abs().max().clamp_min(1e-6))
    # bf16 output cast: the merged fp32 result rounded to bf16 must match the
    # production path (bf16 in -> fp32 acc -> bf16 out) bit-for-bit
    y_bf16 = AC.merge_stream(V.cpu(), keep, didx, gids, "uniform", 0.25)[j0]
    cast_ok = bool(torch.equal(y_bf16,
                               yv.to(torch.bfloat16)))
    g3["max_rel_real"] = relr
    g3["real_bf16_cast_bitexact"] = cast_ok
    g3["ok"] = bool(g3["ok"] and relr < 1e-5 and cast_ok)
    print(f"[G3] rel_syn={g3['max_rel_syn']:.2e} rel_real={relr:.2e} "
          f"bf16_cast={cast_ok}", flush=True)
    res["gates"]["G3"] = g3

    # ---------------------------------------------------------------- G4 --
    g4 = dict(desc="degenerate cases + per-sample invariants (256 visual "
                   "tokens, each dropped token exactly one group, layer/cache "
                   "bookkeeping)",
              syn_ok=False, invariants_ok=0, invariants_total=0, ok=True)
    # synthetic: N == K -> identity
    Xs = torch.randn(8, 5)
    ks = torch.arange(8)
    didx0, gid0, _ = AC.compute_assignment(Xs, ks)
    y_ident = AC.merge_stream(Xs, ks, didx0, gid0, "uniform", 1.0)
    ident = bool(torch.equal(y_ident, Xs))
    # synthetic: singleton groups -> y == a
    Xb = torch.randn(10, 4)
    kb = torch.arange(0, 10, 2)
    didx1, gid1, _ = AC.compute_assignment(Xb, kb)
    nb = didx1.numel()
    gsz = torch.bincount(gid1, minlength=kb.numel())
    yb = AC.merge_stream(Xb, kb, didx1, gid1, "uniform", 0.5)
    single = gsz == 0
    single_ok = bool((yb[single] == Xb[kb[single]]).all()) if single.any() \
        else True
    cover = bool(didx1.numel() == 10 - kb.numel()
                 and int(gid1.min()) >= 0 and int(gid1.max()) < kb.numel())
    g4["syn_ok"] = ident and single_ok and cover
    g4["ok"] = bool(g4["ok"] and g4["syn_ok"])
    print(f"[G4] synthetic: ident={ident} singleton_ok={single_ok} "
          f"cover={cover}", flush=True)

    for s in samples:
        prep = eng.prepare(s["message"], s["ds"])
        V, DS = eng.encode(prep)
        bank = AC.load_bank("dev", s["ds"])
        rec = bank[str(s["idx"])]
        cfg = AC.arm_cfg("U025")
        out = AC.run_one(eng, s["message"], s["ds"], rec, cfg,
                         max_new_tokens=16)
        meta = out["meta"]
        N = prep["n_vis"]
        gid_t = torch.tensor(rec["gid"])
        counts = torch.bincount(gid_t, minlength=K)
        inv_ok = (meta["n_vis_kept"] == K
                  and meta["no_dup"] and meta["in_range"]
                  and meta["ascending"]
                  and meta.get("ds_lengths") == [K] * len(DS)
                  and meta["layer_calls_ok"] and meta["cache_ok"]
                  and int(counts.sum()) == N - K
                  and len(out["keep_idx"]) == K)
        g4["invariants_total"] += 1
        g4["invariants_ok"] += int(bool(inv_ok))
        if not inv_ok:
            g4["ok"] = False
        print(f"[G4] {s['ds']}_{s['idx']} invariants_ok={inv_ok}", flush=True)
    res["gates"]["G4"] = g4

    # ---------------------------------------------------------------- G5 --
    g5 = dict(desc="in-process repeat (predictions identical) + cross-process "
                   "hash comparison",
              inproc_pred_mismatch=0, inproc_total=0,
              crossprocess_equal=None, ok=True)
    repro = {}
    for s in samples[:3]:
        prep = eng.prepare(s["message"], s["ds"])
        V, DS = eng.encode(prep)
        bank = AC.load_bank("dev", s["ds"])
        rec = bank[str(s["idx"])]
        cfg = AC.arm_cfg("U025")
        dropped_idx, gid, sim = AC.compute_assignment(V, rec_keep(rec, V.device))
        h = feat_hash(AC.merge_stream(V, rec_keep(rec, V.device), dropped_idx,
                                      gid, "uniform", 0.25))
        o1 = AC.run_one(eng, s["message"], s["ds"], rec, cfg, max_new_tokens=16)
        o2 = AC.run_one(eng, s["message"], s["ds"], rec, cfg, max_new_tokens=16)
        key = f"{s['ds']}_{s['idx']}"
        g5["inproc_total"] += 1
        if o1["text"] != o2["text"]:
            g5["inproc_pred_mismatch"] += 1
            g5["ok"] = False
        repro[key] = dict(merge_hash=h, pred=o1["text"])
        print(f"[G5] {key} inproc_eq={o1['text'] == o2['text']} hash={h}",
              flush=True)
    prev_repro = (prev or {}).get("repro") if prev else None
    if prev_repro:
        g5["crossprocess_equal"] = all(
            prev_repro.get(k, {}).get("merge_hash") == v["merge_hash"]
            and prev_repro.get(k, {}).get("pred") == v["pred"]
            for k, v in repro.items())
        if not g5["crossprocess_equal"]:
            g5["ok"] = False
    res["gates"]["G5"] = g5
    res["repro"] = repro

    res["all_passed"] = all(g["ok"] for g in res["gates"].values())
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"\n[{'PASS' if res['all_passed'] else 'FAIL'}] "
          f"{ {k: v['ok'] for k, v in res['gates'].items()} } -> {out_path}",
          flush=True)


def rec_keep(rec, device):
    return torch.as_tensor(rec["keep"], dtype=torch.long, device=device)


if __name__ == "__main__":
    main()
