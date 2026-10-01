"""Anchor-Merge Pilot Round 2 — targeted correctness gates.

Covers only the NEW paths of round 2 (stream scope, block8 banks, K=128);
the engine-level G1 gate is unchanged from round 1 and not re-run.
Results -> outputs/anchor_merge_pilot/correctness_round2.json.
"""

from __future__ import annotations

import json
import os

import torch

import amp_common as AC


def main():
    out_path = os.path.join(AC.OUT_DIR, "correctness_round2.json")
    manifest = json.load(open(os.path.join(AC.OUT_DIR, "manifest.json")))
    model = AC.common.load_model(AC.common.BASELINE_MODEL, max_new_tokens=8)
    from model.native_qwen3 import NativeEngine
    eng = NativeEngine(model)
    res = dict(gates={})

    # ---- G-A: lambda-0 per scope == plain gather (logits + predictions) ----
    ga = dict(desc="scope main/ds lam0 vs gather", n=0, ok=True)
    for ds in AC.DS_LIST:
        it = manifest["datasets"][ds]["dev"][0]
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        prep = eng.prepare(msg, ds)
        V, DS = eng.encode(prep)
        bank = AC.load_bank("dev", ds)          # b1@256 legacy bank
        keep = torch.as_tensor(bank[str(it["idx"])]["keep"], dtype=torch.long,
                               device=V.device)
        st_g, _ = None, None
        st_g = eng.prefill(prep, V, DS, keep)
        _, t_g = eng.decode(st_g, 8)
        for scope in ("main", "ds"):
            cfg = dict(kind="uniform", lam=0.0, scope=scope, K=256)
            out = AC.run_one(eng, msg, ds, bank[str(it["idx"])], cfg,
                             max_new_tokens=8)
            d = float((out["state"].logits[0] - st_g.logits[0]).abs().max())
            eq = out["text"] == t_g and d == 0.0
            # feature-level: merged stream at lam0 must equal the raw gather
            if scope == "main":
                y = AC.merge_stream(
                    V, keep, *AC.compute_assignment(V, keep)[:2],
                    "lam0", 0.0)
                eq = eq and bool(torch.equal(y, V[keep]))
            else:
                didx, gid, _ = AC.compute_assignment(V, keep)
                eq = eq and all(torch.equal(
                    AC.merge_stream(d_, keep, didx, gid, "lam0", 0.0),
                    d_[keep]) for d_ in DS)
            ga["n"] += 1
            ga["ok"] = bool(ga["ok"] and eq)
            print(f"[G-A] {ds} scope={scope} logit_d={d:.2e} pred_eq="
                  f"{out['text'] == t_g} ok={eq}", flush=True)
    res["gates"]["GA"] = ga

    # ---- G-B: new banks (b2@256, b1@128, b2@128) == live recomputation ----
    gb = dict(desc="bank == live for b2@256, b1@128, b2@128",
              checked=0, mismatch=0, ok=True)
    for sel, K in (("b2", 256), ("b1", 128), ("b2", 128)):
        for ds in AC.DS_LIST:
            it = manifest["datasets"][ds]["dev"][0]
            dataset = AC.common.build_dataset(ds)
            model.set_dump_image(dataset.dump_image)
            row = dataset.data.iloc[it["idx"]]
            msg = AC.common.build_message(model, dataset, ds, row)
            prep = eng.prepare(msg, ds)
            V, DS = eng.encode(prep)
            text_mean, text_seq = eng.instruction_embeds(msg, ds)
            ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=eng,
                       text_mean=text_mean, text_seq=text_seq)
            from model.e0_selectors import _eadp_parts
            keep_live = _eadp_parts(K, ctx, "facility" if sel == "b1"
                                    else "block8")
            didx_live, gid_live, _ = AC.compute_assignment(V, keep_live)
            bank = AC.load_bank("dev", ds, sel, K)
            rec = bank[str(it["idx"])]
            gb["checked"] += 1
            if rec["keep"] != keep_live.cpu().tolist() or \
                    rec["gid"] != gid_live.cpu().tolist():
                gb["mismatch"] += 1
                gb["ok"] = False
                print(f"[G-B] MISMATCH {sel}@K{K} {ds}", flush=True)
    print(f"[G-B] checked={gb['checked']} mismatch={gb['mismatch']}",
          flush=True)
    res["gates"]["GB"] = gb

    # ---- G-C: invariants for representative new arms ----
    gc = dict(desc="invariants B2BASE/B2U025/K128BASE/B2K128U025",
              ok=True, n=0)
    for arm, sel, K in (("B2BASE", "b2", 256), ("B2U025", "b2", 256),
                        ("K128BASE", "b1", 128), ("B2K128U025", "b2", 128)):
        ds = AC.DS_LIST[0]
        it = manifest["datasets"][ds]["dev"][0]
        dataset = AC.common.build_dataset(ds)
        model.set_dump_image(dataset.dump_image)
        row = dataset.data.iloc[it["idx"]]
        msg = AC.common.build_message(model, dataset, ds, row)
        bank = AC.load_bank("dev", ds, sel, K)
        cfg = AC.arm_cfg(arm)
        cfg["selector"], cfg["K"] = sel, K
        out = AC.run_one(eng, msg, ds, bank[str(it["idx"])], cfg,
                         max_new_tokens=8)
        meta = out["meta"]
        ok = (meta["n_vis_kept"] == K and meta["no_dup"]
              and meta["in_range"] and meta["ascending"]
              and meta.get("ds_lengths") == [K] * 3
              and meta["layer_calls_ok"] and meta["cache_ok"]
              and len(out["keep_idx"]) == K)
        gc["n"] += 1
        gc["ok"] = bool(gc["ok"] and ok)
        print(f"[G-C] {arm} n_vis_kept={meta['n_vis_kept']} ok={ok}",
              flush=True)
    res["gates"]["GC"] = gc

    res["all_passed"] = all(g["ok"] for g in res["gates"].values())
    with open(out_path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"\n[{'PASS' if res['all_passed'] else 'FAIL'}] -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
