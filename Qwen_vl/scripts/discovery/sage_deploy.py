"""
SAGE step 4 -- fresh locked confirmation (decisions 11-13).

Arms, all through the SAME engine, scorer and decode path on the 720 fresh
instances of sage_plan.json:

    B2         incumbent identity reference (its own forward, no injection)
    B1         official EADP facility@256
    SAGE       the live survivor-conditioned critic, frozen (g*, tau*), ONE
               decoder pass; records E(x) + scores for the RND/PERM controls
    UNARY      the same-label unary scorer control, its own frozen tau
    RND        uniform random edge from E(x) on exactly the instances SAGE
               exchanged (matched rate), S0 elsewhere
    PERM       within-image permutation of SAGE's edge scores, argmax, on
               exactly the instances SAGE exchanged (matched rate)
    HINDSIGHT  4 sampled edges per instance, generated -- the realized ceiling

Gold answers enter ONLY per_sample_hits. Nothing here selects anything with a
teacher or a gold answer.

Usage
    python scripts/discovery/sage_deploy.py --arms B2 B1 SAGE UNARY
    python scripts/discovery/sage_deploy.py --arms RND PERM HINDSIGHT
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                       # noqa: E402
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine              # noqa: E402
from m2_accuracy import MAX_NEW                                      # noqa: E402
from m5_common import BUDGET                                         # noqa: E402
from m6_common import load_bank                                      # noqa: E402
from m7_accuracy import SetPruner, install                           # noqa: E402
from scoring import per_sample_hits                                  # noqa: E402
import sage_common as S                                              # noqa: E402
from sage_common import dump_json                                    # noqa: E402
from sage_live import load_critic, install_sage, run_one             # noqa: E402
from sage_live import _lex_argmax                                    # noqa: E402

ARMS = ("B2", "B1", "SAGE", "UNARY", "RND", "PERM", "PERM-X", "HINDSIGHT")


def one_hit(ds, row, prediction):
    return float(per_sample_hits(ds, [row], [prediction])[0])


def build_items(model):
    plan = S.load_json(S.F_PLAN)
    dataset_cache, items = {}, []
    for p in plan["instances"]:
        ds, idx = p["ds"], int(p["idx"])
        if ds not in dataset_cache:
            dataset_cache[ds] = common.build_dataset(ds)
            model.set_dump_image(dataset_cache[ds].dump_image)
        dataset = dataset_cache[ds]
        row = dataset.data.iloc[idx]
        items.append(dict(key=p["key"], ds=ds, row=row,
                          msg=common.build_message(model, dataset, ds, row)))
    return items


def load_arm(arm):
    path = os.path.join(OUTPUT_DIR, S.F_CONF.format(arm=arm))
    if os.path.exists(path):
        return json.load(open(path)), path
    return dict(arm=arm, seed=S.SAGE_SEED, records={}), path


def run_engine_arm(eng, model, items, arm, rec, set_of=None):
    """Arms whose delivered set comes from `set_of(key)`; None = identity.

    Identity arms run the engine's OWN pruner (B2 -> block8, B1 -> facility):
    m7's install() hardwires BASE_SELECTOR, so calling it here would silently
    turn B1 into a second block8 arm (caught: B1 == B2 on all 720).
    """
    pruner = eng.pruner if set_of is None else install(eng, model, None)
    t0 = time.time()
    for rank, it in enumerate(items):
        if it["key"] in rec["records"]:
            continue
        try:
            Se = None if set_of is None else set_of(it["key"], rank)
            pruner.final = Se
            out = run_one(eng, it, MAX_NEW)
            rec["records"][it["key"]] = dict(
                ds=it["ds"], prediction=out["prediction"],
                hit=one_hit(it["ds"], it["row"], out["prediction"]),
                ttft_ms=out["ttft_ms"], exchanged=bool(Se is not None))
            del out
        except Exception:
            rec["records"][it["key"]] = dict(ds=it["ds"], error=True,
                                             err=traceback.format_exc()[-1500:])
            print(f"  [ERROR] {it['key']}", flush=True)
        if len(rec["records"]) % 50 == 0:
            save_arm(arm, rec)
            print(f"  {arm} {len(rec['records'])}/{len(items)} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    save_arm(arm, rec)


def path_of(arm):
    return os.path.join(OUTPUT_DIR, S.F_CONF.format(arm=arm))


def save_arm(arm, rec):
    path = path_of(arm)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B2", "B1", "SAGE", "UNARY"])
    args = ap.parse_args()

    calib = S.load_json(S.F_CALIB)
    bank = load_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    torch.set_grad_enabled(False)
    dev = next(model.model.parameters()).device
    items = build_items(model)
    print(f"[confirmation] {len(items)} instances; arms {args.arms}")

    for arm in args.arms:
        assert arm in ARMS, arm
        rec, _ = load_arm(arm)
        t0 = time.time()
        if arm in ("B2", "B1"):
            selector = "block8" if arm == "B2" else "facility"
            eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                               selector=selector, tag=f"SAGE-{arm}"))
            run_engine_arm(eng, model, items, arm, rec, set_of=None)
        elif arm in ("SAGE", "UNARY"):
            cfg = calib["chosen"] if arm == "SAGE" else calib["unary"]
            pt = (S.F_CRITIC.format(g=cfg["g"], w=cfg["width"]) if arm == "SAGE"
                  else f"sage_unary_g{cfg['g']}_w{cfg['width']}.pt")
            critic, mu, sd, _ = load_critic(pt, dev)
            eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                               selector="block8", tag=f"SAGE-{arm}"))
            pruner = install_sage(eng, model, critic, mu, sd, tau=cfg["tau"],
                                  g=cfg["g"],
                                  family=("unary" if arm == "UNARY" else "set"),
                                  record="full")
            t0 = time.time()
            edges_flat, plus_flat, ptr, scores_all = [], [], [0], []
            for rank, it in enumerate(items):
                if it["key"] in rec["records"]:
                    continue
                try:
                    out = run_one(eng, it, MAX_NEW)
                    sg = pruner.read_sage_ms()
                    dec = pruner.last_sage
                    hit = one_hit(it["ds"], it["row"], out["prediction"])
                    rec["records"][it["key"]] = dict(
                        ds=it["ds"], prediction=out["prediction"], hit=hit,
                        ttft_ms=out["ttft_ms"], sage_ms=sg,
                        accepted=dec["accepted"], pred_gain=dec.get("pred_gain"),
                        n_edges=dec["n_edges"], g=dec["g"],
                        final_idx=dec["final_idx"], s0_idx=dec["s0_idx"])
                    for e in dec.get("edges", []):
                        edges_flat.extend(e["minus"])
                        plus_flat.extend(e["plus"])
                        ptr.append(len(edges_flat))
                    scores_all.extend(dec.get("scores", []))
                    del out
                except Exception:
                    rec["records"][it["key"]] = dict(ds=it["ds"], error=True,
                                                     err=traceback.format_exc()[-1500:])
                    print(f"  [ERROR] {it['key']}", flush=True)
                if len(rec["records"]) % 50 == 0:
                    save_arm(arm, rec)
                    print(f"  {arm} {len(rec['records'])}/{len(items)} "
                          f"{(time.time()-t0)/60:.1f} min", flush=True)
            save_arm(arm, rec)
            if edges_flat:
                # per-KEY edge counts (ptr is per-edge and useless for lookup)
                counts = [rec["records"][it["key"]].get("n_edges", 0)
                          for it in items]
                np.savez(os.path.join(OUTPUT_DIR, f"sage_conf_{arm}_edges.npz"),
                         minus_flat=np.array(edges_flat, dtype=np.int32),
                         plus_flat=np.array(plus_flat, dtype=np.int32),
                         ptr=np.array(ptr, dtype=np.int64),
                         key_edge_counts=np.array(counts, dtype=np.int64),
                         scores=np.array(scores_all, dtype=np.float32),
                         keys=np.array([it["key"] for it in items]))
            print(f"[{arm}] done ({time.time()-t0:.0f}s)")
        else:
            sage_rec = json.load(open(path_of("SAGE")))
            ez = np.load(os.path.join(OUTPUT_DIR, "sage_conf_SAGE_edges.npz"))
            edge_index = {str(k): j for j, k in enumerate(ez["keys"])}
            keys = [it["key"] for it in items]
            cfg = calib["chosen"]
            # per-key edge offsets, in EDGE units; `ptr` is per-edge only
            if "key_edge_counts" in ez:
                counts = ez["key_edge_counts"]
            else:
                counts = [sage_rec["records"][k].get("n_edges", 0) for k in keys]
            key_off = np.concatenate([[0], np.cumsum(counts)])
            n_edge = int(cfg["g"])

            def edges_of(key):
                j = edge_index[key]
                e_lo, e_hi = int(key_off[j]), int(key_off[j + 1])
                lo, hi = e_lo * n_edge, e_hi * n_edge
                return [dict(minus=ez["minus_flat"][a:a + n_edge].tolist(),
                             plus=ez["plus_flat"][a:a + n_edge].tolist())
                        for a in range(lo, hi, n_edge)]

            def scores_of(key):
                j = edge_index[key]
                e_lo, e_hi = int(key_off[j]), int(key_off[j + 1])
                return np.array(ez["scores"][e_lo:e_hi])

            # G-SET gate on the RECONSTRUCTED edges of the first accepted instance
            k0 = next(k for k in keys if sage_rec["records"][k].get("accepted"))
            _s0 = set(sage_rec["records"][k0]["s0_idx"])
            _e0 = edges_of(k0)
            assert _e0 and all(set(e["minus"]) <= _s0
                               and not (set(e["plus"]) & _s0) for e in _e0[:8]), \
                "reconstructed SAGE edges fail the G-SET gate"
            assert len((_s0 - set(_e0[0]["minus"])) | set(_e0[0]["plus"])) == BUDGET

            if arm in ("RND", "PERM", "PERM-X"):
                accepted = {k for k, r in sage_rec["records"].items()
                            if r.get("accepted")}
                if arm == "RND":
                    def set_of(key, rank):
                        if key not in accepted:
                            return None
                        edges = edges_of(key)
                        rng = np.random.default_rng(S.SAGE_SEED * 7 + rank)
                        e = edges[int(rng.integers(len(edges)))]
                        s0 = sage_rec["records"][key]["s0_idx"]
                        return sorted((set(s0) - set(e["minus"])) | set(e["plus"]))
                elif arm == "PERM":
                    # Pre-registered within-image score permutation. KEPT for
                    # the record: it is degenerate under eq. 5 deployment --
                    # argmax of a permuted score list returns the same ELEMENT,
                    # so this arm provably reproduces SAGE set-for-set (the
                    # M5 analysis confirmed: delta exactly 0.000 [0,0]).
                    def set_of(key, rank):
                        if key not in accepted:
                            return None
                        edges = edges_of(key)
                        scores = scores_of(key)
                        assert scores.size == len(edges)
                        rng = np.random.default_rng(S.SAGE_SEED * 11 + rank)
                        perm = rng.permutation(scores.size)
                        j2 = _lex_argmax(scores[perm], [edges[p] for p in perm])
                        e = edges[perm[j2]]
                        s0 = sage_rec["records"][key]["s0_idx"]
                        return sorted((set(s0) - set(e["minus"])) | set(e["plus"]))
                else:  # PERM-X, the amendment control
                    # AMENDMENT (declared before its result was seen): the
                    # within-image permutation above is degenerate; the
                    # non-degenerate control that keeps the idea's intent is a
                    # CROSS-IMAGE rotation (M2's C1-SHUF precedent): image i's
                    # edges are ranked by image (i+7)'s scores, cycled to len,
                    # exchanged on exactly SAGE's accepted set.
                    def set_of(key, rank):
                        if key not in accepted:
                            return None
                        edges = edges_of(key)
                        donor = keys[(rank + 7) % len(keys)]
                        sc = scores_of(donor)
                        s = np.array([sc[i % sc.size]
                                      for i in range(len(edges))])
                        j = _lex_argmax(s, edges)
                        e = edges[j]
                        s0 = sage_rec["records"][key]["s0_idx"]
                        return sorted((set(s0) - set(e["minus"]))
                                      | set(e["plus"]))
                eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                                   selector="block8",
                                                   tag=f"SAGE-{arm}"))
                run_engine_arm(eng, model, items, arm, rec, set_of=set_of)
            else:  # HINDSIGHT
                eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                                   selector="block8",
                                                   tag="SAGE-HIND"))
                pruner = install(eng, model, None)
                b2_rec = json.load(open(path_of("B2")))["records"]
                for rank, it in enumerate(items):
                    if it["key"] in rec["records"]:
                        continue
                    try:
                        edges = edges_of(it["key"])
                        sampled = S.sample_edges(edges, S.N_SAMPLE_CONF,
                                                 S.SAGE_SEED, rank)
                        h0 = b2_rec[it["key"]]["hit"]
                        gens = []
                        for e in sampled:
                            s0 = sage_rec["records"][it["key"]]["s0_idx"]
                            Se = sorted((set(s0) - set(e["minus"])) | set(e["plus"]))
                            assert len(Se) == BUDGET
                            pruner.final = Se
                            out = run_one(eng, it, MAX_NEW)
                            gens.append(dict(minus=e["minus"], plus=e["plus"],
                                             hit=one_hit(it["ds"], it["row"],
                                                         out["prediction"])))
                            del out
                        rec["records"][it["key"]] = dict(
                            ds=it["ds"], h0=h0, sampled=gens)
                    except Exception:
                        rec["records"][it["key"]] = dict(ds=it["ds"], error=True,
                                                         err=traceback.format_exc()[-1500:])
                        print(f"  [ERROR] {it['key']}", flush=True)
                    if len(rec["records"]) % 25 == 0:
                        save_arm(arm, rec)
                        print(f"  HINDSIGHT {len(rec['records'])}/{len(items)} "
                              f"{(time.time()-t0)/60:.1f} min", flush=True)
                save_arm("HINDSIGHT", rec)
        print(f"[{arm}] complete ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
