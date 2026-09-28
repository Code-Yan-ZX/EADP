"""
SAGE step 1 -- paired answer-effect labels on the image-disjoint fit/val splits.

For every bank image (fit 240 / val 60):
    y(S0)   one B2-identity generation (SetPruner final=None -- byte-identical
            to the incumbent by construction, M7's convention);
    y(Se)   one generation per sampled edge, for each g in G_GRID -- 8 edges per
            (image, g), uniform over E(x) (decision 5);
    Delta   official per-sample score of Se minus S0 (decision 15: gold answers
            touch ONLY this offline labeling).

Also captures the mean instruction embedding q̄ for every processed image
(sage_qbank.npz) -- the one critic input the bank does not store.

Incremental + resumable: the JSON on disk is rewritten atomically after every
image, and a resumed run skips images whose record is complete.

Usage
    python scripts/discovery/sage_label.py --split fit
    python scripts/discovery/sage_label.py --split val
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


def one_hit(ds: str, row, prediction: str) -> float:
    return float(per_sample_hits(ds, [row], [prediction])[0])


def complete(rec: dict, key: str, g_list) -> bool:
    if key not in rec["s0_hits"]:
        return False
    for g in g_list:
        n = len(rec["edges"].get(str(g), {}).get(key, []))
        if n < S.N_SAMPLE_FIT:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("fit", "val"), required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="sanity mode: only the first N images")
    args = ap.parse_args()

    bank = load_bank()
    n_bank = bank["n"]
    want = [i for i in range(n_bank) if bank["split"][i] == args.split]
    if args.limit:
        want = want[:args.limit]
        fname = S.F_LABELS.format(split=f"{args.split}_sanity")
        print(f"[SANITY] limit={args.limit}, output {fname}")
    else:
        fname = S.F_LABELS.format(split=args.split)
    print(f"[split {args.split}] {len(want)} images")

    path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(path):
        rec = json.load(open(path))
        print(f"[resume] {fname} loaded")
    else:
        rec = dict(split=args.split, g_grid=list(S.G_GRID),
                   n_sample=S.N_SAMPLE_FIT, seed=S.SAGE_SEED,
                   s0_hits={}, s0_preds={}, edges={str(g): {} for g in S.G_GRID},
                   q={}, errors=[])

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    torch.set_grad_enabled(False)
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector="block8", tag=f"SAGELBL-{args.split}"))
    # installed once; `final` is re-pointed per generation (M7's pattern).
    # final=None is the byte-identical B2 identity -- this is what makes y(S0)
    # the incumbent's own answer rather than a leftover edge's.
    pruner = install(eng, model, None)

    dataset_cache = {}
    t0 = time.time()
    n_gen = 0
    for rank, i in enumerate(want):
        key, ds, idx = bank["key"][i], bank["ds"][i], int(bank["idx"][i])
        if complete(rec, key, S.G_GRID):
            continue
        if ds not in dataset_cache:
            dataset_cache[ds] = common.build_dataset(ds)
            model.set_dump_image(dataset_cache[ds].dump_image)
        dataset = dataset_cache[ds]
        row = dataset.data.iloc[idx]
        msg = common.build_message(model, dataset, ds, row)

        # q̄ from the engine's own instruction-embedding path -- the exact call
        # the deployed pruner reads (train/serve consistency by construction)
        if key not in rec["q"]:
            prep = eng.prepare(msg, ds)
            text_llm, _ = eng._instruction_embeds(prep)
            rec["q"][key] = text_llm[0].float().cpu().tolist()
            del prep

        s0 = np.sort(np.asarray(bank["s0"][i], dtype=np.int64))
        assert s0.size == BUDGET
        imp = bank["X"][i][:, bank["fi"]["imp"]]
        cos = bank["X"][i][:, bank["fi"]["cos_s0c"]]

        try:
            if key not in rec["s0_hits"]:
                pruner.final = None
                out = eng.run(msg, ds, MAX_NEW)
                rec["s0_hits"][key] = one_hit(ds, row, out["prediction"])
                rec["s0_preds"][key] = out["prediction"]
                n_gen += 1
                del out

            for g in S.G_GRID:
                store = rec["edges"][str(g)].setdefault(key, [])
                done = {tuple(sorted(e["minus"] + e["plus"])) for e in store}
                edges = S.build_edge_support(imp, cos, s0, g=g)
                sampled = S.sample_edges(edges, S.N_SAMPLE_FIT, S.SAGE_SEED, rank)
                if rank == 0:
                    S.validate_edges(edges, s0, g)      # G-SET on the first image
                for e in sampled:
                    if tuple(sorted(e["minus"] + e["plus"])) in done:
                        continue
                    Se = sorted((set(s0.tolist()) - set(e["minus"])) | set(e["plus"]))
                    assert len(Se) == BUDGET
                    pruner.final = Se
                    out = eng.run(msg, ds, MAX_NEW)
                    h = one_hit(ds, row, out["prediction"])
                    store.append(dict(minus=e["minus"], plus=e["plus"],
                                      hit=h, pred=out["prediction"]))
                    n_gen += 1
                    del out
        except Exception:
            rec["errors"].append(dict(key=key, at=time.time() - t0,
                                      err=traceback.format_exc()[-2000:]))
            print(f"  [ERROR] {key} ({len(rec['errors'])} so far)", flush=True)
        finally:
            if len(rec["errors"]) > 30:
                raise RuntimeError("too many labeling errors; aborting")
            S.dump_json(fname, rec)
        if (rank + 1) % 10 == 0:
            print(f"  {rank+1}/{len(want)}  gens={n_gen}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    # q̄ side file, bank-row aligned (not written in sanity mode)
    if args.limit:
        print(f"[SANITY done] no qbank written")
        return
    keys_all = [bank["key"][i] for i in want]
    q = np.array([rec["q"][k] for k in keys_all], dtype=np.float32)
    np.savez(os.path.join(OUTPUT_DIR, S.F_QBANK.replace(".npz", f"_{args.split}.npz")),
             key=np.array(keys_all), q=q)
    print(f"[done] {args.split}: {len(rec['s0_hits'])} S0 hits, "
          f"{sum(len(v) for g in rec['edges'].values() for v in g.values())} edges, "
          f"{len(rec['errors'])} errors, {n_gen} new gens, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
