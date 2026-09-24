"""
M3-v2 step 3 -- the HEAD GATE: does the auditor find the teacher's front end?

This is the brief's §7 early gate, and it runs BEFORE any generation.  Every
scorer below is reduced to the same object -- a (768,) score vector over the
tokens B2 dropped -- and scored by the same head metrics, so the comparison is
between rankings and nothing else:

    teacher     the oracle: the teacher's own ranking of the dropped set
    v0          the frozen v0 student (`m3_miss.pt`), rebuilt from the bank
    v0-v1s{0,1,2}  the three v1 seeds (22 features), same rebuild
    imp         the EADP importance scalar alone -- what the base selector
                already knew, and the cheapest possible content baseline
    random      content-free, 20 draws, to show the sampling spread
    v2 <cfg>    each trained HeadAuditor checkpoint

Gate (brief §7), evaluated on `val` -- the split the checkpoints were selected
on, and the only one that may be read before the decision:

    PASS  if  Top-16 recall@16 >= 0.35
          or  mean teacher rank of the 16 rescued tokens improves >= 30 % on v0
    and   at least 2 of the 3 seeds agree in direction.

`test` (the held-out 150) is reported as `test_posthoc` and decides nothing.

Usage
    python scripts/discovery/m3v2_proxy.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import OUTPUT_DIR                                       # noqa: E402
from m3_common import (FEATURES, MissStudent, feature_index)        # noqa: E402
from m3v2_common import (BankV2, aggregate, build_auditor,          # noqa: E402
                         fit_stats, instance_head_metrics, standardise)

GATE_RECALL = 0.35          # brief §7
GATE_RANK_GAIN = 0.30       # 30 % better mean teacher rank than v0
STOP_RECALL = 0.30          # below this the auditor is stopped, not tuned


# ---------------------------------------------------------------------------
# scorers: each returns a (768,) float vector over the dropped set of row i
# ---------------------------------------------------------------------------
def _v0_scorer(bank, ck, device):
    """Rebuild `MissGuardPruner.audit_scores` from the bank.

    Byte-for-byte the live path: the hand features in the bank ARE
    `handcrafted_t`'s output, and `bank.vis` is the fp16 tensor the live pruner
    quantises to before standardising.
    """
    idx = feature_index(ck["features"])
    mu_h = torch.from_numpy(np.asarray(ck["mu_hand"], np.float32)).to(device)
    sd_h = torch.from_numpy(np.asarray(ck["sd_hand"], np.float32)).to(device)
    mu_v = torch.from_numpy(np.asarray(ck["mu_vis"], np.float32)).to(device)
    sd_v = torch.from_numpy(np.asarray(ck["sd_vis"], np.float32)).to(device)
    per_z = bool(ck["per_instance_z"])
    model = ck["student"].to(device).eval()
    return idx, mu_h, sd_h, mu_v, sd_v, per_z, model


def v0_scores(bank, i, st_, device):
    idx, mu_h, sd_h, mu_v, sd_v, per_z, model = st_
    d = bank.drop[i]
    Xh = torch.from_numpy(bank.X[i][d]).to(device)[:, idx]
    V = torch.from_numpy(bank.vis[i][d].astype(np.float32)).to(device)
    with torch.no_grad():
        zh = (Xh - mu_h) / sd_h
        zv = (V - mu_v) / sd_v
        if per_z:
            zh = (zh - zh.mean(0, keepdim=True)) / zh.std(0, keepdim=True).clamp_min(1e-6)
        return model(zh, zv).float().cpu().numpy()


def v2_scorer(bank, ck, device):
    model = build_auditor(ck, device).eval()
    st = {k: torch.from_numpy(np.asarray(ck[k], np.float32)).to(device)
          for k in ("mu_vis", "sd_vis", "mu_txt", "sd_txt", "mu_hand", "sd_hand")}

    def run(i):
        d = bank.drop[i]
        vis = torch.from_numpy(bank.vis[i:i + 1]).to(device)
        txt = torch.from_numpy(bank.txt[i:i + 1]).to(device)
        X = torch.from_numpy(bank.X[i:i + 1]).to(device)
        s0 = torch.from_numpy(bank.s0[i:i + 1]).to(device)
        mask = torch.arange(txt.shape[1], device=device)[None, :] \
            < int(bank.txt_len[i])
        with torch.no_grad():
            v = standardise(vis, st["mu_vis"], st["sd_vis"])
            t = standardise(txt, st["mu_txt"], st["sd_txt"])
            x = standardise(X, st["mu_hand"], st["sd_hand"])
            sc = model(v, t, x, s0, mask)
        return sc[0][torch.from_numpy(d).to(device)].float().cpu().numpy()
    return run


def teacher_scores(bank, i):
    return bank.g2[i][bank.drop[i]]


def imp_scores(bank, i):
    return bank.X[i][bank.drop[i]][:, FEATURES.index("imp")]


def random_scores(bank, i, rng):
    return rng.standard_normal(bank.drop[i].size)


# ---------------------------------------------------------------------------
def evaluate(bank, rows, score_of):
    per = [instance_head_metrics(bank.g2[i], bank.drop[i], score_of(i))
           for i in rows]
    return aggregate(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", default="m3v2_auditor_*.pt")
    ap.add_argument("--out", default="m3v2_proxy")
    ap.add_argument("--v0", default="m3_miss.pt")
    ap.add_argument("--v0-extra", nargs="*", default=["m3_miss_v1_s0.pt",
                                                      "m3_miss_v1_s1.pt",
                                                      "m3_miss_v1_s2.pt"])
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--xcheck-arm", default="MG-maxred-r16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    bank = BankV2()
    stats = fit_stats(bank, bank.rows("fit"))
    val_rows, test_rows = bank.rows("val"), bank.rows("test")
    fit_rows = bank.rows("fit")
    print(f"[bank] fit {len(bank.rows('fit'))}  val {len(val_rows)}  "
          f"test {len(test_rows)}  ({time.time()-t0:.0f}s)", flush=True)

    scorers = {}
    # -- content-free and cheap references ---------------------------------
    scorers["random"] = (None, lambda i: np.mean(
        [random_scores(bank, i, np.random.default_rng(1000 + k))
         for k in range(args.n_random)], axis=0))
    scorers["imp"] = (None, lambda i: imp_scores(bank, i))
    scorers["teacher"] = (None, lambda i: teacher_scores(bank, i))

    # -- a RANDOM-INIT v0 student -------------------------------------------
    # Same architecture, same features, same standardisers as `m3_miss`, with
    # untrained weights.  This is the control that says how much of v0's 0.2365
    # is the training and how much is the architecture plus the feature bank.
    # v2 gets the identical control for free: `init_val` in `m3v2_train.json`.
    _p = os.path.join(OUTPUT_DIR, args.v0)
    if os.path.exists(_p):
        _ck = torch.load(_p, map_location="cpu", weights_only=False)
        torch.manual_seed(0)
        _m = MissStudent(_ck["d_hand"], _ck["d_vis"])
        _ck2 = dict(_ck)
        _ck2["student"] = _m
        scorers["v0_randinit"] = (_v0_scorer(bank, _ck2, device), None)

    # -- v0 and the v1 seeds ------------------------------------------------
    v0_specs = {}
    for tag in [args.v0] + list(args.v0_extra):
        p = os.path.join(OUTPUT_DIR, tag)
        if not os.path.exists(p):
            print(f"[skip] {tag} not found")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = MissStudent(ck["d_hand"], ck["d_vis"])
        m.load_state_dict(ck["state_dict"])
        ck["student"] = m
        spec = _v0_scorer(bank, ck, device)
        v0_specs[tag.replace(".pt", "")] = spec
        scorers[tag.replace(".pt", "")] = (spec, None)

    # -- v2 checkpoints -----------------------------------------------------
    v2 = {}
    for p in sorted(glob.glob(os.path.join(OUTPUT_DIR, args.ckpt_glob))):
        # `m3v2_auditor[_<grid>]_<cfg>_s<seed>.pt` -- parse from the right so a
        # grid prefix (e.g. `_reg`) needs no special case.
        parts = os.path.basename(p)[:-3].split("_")
        if len(parts) < 4 or parts[0] != "m3v2" or parts[1] != "auditor":
            continue
        if not parts[-1].startswith("s") or not parts[-1][1:].isdigit():
            continue
        grid = "_".join(parts[2:-2]) or "primary"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        v2[f"{ck['objective']}:{ck['ablation']}|s{ck['seed']}"] = (
            ck, v2_scorer(bank, ck, device), grid)
        assert f"{ck['objective']}{ck['ablation']}" == parts[-2], (p, parts)
    print(f"[ckpt] {len(v2)} v2 checkpoints: {sorted(v2)}", flush=True)

    def run_one(name, sc):
        st_, fn = sc
        if st_ is not None:
            f = lambda i: v0_scores(bank, i, st_, device)      # noqa: E731
        else:
            f = fn
        out = {}
        # `fit` is the third leg of the diagnostic: how much of the teacher's
        # head a scorer recovers on the rows it was FITTED on.  A scorer whose
        # fit and val head recalls are both low is not overfitting -- it is
        # unable to represent the head at all, and no amount of regularisation
        # will change that.
        for split, rows in (("val", val_rows), ("test", test_rows),
                            ("fit", fit_rows)):
            agg, per = evaluate(bank, rows, f)
            out[split] = agg
            out[split + "_per_instance"] = per
        return out

    report = dict(gate=dict(recall=GATE_RECALL, rank_gain=GATE_RANK_GAIN,
                            stop_recall=STOP_RECALL, split="val"),
                  n_random=args.n_random, scorers={}, v2={})
    # Before any metric is read: does the rebuilt v0 scorer reproduce the ids
    # the live v0 grid actually rescued?
    report["xcheck_v0"] = crosscheck_v0(bank, v0_specs[args.v0.replace(".pt", "")],
                                        device, arm=args.xcheck_arm)
    print(f"[xcheck] {report['xcheck_v0']}", flush=True)
    for name, sc in scorers.items():
        r = run_one(name, sc)
        report["scorers"][name] = r
        m = r["val"]
        print(f"  {name:16s} val  Top16rec@16 {m['top16_recall@16']:.4f}"
              f"  meanrank@16 {m['mean_rank@16']:7.2f}  hw@16 {m['hw@16']:.4f}"
              f"  ndcg@16 {m['ndcg@16']:.4f}   | test {r['test']['top16_recall@16']:.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(report, open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w"),
                  indent=1)

    for name, (ck, fn, grid) in v2.items():
        r = run_one(name, (None, fn))
        r["meta"] = dict(objective=ck["objective"], ablation=ck["ablation"],
                         seed=ck["seed"], best_epoch=ck["best_epoch"], grid=grid,
                         init_val=ck.get("init_val_metrics"),
                         fit=ck.get("fit_metrics"),
                         n_params=int(sum(v.numel() for v in ck["state_dict"].values()
                                          if v.dim() > 0)),
                         val_selected=ck["selection"])
        report["v2"][name] = r
        m = r["val"]
        print(f"  {name:16s} val  Top16rec@16 {m['top16_recall@16']:.4f}"
              f"  meanrank@16 {m['mean_rank@16']:7.2f}  hw@16 {m['hw@16']:.4f}"
              f"  ndcg@16 {m['ndcg@16']:.4f}   | test {r['test']['top16_recall@16']:.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(report, open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w"),
                  indent=1)

    _gate(report)
    json.dump(report, open(os.path.join(OUTPUT_DIR, f"{args.out}.json"), "w"), indent=1)
    print(f"[saved] {args.out}.json  ({time.time()-t0:.0f}s)")


def crosscheck_v0(bank, spec, device, json_name="m3_accuracy.json",
                  arm="MG-maxred-r16", r=16):
    """Prove the bank-rebuilt scorer IS the scorer that produced the stored grid.

    The v0 grid recorded, for every held-out instance, the exact 16 token ids
    the live pruner rescued.  Recomputing those ids from the bank alone -- no
    model, no generation -- closes the loop on three things at once: that the
    bank's X/vis/s0 are what the live run saw, that `feature_index` resolves a
    18-feature checkpoint inside today's 22-column FEATURES correctly, and that
    the rebuilt arithmetic is the live arithmetic.  Without this the whole
    proxy table could be measuring a different student than the one M3-v0
    reported.
    """
    path = os.path.join(OUTPUT_DIR, json_name)
    if not os.path.exists(path):
        return dict(checked=False, reason=f"{json_name} missing")
    arms = json.load(open(path))["arms"]
    if arm not in arms:
        return dict(checked=False, reason=f"arm {arm} missing")
    idx_of = {k: i for i, k in enumerate(bank.key)}
    same, ties, bad, missing = 0, 0, [], 0
    for meta in arms[arm]["per_image_meta"]:
        i = idx_of.get(meta["key"])
        if i is None or meta.get("rescue_idx") is None:
            missing += 1
            continue
        sc = v0_scores(bank, i, spec, device)
        d = bank.drop[i]
        got = np.sort(d[np.argsort(-sc, kind="stable")[:r]])
        want = np.sort(np.asarray(meta["rescue_idx"]))
        if np.array_equal(got, want):
            same += 1
        elif np.array_equal(np.sort(bank.g2[i][got]), np.sort(bank.g2[i][want])):
            # the stored and rebuilt rescue are both teacher-optimal but sit on
            # a bit-identical teacher-score tie at the r-th boundary
            ties += 1
        else:
            bad.append(meta["key"])
    return dict(checked=True, arm=arm, n=len(arms[arm]["per_image_meta"]),
                exact=same, tie_broken=ties, mismatch=len(bad),
                missing=missing, examples=bad[:5],
                passed=bool(not bad and same > 0))


def _gate(report):
    v0 = report["scorers"].get("m3_miss", {}).get("val")
    assert v0 is not None, "the v0 baseline is missing; the gate has no reference"
    ref_recall, ref_rank = v0["top16_recall@16"], v0["mean_rank@16"]
    rows = []
    for name, r in sorted(report["v2"].items()):
        m = r["val"]
        rec, rk = m["top16_recall@16"], m["mean_rank@16"]
        gain = (ref_rank - rk) / ref_rank if ref_rank > 0 else 0.0
        rows.append(dict(name=name, recall=rec, rank=rk, rank_gain=gain,
                         by_recall=bool(rec >= GATE_RECALL),
                         by_rank=bool(gain >= GATE_RANK_GAIN),
                         passed=bool(rec >= GATE_RECALL or gain >= GATE_RANK_GAIN),
                         below_stop=bool(rec < STOP_RECALL and gain < GATE_RANK_GAIN)))
    # seeds agree in direction: >= 2 of the 3 seeds of a config pass
    by_cfg = {}
    for x in rows:
        by_cfg.setdefault(x["name"].split("|")[0], []).append(x)
    verdicts = {}
    for cfg, xs in sorted(by_cfg.items()):
        n_pass = sum(int(x["passed"]) for x in xs)
        verdicts[cfg] = dict(n_seeds=len(xs), n_pass=n_pass,
                             seed_recall=[round(x["recall"], 4) for x in xs],
                             seed_mean_recall=float(np.mean([x["recall"] for x in xs])),
                             seed_mean_rank=float(np.mean([x["rank"] for x in xs])),
                             seed_mean_hw=float(np.mean(
                                 [report["v2"][x["name"]]["val"]["hw@16"] for x in xs])),
                             gate_pass=bool(n_pass >= 2))
    report["gate"]["v0_reference"] = dict(top16_recall_at_16=ref_recall,
                                          mean_rank_at_16=ref_rank,
                                          hw_at_16=v0["hw@16"],
                                          ndcg_at_16=v0["ndcg@16"])
    report["gate"]["per_run"] = rows
    report["gate"]["per_config"] = verdicts
    report["gate"]["any_pass"] = bool(any(v["gate_pass"] for v in verdicts.values()))
    print("\n[gate] v0 reference: Top16rec@16 "
          f"{ref_recall:.4f}   mean rank@16 {ref_rank:.2f}")
    for cfg, v in verdicts.items():
        print(f"[gate] {cfg:6s} seeds pass {v['n_pass']}/{v['n_seeds']}  "
              f"mean rec {v['seed_mean_recall']:.4f}  mean rank "
              f"{v['seed_mean_rank']:7.2f}  mean hw {v['seed_mean_hw']:.4f}  "
              f"-> {'PASS' if v['gate_pass'] else 'FAIL'}")
    print(f"[gate] HEAD-GATE {'PASS' if report['gate']['any_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
