"""
M2 — correctness gates. All six must pass before any accuracy or timing number
is reported (pre-registration §4).

  G-A  online L4 extraction  ==  the published M1 feature cache
  G-B  online scorer         ==  the cached-scorer reading, identical Top-256
  G-C  K = 1024 identity     ==  the stock HF logits, to a fixed tolerance
  G-D  K = 256 invariants       (budget, uniqueness, text retention, shapes,
                                 cache length, position policy)
  G-E  generation sanity        (deterministic, non-empty, no NaN, no implicit
                                 full recomputation)
  G-F  harness identity against the published pre-LLM baselines -- computed by
       m2_accuracy.py, which has to run the arms anyway.

Usage
    python scripts/discovery/m2_correctness.py                # G-A .. G-E
    python scripts/discovery/m2_correctness.py --gates C D E
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                     # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                    # noqa: E402
from m1_common import load_m1_plan                                # noqa: E402
from m2_gdep import (LAYER, MODE_FULL, MODE_GDEP, MODE_PRELLM,    # noqa: E402
                     POLICY_PRESERVE, POLICY_RENUMBER,
                     GDEPConfig, GDEPEngine, N_VIS, dump_json)

N_GATE = 8              # instances per gate (G-D and G-E use more)
N_INV = 12
TOL_GA = 0.01           # 1 % of the cache's mean |h|, the m1_features G1 rule
TOL_GC = 1e-3           # absolute, on pre-softmax logits of order 10


def heldout_rows(limit=None):
    """(plan, keys, rows) for the frozen held-out 150, in benchmark order."""
    _, plan, keys, rows_of = load_m1_plan()
    rows = list(rows_of["test"])
    assert len(rows) == 150, f"{len(rows)} test rows"
    if limit:
        rows = rows[:limit]
    return plan, keys, rows


_DATASETS = {}


def dataset_of(ds):
    """Load each benchmark once; the TSVs are large and the gates revisit them."""
    if ds not in _DATASETS:
        _DATASETS[ds] = common.build_dataset(ds)
    return _DATASETS[ds]


def instance_message(model, key):
    ds, idx = key.rsplit("_", 1)
    dataset = dataset_of(ds)
    model.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[int(idx)]
    return ds, common.build_message(model, dataset, ds, row)


def build(model, mode, **kw):
    cfg = GDEPConfig(mode=mode, **kw)
    return GDEPEngine.from_checkpoint(cfg, model=model, max_new_tokens=32)


# ---------------------------------------------------------------------------
def gate_A(model, rows_keys, report):
    """Online L4 extraction vs the published M1 cache (s2c1_feats_L4.npy)."""
    plan, keys, rows = rows_keys
    H = np.load(os.path.join(OUTPUT_DIR, "s2c1_feats_L4.npy"), mmap_mode="r")
    eng = build(model, MODE_GDEP, budget=1024, selector="topk", n_arm=960,
                seed=2, tag="GA")
    out = []
    for r in rows[:N_GATE]:
        key = keys[r]
        ds, msg = instance_message(model, key)
        prep = eng.prepare(msg, ds)
        st, info = eng.prefill(prep)
        online = st["h4"].detach().float().cpu().numpy().astype(np.float32)
        ref = np.asarray(H[r], dtype=np.float32)
        d = np.abs(online - ref)
        out.append(dict(key=key, max_abs_diff=float(d.max()),
                        mean_abs_diff=float(d.mean()),
                        frac_exact=float((online == ref).mean()),
                        ref_abs_mean=float(np.abs(ref).mean()),
                        online_abs_mean=float(np.abs(online).mean())))
        print(f"[G-A] {key}: max|d|={d.max():.3e} mean|d|={d.mean():.3e} "
              f"exact={100.0*(online == ref).mean():.2f}%  "
              f"(|h| mean {np.abs(ref).mean():.4f})")
        del st, prep
        torch.cuda.empty_cache()
    scale = float(np.mean([r["ref_abs_mean"] for r in out]))
    worst = max(r["max_abs_diff"] for r in out)
    report["G-A"] = dict(rows=out, ref_abs_mean=scale, worst_max_abs_diff=worst,
                         threshold=TOL_GA * scale,
                         passed=bool(worst < TOL_GA * scale),
                         n_compared=len(out), layer=LAYER)
    print(f"[G-A] worst max|d| = {worst:.3e}  threshold {TOL_GA*scale:.3e}  "
          f"-> {'PASS' if report['G-A']['passed'] else 'FAIL'}")


def gate_B(model, rows_keys, report):
    """Online scorer vs the cached-h4 scorer; Top-256 must be identical."""
    plan, keys, rows = rows_keys
    H = np.load(os.path.join(OUTPUT_DIR, "s2c1_feats_L4.npy"), mmap_mode="r")
    eng = build(model, MODE_GDEP, budget=256, selector="topk", n_arm=960,
                seed=2, tag="GB")
    out, worst_frac = [], 0.0
    for r in rows[:N_GATE]:
        key = keys[r]
        ds, msg = instance_message(model, key)
        prep = eng.prepare(msg, ds)
        st, info = eng.prefill(prep)
        online = info["scores"]
        cached_h4 = torch.from_numpy(np.asarray(H[r], dtype=np.float32)).to(eng.dev)
        cached = eng.score_online(cached_h4).cpu().numpy()
        d = np.abs(online - cached)
        scale = float(np.abs(cached).mean()) + 1e-12
        a = set(np.argsort(-online, kind="stable")[:256].tolist())
        b = set(np.argsort(-cached, kind="stable")[:256].tolist())
        same = a == b
        diff = sorted(a ^ b)
        out.append(dict(key=key, max_abs_diff=float(d.max()),
                        mean_abs_diff=float(d.mean()), score_abs_mean=scale,
                        rel_max=float(d.max() / scale),
                        top256_identical=bool(same),
                        top256_symmetric_diff=int(len(diff)),
                        differing_tokens=[dict(idx=int(i),
                                               online=float(online[i]),
                                               cached=float(cached[i]))
                                          for i in diff[:20]]))
        print(f"[G-B] {key}: max|ds|={d.max():.3e} ({100*d.max()/scale:.3f}% of "
              f"scale) top256 identical={same} symdiff={len(diff)}")
        del st, prep
        torch.cuda.empty_cache()
    worst_frac = max(r["rel_max"] for r in out)
    all_same = all(r["top256_identical"] for r in out)
    report["G-B"] = dict(rows=out, worst_rel_max=worst_frac,
                         all_top256_identical=bool(all_same),
                         n_compared=len(out),
                         passed=bool(all_same))
    print(f"[G-B] Top-256 identical on {sum(r['top256_identical'] for r in out)}"
          f"/{len(out)} instances -> "
          f"{'PASS' if all_same else 'FAIL'}")


def gate_C(model, rows_keys, report):
    """K = 1024 identity: the engine's manual layer loop vs the stock forward.

    Three arms per instance:
      full      the engine in `full` mode   -- one uninterrupted 36-layer pass
      K=1024    the engine in `gdep` mode with T = 1024, no-op compaction
      forced    the same, with the split-and-rejoin path forced on

    All three carry the pre-registered 1e-3 absolute tolerance on logits of
    order 40. `forced` is the sharpest of the three: it additionally exercises
    the tensor gather, the cache gather and the second layer-loop entry, so a
    tolerance it meets is a tolerance the compaction path meets. The ulp column
    is reported alongside so the deviation is legible at the dtype's resolution
    even when it is zero.
    """
    plan, keys, rows = rows_keys
    eng = build(model, MODE_GDEP, budget=1024, selector="topk", n_arm=960,
                seed=2, tag="GC")
    eng_sp = build(model, MODE_GDEP, budget=1024, selector="topk", n_arm=960,
                   seed=2, tag="GC-forced", force_split=True)
    eng_full = build(model, MODE_FULL, tag="GC-full")
    out = []
    for r in rows[:N_GATE]:
        key = keys[r]
        ds, msg = instance_message(model, key)
        prep = eng.prepare(msg, ds)
        with torch.no_grad():
            ref = model.model.model(inputs_embeds=prep["prompt"],
                                    attention_mask=prep["mask"],
                                    use_cache=False, return_dict=True)
            ref_logits = model.model.lm_head(ref.last_hidden_state[:, -1:, :])[:, -1, :]
        st_fu, _ = eng_full.prefill(prep)
        st_id, info_id = eng.prefill(prep)
        st_sp, _ = eng_sp.prefill(prep)
        d_fu = float((st_fu["logits"] - ref_logits).abs().max())
        d_id = float((st_id["logits"] - ref_logits).abs().max())
        d_sp = float((st_sp["logits"] - ref_logits).abs().max())
        scale = float(ref_logits.abs().max())
        ulp = float(torch.finfo(torch.bfloat16).eps) * scale      # bf16 ulp here
        out.append(dict(key=key, max_abs_diff_full_mode=d_fu,
                        max_abs_diff_K1024=d_id,
                        max_abs_diff_forced_split=d_sp,
                        logit_abs_max=scale, bf16_ulp=ulp,
                        forced_split_ulps=d_sp / ulp,
                        top1_match=bool(int(st_id["logits"].argmax())
                                        == int(ref_logits.argmax())),
                        top1_match_forced=bool(int(st_sp["logits"].argmax())
                                               == int(ref_logits.argmax())),
                        n_kept=info_id["n_kept"],
                        identity_fast_path=info_id.get("identity_fast_path"),
                        identity_permutation=bool(
                            info_id["select_idx"] == list(range(N_VIS)))))
        print(f"[G-C] {key}: full={d_fu:.3e}  K1024={d_id:.3e}  "
              f"forced-split={d_sp:.3e} ({d_sp/ulp:.2f} bf16 ulp, logit scale "
              f"{scale:.2f})  identity={out[-1]['identity_permutation']}")
        del st_id, st_sp, st_fu, prep
        torch.cuda.empty_cache()
    worst_full = max(r["max_abs_diff_full_mode"] for r in out)
    worst_k = max(r["max_abs_diff_K1024"] for r in out)
    worst_sp = max(r["max_abs_diff_forced_split"] for r in out)
    report["G-C"] = dict(
        rows=out, tolerance=TOL_GC,
        worst_full_mode=worst_full, worst_K1024=worst_k,
        worst_forced_split=worst_sp,
        worst_forced_split_ulps=max(r["forced_split_ulps"] for r in out),
        all_identity=bool(all(r["identity_permutation"] for r in out)),
        all_fast_path=bool(all(r["identity_fast_path"] for r in out)),
        all_top1=bool(all(r["top1_match"] for r in out)),
        all_top1_forced=bool(all(r["top1_match_forced"] for r in out)),
        passed=bool(worst_full < TOL_GC and worst_k < TOL_GC
                    and all(r["identity_permutation"] for r in out)
                    and all(r["top1_match"] for r in out)))
    print(f"[G-C] worst full-mode {worst_full:.3e}, worst K=1024 {worst_k:.3e} "
          f"(tol {TOL_GC:.0e}); forced-split {worst_sp:.3e} "
          f"({report['G-C']['worst_forced_split_ulps']:.2f} bf16 ulp) -> "
          f"{'PASS' if report['G-C']['passed'] else 'FAIL'}")


def _invariants(info, state, policy, key, violations):
    n_ctx = state["hidden_len"]
    n_text = info["n_text"]
    if info["n_kept"] != 256:
        violations.append(f"{key}: n_kept={info['n_kept']}")
    idx = np.array(info["select_idx"])
    if len(set(idx.tolist())) != len(idx):
        violations.append(f"{key}: duplicate selected indices")
    if idx.min() < 0 or idx.max() >= N_VIS:
        violations.append(f"{key}: selected index out of range")
    if info["context_len"] != 256 + n_text:
        violations.append(f"{key}: context_len={info['context_len']} "
                          f"!= 256+{n_text}")
    if info["kv_seq_len"] != info["context_len"]:
        violations.append(f"{key}: kv_seq_len={info['kv_seq_len']} "
                          f"!= context_len={info['context_len']}")
    keep = np.array(info["keep_full"])
    s, e = info["vis_start"], info["vis_end"]
    if not np.array_equal(np.sort(keep), keep):
        violations.append(f"{key}: keep_full not sorted")
    if len(set(keep.tolist())) != len(keep):
        violations.append(f"{key}: duplicate positions in keep_full")
    if not np.array_equal(keep[keep < s], np.arange(0, s)):
        violations.append(f"{key}: leading text tokens not all kept")
    if not np.array_equal(keep[keep >= e], np.arange(e, info["seq_full"])):
        violations.append(f"{key}: trailing text tokens not all kept")
    kept_vis = keep[(keep >= s) & (keep < e)] - s
    if not np.array_equal(np.sort(kept_vis), idx):
        violations.append(f"{key}: keep_full visual part != select_idx")
    p1 = state["pos1"].cpu().numpy()
    if len(p1) != info["context_len"]:
        violations.append(f"{key}: position_ids length {len(p1)} != "
                          f"{info['context_len']}")
    if policy == POLICY_PRESERVE:
        if np.any(np.diff(p1) <= 0):
            violations.append(f"{key}: position_ids not strictly increasing")
        if not np.isin(p1, np.arange(info["seq_full"])).all():
            violations.append(f"{key}: position_ids not a subsequence of the "
                              f"full range")
        if np.array_equal(p1, np.arange(len(p1))):
            violations.append(f"{key}: preserve policy produced contiguous ids")
    else:
        if not np.array_equal(p1, np.arange(len(p1))):
            violations.append(f"{key}: renumber policy produced non-contiguous ids")


def gate_D(model, rows_keys, report):
    """K = 256 invariants, both position policies."""
    plan, keys, rows = rows_keys
    out = {}
    for policy in (POLICY_PRESERVE, POLICY_RENUMBER):
        eng = build(model, MODE_GDEP, budget=256, selector="topk", n_arm=960,
                    seed=2, pos_policy=policy, tag=f"GD-{policy}")
        viol, recs = [], []
        for r in rows[:N_INV]:
            key = keys[r]
            ds, msg = instance_message(model, key)
            prep = eng.prepare(msg, ds)
            st, info = eng.prefill(prep)
            _invariants(info, st, policy, key, viol)
            recs.append(dict(key=key, n_kept=info["n_kept"],
                             context_len=info["context_len"],
                             n_text=info["n_text"],
                             kv_seq_len=info["kv_seq_len"],
                             pos_ids_contiguous=info["pos_ids_contiguous"],
                             kv_bytes_early=info["kv_bytes_early"],
                             kv_bytes_late=info["kv_bytes_late"]))
            del st, prep
            torch.cuda.empty_cache()
        out[policy] = dict(instances=recs, violations=viol,
                           passed=bool(not viol))
        print(f"[G-D] {policy}: {len(recs)} instances, "
              f"{len(viol)} violations -> {'PASS' if not viol else 'FAIL'}")
        for v in viol[:10]:
            print("      ", v)
    report["G-D"] = dict(by_policy=out,
                         passed=bool(all(v["passed"] for v in out.values())))
    print(f"[G-D] -> {'PASS' if report['G-D']['passed'] else 'FAIL'}")


def gate_E(model, rows_keys, report):
    """Deterministic generation, no NaN, no implicit full recomputation."""
    plan, keys, rows = rows_keys
    eng = build(model, MODE_GDEP, budget=256, selector="topk", n_arm=960,
                seed=2, tag="GE")
    n_layers = eng.n_layers
    recs = []
    for r in rows[:N_INV]:
        key = keys[r]
        ds, msg = instance_message(model, key)
        try:
            out = eng.run(msg, ds, max_new_tokens=64)
        except Exception:
            traceback.print_exc()
            recs.append(dict(key=key, ok=False, error="exception"))
            continue
        # The engine counts its own decode steps, so the check is exact rather
        # than inferred from the emitted-token count: decode issues one
        # `_run` per step, and the loop breaks on EOS *after* appending, so a
        # generation that ends on EOS costs `n_decode - 1` steps, not `n_decode`.
        steps = eng.counts.get("decode_steps", -1)
        expected = n_layers * (1 + steps)
        got = eng.counts.get("layer_calls", -1)
        recs.append(dict(key=key, ok=True, n_decode=out["n_decode"],
                         decode_steps=steps,
                         prediction=out["prediction"][:200],
                         nonempty=bool(len(out["prediction"].strip()) > 0),
                         layer_calls=got, layer_calls_expected=expected,
                         no_recompute=bool(got == expected),
                         kv_seq_len=out["info"]["kv_seq_len"]))
        print(f"[G-E] {key}: {out['n_decode']} tokens/{steps} decode steps, "
              f"layer_calls {got}/{expected}, kv_len "
              f"{out['info']['kv_seq_len']}, pred={out['prediction'][:60]!r}")
        torch.cuda.empty_cache()
    ok = [r for r in recs if r["ok"]]
    passed = (len(ok) >= 10 and all(r["nonempty"] for r in ok)
              and all(r["no_recompute"] for r in ok)
              and all(r["n_decode"] >= 1 for r in ok))
    report["G-E"] = dict(records=recs, n_ok=len(ok),
                         n_expected_layer_calls_rule=f"{n_layers}*(1+decode_steps)",
                         passed=bool(passed))
    print(f"[G-E] {len(ok)}/{len(recs)} completed -> "
          f"{'PASS' if passed else 'FAIL'}")


def stage(model, gates=("A", "B", "C", "D", "E"), tag="m2_correctness"):
    """Run the gates against an already-loaded model; returns the report dict."""
    torch.set_grad_enabled(False)
    rows_keys = heldout_rows()
    args = argparse.Namespace(gates=list(gates), tag=tag)

    report = dict(stage="M2", gates=args.gates,
                  environment=dict(torch=torch.__version__,
                                   cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(0),
                                   layer=LAYER, n_vis=N_VIS),
                  tol_ga_relative=TOL_GA, tol_gc_absolute=TOL_GC)
    fn = dict(A=gate_A, B=gate_B, C=gate_C, D=gate_D, E=gate_E)
    for g in args.gates:
        try:
            fn[g](model, rows_keys, report)
        except Exception:
            traceback.print_exc()
            report[f"G-{g}"] = dict(passed=False, error=traceback.format_exc()[-2000:])
    report["all_passed"] = bool(all(report.get(f"G-{g}", {}).get("passed", False)
                                    for g in args.gates))
    dump_json(f"{args.tag}.json", report)
    print(f"\n[GATES] {args.gates} -> all_passed={report['all_passed']}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--tag", default="m2_correctness")
    args = ap.parse_args()
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=32)
    model.model.eval()
    stage(model, args.gates, args.tag)


if __name__ == "__main__":
    main()
