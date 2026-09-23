"""
M2 — implementation-defect amendment: post-fix correctness gates (HG-1..HG-9).

The interim audit (6a74c0b) localised a frozen-RoPE-position defect in the
PRESERVE branch of `GDEPEngine.decode`. The engine now carries decode version
`gdep_preserve_v2_advancing_rope`. These gates verify the fix against the nine
properties the amendment pre-registers, BEFORE any corrected accuracy or
timing number is produced:

  HG-1  K=1024 MULTI-STEP generation is token-for-token the stock HF greedy
        generation, on both the `full` path and the gdep K=1024 identity path,
        with >= 16 steps force-decoded (past EOS on both sides, so the check
        cannot be satisfied by an early stop). G-C/D1 compared logits and one
        short generation; a frozen position can hide under both.
  HG-2  corrected PRESERVE: decode RoPE positions start at
        max(prefill_position_ids)+1 and advance +1 per step, no repeats.
  HG-3  RENUMBER: decode positions advance contiguously (unchanged behaviour,
        asserted so the fix demonstrably did not disturb the control).
  HG-4  no frozen position survives: 64 force-decoded steps on the three D5
        instances -- the position set size must equal the step count.
  HG-5  the three D5 looping instances answer like the audit's P' column under
        the corrected engine (token-id equality with the recorded ids).
  HG-6  the previously capped instances (>= 20, taken from the invalid PRESERVE
        records in m2_accuracy.json) re-decoded with the corrected engine:
        termination without hitting the cap.
  HG-7  EOS rate on that set (the amendment's "EOS rate returns to normal").
  HG-8  the decode fix changes no prefill quantity: online scores and the
        selected token set are BIT-identical between the v1 engine (loaded from
        git blob 6a74c0b, the version that ran the grid) and the v2 engine; and
        RENUMBER decode ids are identical v1 vs v2 (force-decoded 24 steps).
        This is what licenses keeping the existing C1-R records.
  HG-9  no extra forward/recompute on the fixed path: layer_calls equals
        36*(1+decode_steps) exactly, and prefill still runs 36 layer calls.

Writes m2_fixgates.json. Exit code 0 only if every gate passes.

Usage
    python scripts/discovery/m2_fixgates.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                     # noqa: E402
from common import OUTPUT_DIR, eadp_model_name                    # noqa: E402
from m1_common import load_m1_plan                                # noqa: E402
import m2_gdep                                                    # noqa: E402
from m2_gdep import (MODE_FULL, MODE_GDEP, POLICY_PRESERVE,        # noqa: E402
                     POLICY_RENUMBER, GDEPConfig, GDEPEngine,
                     ENGINE_VERSION, N_VIS, dump_json)
from m2_interim_diag import repetition                            # noqa: E402

N_ID = 8            # instances for HG-1 / HG-8
N_POS = 6           # instances for HG-2/3
FORCED_STEPS = 24   # HG-1 / HG-8 renumber comparison
D5_KEYS = ["TextVQA_VAL_4127", "DocVQA_VAL_1077", "DocVQA_VAL_1292"]
V1_COMMIT = "6a74c0b"   # the exact code the contaminated grid ran


def heldout_keys():
    _, plan, keys, rows_of = load_m1_plan()
    return keys, list(rows_of["test"])


_DATASETS = {}


def message_for(model, key):
    ds, idx = key.rsplit("_", 1)
    if ds not in _DATASETS:
        _DATASETS[ds] = common.build_dataset(ds)     # load each benchmark once
    dataset = _DATASETS[ds]
    model.set_dump_image(dataset.dump_image)
    row = dataset.data.iloc[int(idx)]
    return ds, common.build_message(model, dataset, ds, row)


def build(model, **kw):
    return GDEPEngine.from_checkpoint(GDEPConfig(**kw), model=model)


# ---------------------------------------------------------------------------
def hf_greedy_reference(model, prompt, mask, n_steps):
    """Stock HF greedy continuation of exactly `n_steps` tokens.

    Same prompt tensor the engine was handed (inputs_embeds, no image_grid_thw)
    and the same 1-D position convention HF derives from it. The forcing
    convention is `eos_token_id=-1`: a stop token that no vocabulary entry
    matches, so the sequence is genuinely forced to n_steps tokens and greedy
    picks are NOT reshaped. `min_new_tokens` must not be used here -- it
    installs a logits processor that MASKS the real EOS, so the reference and
    the engine's ignore_eos branch would disagree from the step where EOS is
    the greedy token (this was the first version's false FAIL: the engine
    emits 151645, the masked reference emits 7377).
    """
    gen_cfg = model.model.generation_config
    with torch.no_grad():
        out = model.model.generate(
            inputs_embeds=prompt, attention_mask=mask,
            max_new_tokens=n_steps, do_sample=False,
            pad_token_id=gen_cfg.pad_token_id, eos_token_id=-1,
            return_dict_in_generate=True)
    return out.sequences[0].tolist()


def gate_1(model, keys, rows, report):
    """HG-1: multi-step identity, `full` and gdep-K1024, vs stock HF."""
    eng_full = build(model, mode=MODE_FULL, tag="HG1-full")
    eng_id = build(model, mode=MODE_GDEP, budget=N_VIS, selector="topk",
                   n_arm=960, seed=2, pos_policy=POLICY_PRESERVE, tag="HG1-id")
    recs, ok = [], True
    for r in rows[:N_ID]:
        key = keys[r]
        ds, msg = message_for(model, key)
        prep = eng_full.prepare(msg, ds)
        ref = hf_greedy_reference(model, prep["prompt"], prep["mask"],
                                  FORCED_STEPS)
        stf, _ = eng_full.prefill(prep)
        ids_f = eng_full.decode(stf, FORCED_STEPS, ignore_eos=True)
        sti, _ = eng_id.prefill(prep)
        ids_i = eng_id.decode(sti, FORCED_STEPS, ignore_eos=True)
        eq_f = ids_f == ref
        eq_i = ids_i == ref
        ok &= eq_f and eq_i
        recs.append(dict(key=key, n_steps=FORCED_STEPS,
                         first_new=int(ref[0]),
                         full_equals_hf=bool(eq_f),
                         gdep_k1024_equals_hf=bool(eq_i),
                         first_diff_full=(next((i for i, (a, b) in enumerate(
                             zip(ids_f, ref)) if a != b), None)),
                         first_diff_id=(next((i for i, (a, b) in enumerate(
                             zip(ids_i, ref)) if a != b), None)),
                         layer_calls=eng_full.counts.get("layer_calls")))
        print(f"[HG-1] {key}: full=={eq_f} gdep1024=={eq_i} "
              f"(ref starts {ref[:4]})")
        del stf, sti, prep
        torch.cuda.empty_cache()
    report["HG-1_multi_step_identity"] = dict(
        rows=recs, n_compared=len(recs), forced_steps=FORCED_STEPS,
        passed=bool(ok and len(recs) == N_ID))
    print(f"[HG-1] {sum(r['full_equals_hf'] and r['gdep_k1024_equals_hf'] for r in recs)}"
          f"/{len(recs)} exact -> {'PASS' if ok else 'FAIL'}")


def gate_positions(model, keys, rows, report):
    """HG-2 + HG-3 + HG-4 + HG-9: decode position bookkeeping on both policies."""
    for name, policy in (("HG-2 preserve", POLICY_PRESERVE),
                         ("HG-3 renumber", POLICY_RENUMBER)):
        eng = build(model, mode=MODE_GDEP, budget=256, selector="topk",
                    n_arm=960, seed=2, pos_policy=policy, tag=name.split()[1])
        recs, ok = [], True
        for r in rows[:N_POS]:
            key = keys[r]
            ds, msg = message_for(model, key)
            prep = eng.prepare(msg, ds)
            st, info = eng.prefill(prep)
            prefill_max = int(st["pos1"].max().item())
            cache0 = int(st["cache"].get_seq_length(0))
            ids = eng.decode(st, FORCED_STEPS, ignore_eos=True)
            used = eng.last_decode_positions           # (cache_position, rope)
            cps = [c for c, _ in used]
            rps = [p for _, p in used]
            start_expect = prefill_max + 1 if policy == POLICY_PRESERVE else cache0
            adv_ok = (rps == list(range(start_expect, start_expect + len(rps))))
            cp_ok = (cps == list(range(cache0, cache0 + len(cps))))
            distinct = len(set(rps)) == len(rps)
            frozen_like_v1 = len(set(rps)) == 1
            # HG-9: every decode step is exactly one 36-layer pass; count is
            # checked against the recorded step count, not an assumed one.
            # (151645/151643 are this model's two stop tokens, read from its
            # generation config by the audit that found the defect.)
            steps = eng.counts.get("decode_steps", -1)
            no_recompute = eng.counts.get("layer_calls", -1) == 36 * (1 + steps)
            ok &= adv_ok and cp_ok and distinct and not frozen_like_v1 and no_recompute
            recs.append(dict(key=key, policy=policy, prefill_max_pos=prefill_max,
                             cache_len0=cache0,
                             rope_first=rps[0], rope_last=rps[-1],
                             start_expected=start_expect,
                             advances_consecutively=bool(adv_ok),
                             cache_position_advances=bool(cp_ok),
                             n_positions=len(rps), n_distinct=len(set(rps)),
                             decode_steps=steps,
                             layer_calls=eng.counts.get("layer_calls"),
                             layer_calls_expected=36 * (1 + steps),
                             no_recompute=bool(no_recompute)))
            print(f"[{name}] {key}: rope {rps[0]}..{rps[-1]} "
                  f"({len(set(rps))}/{len(rps)} distinct), layer_calls "
                  f"{eng.counts['layer_calls']}/{36 * (1 + steps)} -> "
                  f"{'ok' if recs[-1]['advances_consecutively'] and recs[-1]['no_recompute'] else 'BAD'}")
            del st, prep
            torch.cuda.empty_cache()
        report[name + "_decode_positions"] = dict(
            rows=recs, forced_steps=FORCED_STEPS, passed=bool(ok))
        print(f"[{name}] -> {'PASS' if ok else 'FAIL'}")


def gate_4_long(model, keys, rows, report):
    """HG-4: 64 force-decoded steps on the D5 trio -- no frozen position."""
    key_rows = {keys[r]: r for r in rows}
    eng = build(model, mode=MODE_GDEP, budget=256, selector="topk",
                n_arm=960, seed=0, pos_policy=POLICY_PRESERVE, tag="HG-4")
    recs, ok = [], True
    for key in D5_KEYS:
        ds, msg = message_for(model, key)
        prep = eng.prepare(msg, ds)
        st, info = eng.prefill(prep)
        eng.decode(st, 64, ignore_eos=True)
        rps = [p for _, p in eng.last_decode_positions]
        good = len(rps) == 64 and len(set(rps)) == 64 and \
            rps == list(range(rps[0], rps[0] + 64))
        ok &= good
        recs.append(dict(key=key, n=rps and len(rps), first=rps[0],
                         last=rps[-1], contiguous=bool(good)))
        print(f"[HG-4] {key}: 64 forced steps, positions {rps[0]}..{rps[-1]} "
              f"distinct={len(set(rps))} -> {'ok' if good else 'BAD'}")
        del st, prep
        torch.cuda.empty_cache()
    report["HG-4_no_frozen_position"] = dict(rows=recs, passed=bool(ok),
                                             steps=64)
    print(f"[HG-4] -> {'PASS' if ok else 'FAIL'}")


def gate_5(model, keys, rows, report):
    """HG-5: the D5 trio under corrected PRESERVE == D5's recorded P' ids."""
    d5 = json.load(open(os.path.join(OUTPUT_DIR, "m2_interim_d5.json")))
    expect = {it["key"]: it["P2"]["first_ids"] for it in d5["instances"]}
    eng = build(model, mode=MODE_GDEP, budget=256, selector="topk",
                n_arm=960, seed=0, pos_policy=POLICY_PRESERVE, tag="HG-5")
    key_rows = {keys[r]: r for r in rows}
    recs, ok = [], True
    for key in D5_KEYS:
        ds, msg = message_for(model, key)
        prep = eng.prepare(msg, ds)
        st, info = eng.prefill(prep)
        ids = eng.decode(st, 48)
        text = model.processor.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # D5 recorded `first_ids` = ids[:12] of its P2 run, not the whole list;
        # compare the prefix. (v1's decode_advanced also used max 48 steps, so
        # the prefix comparison is exact for the recorded window.)
        same = ids[:len(expect[key])] == expect[key]
        ok &= same
        recs.append(dict(key=key, ids=ids, expected_v1_P2=expect[key],
                         equals_P2=bool(same), text=text[:120],
                         chars=len(text), repetition=repetition(text),
                         terminated=bool(ids and ids[-1] in (151645, 151643))))
        print(f"[HG-5] {key}: {text[:40]!r} vs D5-P' "
              f"{model.processor.tokenizer.decode(expect[key], skip_special_tokens=True)!r} "
              f"-> {'==' if same else 'DIFFERS'}")
        del st, prep
        torch.cuda.empty_cache()
    report["HG-5_d5_trio_delooped"] = dict(rows=recs, passed=bool(ok))
    print(f"[HG-5] -> {'PASS' if ok else 'FAIL'}")


def gate_67(model, keys, rows, report):
    """HG-6 + HG-7: every previously capped PRESERVE instance, corrected.

    The capped instances are re-derived from the invalid records themselves
    (prediction length > 2000 chars, the same rule as consolidation's
    degeneracy section), so the set is exactly what the defect produced. The
    arm that capped them defines the scorer configuration (C0 -> n=240 with its
    seed's scorer; C1/C2 -> n=960)."""
    acc = json.load(open(os.path.join(OUTPUT_DIR, "m2_accuracy.json")))
    capped = {}
    for run_key, rec in acc["arms"].items():
        if not str(rec.get("validity", "")).startswith("INVALID"):
            continue
        for i, p in enumerate(rec.get("predictions", [])):
            if len(p) > 2000:
                capped.setdefault(acc["keys"][i], []).append(
                    (run_key, rec["arm"], rec["seed"]))
    print(f"[HG-6] {len(capped)} distinct previously-capped instances "
          f"across {sum(len(v) for v in capped.values())} runs")
    assert len(capped) >= 20, "amendment requires >= 20 instances"
    engines, recs = {}, []
    ok_term = 0
    for key, origins in sorted(capped.items()):
        arm, seed = origins[0][1], origins[0][2]
        n_arm = 240 if arm == "C0" else 960
        selector = "block8" if arm == "C2" else "topk"
        ekey = (arm, seed, n_arm, selector)
        if ekey not in engines:
            engines[ekey] = build(
                model, mode=MODE_GDEP, budget=256, selector=selector,
                n_arm=n_arm, seed=seed, pos_policy=POLICY_PRESERVE,
                tag=f"HG6-{arm}")
        eng = engines[ekey]
        ds, msg = message_for(model, key)
        prep = eng.prepare(msg, ds)
        st, info = eng.prefill(prep)
        ids = eng.decode(st, 512)
        text = model.processor.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rep = repetition(text)
        eos = bool(ids and ids[-1] in (151645, 151643))
        capped_now = len(ids) >= 512
        ok_term += int(eos and not capped_now)
        recs.append(dict(key=key, re_run=f"{arm}|s{seed}", n_tokens=len(ids),
                         terminated_by_eos=bool(eos), capped_now=bool(capped_now),
                         chars=len(text), repetition=rep,
                         text_head=text[:60].replace("\n", " ")))
        print(f"[HG-6] {key} (was {len(origins)} runs): {len(ids)} tok "
              f"{'EOS' if eos else 'CAPPED'}  {text[:40]!r}")
        del st, prep
        torch.cuda.empty_cache()
    n = len(recs)
    report["HG-6_previously_capped"] = dict(
        n_instances=n, rows=recs,
        n_terminated=ok_term, termination_rate=float(ok_term / n),
        passed=bool(ok_term >= 0.9 * n and n >= 20))
    report["HG-7_eos_rate"] = dict(
        termination_rate=float(ok_term / n),
        baseline_rate_from_interim=dict(B0_B1_B2=1.0),
        passed=bool(ok_term >= 0.9 * n))
    print(f"[HG-6/7] {ok_term}/{n} terminate with EOS -> "
          f"{'PASS' if ok_term >= 0.9 * n else 'FAIL'}")


def load_v1_module():
    """The exact m2_gdep.py that ran the contaminated grid, as a module."""
    code = subprocess.check_output(
        ["git", "-C", "/media/disk2/YZX/research/EADP", "show",
         f"{V1_COMMIT}:Qwen_vl/scripts/discovery/m2_gdep.py"],
        text=False)
    path = os.path.join(OUTPUT_DIR, "m2_gdep_v1_from_git.py")
    with open(path, "wb") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location("m2_gdep_v1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m2_gdep_v1"] = mod
    spec.loader.exec_module(mod)
    return mod


def gate_8(model, keys, rows, report):
    """HG-8: the decode fix changes no prefill quantity, and no RENUMBER decode.

    v1 (the grid's code, read from git) and v2 (this file) run the same 8
    prefills: scores must be bit-identical and the selected token sets exactly
    equal. Then the RENUMBER arm decodes 24 forced steps on both engines from
    both prefills: ids must match step-for-step. Together these are what let
    the existing C1-R records stand as measurements of the fixed engine."""
    v1 = load_v1_module()
    eng_v2 = build(model, mode=MODE_GDEP, budget=256, selector="topk",
                   n_arm=960, seed=2, pos_policy=POLICY_PRESERVE, tag="HG8-v2")
    eng_v1 = v1.GDEPEngine.from_checkpoint(
        v1.GDEPConfig(mode=v1.MODE_GDEP, budget=256, selector="topk",
                      n_arm=960, seed=2, pos_policy=v1.POLICY_PRESERVE,
                      tag="HG8-v1"), model=model, max_new_tokens=32)
    recs, ok = [], True
    for r in rows[:N_ID]:
        key = keys[r]
        ds, msg = message_for(model, key)
        p1 = eng_v1.prepare(msg, ds)
        s1, i1 = eng_v1.prefill(p1)
        p2 = eng_v2.prepare(msg, ds)
        s2, i2 = eng_v2.prefill(p2)
        sc1 = np.asarray(i1["scores"], np.float32)
        sc2 = np.asarray(i2["scores"], np.float32)
        bit = bool(sc1.tobytes() == sc2.tobytes())
        setsame = i1["select_idx"] == i2["select_idx"]
        keep = i1["keep_full"] == i2["keep_full"]
        ok &= bit and setsame and keep
        recs.append(dict(key=key, scores_bit_identical=bit,
                         select_idx_identical=bool(setsame),
                         keep_full_identical=bool(keep),
                         n_sym_diff_select=len(set(i1["select_idx"])
                                                ^ set(i2["select_idx"]))))
        print(f"[HG-8] {key}: scores bit={bit} top256={setsame} keep={keep}")
        del s1, s2, p1, p2
        torch.cuda.empty_cache()

    # RENUMBER decode equivalence, both engines, forced 24 steps past EOS
    dn, ok_d = [], True
    engR2 = build(model, mode=MODE_GDEP, budget=256, selector="topk",
                  n_arm=960, seed=2, pos_policy=POLICY_RENUMBER, tag="HG8R-v2")
    engR1 = v1.GDEPEngine.from_checkpoint(
        v1.GDEPConfig(mode=v1.MODE_GDEP, budget=256, selector="topk",
                      n_arm=960, seed=2, pos_policy=v1.POLICY_RENUMBER,
                      tag="HG8R-v1"), model=model, max_new_tokens=32)
    for r in rows[:4]:
        key = keys[r]
        ds, msg = message_for(model, key)
        p1 = engR1.prepare(msg, ds)
        s1, _ = engR1.prefill(p1)
        d1 = engR1.decode(s1, FORCED_STEPS, ignore_eos=True)
        p2 = engR2.prepare(msg, ds)
        s2, _ = engR2.prefill(p2)
        d2 = engR2.decode(s2, FORCED_STEPS, ignore_eos=True)
        same = d1 == d2
        ok_d &= same
        dn.append(dict(key=key, ids_identical=bool(same), n=FORCED_STEPS,
                       first_diff=next((i for i, (a, b) in enumerate(zip(d1, d2))
                                        if a != b), None)))
        print(f"[HG-8] RENUMBER decode {key}: v1==v2 {same}")
        del s1, s2, p1, p2
        torch.cuda.empty_cache()
    report["HG-8_prefill_and_renumber_invariance"] = dict(
        prefill=recs, renumber_decode=dn,
        passed=bool(ok and ok_d))
    print(f"[HG-8] -> {'PASS' if ok and ok_d else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m2_fixgates")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model = common.load_model(eadp_model_name(256, 0.5, 2.0), max_new_tokens=512)
    model.model.eval()
    keys, rows = heldout_keys()
    report = dict(stage="M2 amendment", engine_version=ENGINE_VERSION,
                  v1_commit=V1_COMMIT, gates=[])
    gates = [("HG-1", gate_1), ("HG-2/3", gate_positions),
             ("HG-4", gate_4_long), ("HG-5", gate_5),
             ("HG-6/7", gate_67), ("HG-8", gate_8)]
    for name, fn in gates:
        if args.only and name not in args.only:
            continue
        print(f"\n=== {name} ===")
        try:
            fn(model, keys, rows, report)
        except Exception:
            traceback.print_exc()
            report[f"gate_{name}"] = dict(passed=False,
                                          error=traceback.format_exc()[-2000:])
    flat = [v for k, v in report.items() if isinstance(v, dict) and "passed" in v]
    report["all_passed"] = bool(flat and all(v["passed"] for v in flat))
    dump_json(f"{args.tag}.json", report)
    print(f"\n[fixgates] all_passed={report['all_passed']}")
    sys.exit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()
