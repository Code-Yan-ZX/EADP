"""E0 M1 correctness gates N1-N5 (prereg §3.3).

Any gate failure stops E0: no accuracy number may be produced before every
gate passes.  Results -> outputs/e0/e0_gates.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

DISC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "discovery")
sys.path.insert(0, DISC_DIR)
import common  # noqa: E402  (env bootstrap)

OUT_DIR = os.path.join(common.QWEN_ROOT, "outputs", "e0")

N1_SAMPLES = 16
N3_K = 256
N4_SAMPLES = 12
N4_DECODE_STEPS = 16
N5_SAMPLES = 32
N1_DECODE_STEPS = 32


def dev_rows(ds, n):
    plan = json.load(open(os.path.join(OUT_DIR, "e0_plan.json")))
    return plan["datasets"][ds]["dev_rows"][:n]


def build_items(model, spec):
    """spec: list of (ds, row_idx); returns list of dicts with message etc."""
    from collections import OrderedDict
    by_ds = OrderedDict()
    for ds, idx in spec:
        by_ds.setdefault(ds, []).append(idx)
    items = []
    for ds, idxs in by_ds.items():
        dataset = common.build_dataset(ds)
        if hasattr(model, "set_dump_image"):
            model.set_dump_image(dataset.dump_image)
        for idx in idxs:
            row = dataset.data.iloc[idx]
            msg = common.build_message(model, dataset, ds, row)
            items.append(dict(ds=ds, idx=int(idx), message=msg, row=row))
    return items


def stock_reference(engine, prep, n_steps):
    """Stock prefill + n_steps greedy decode logits, through the untouched
    ForConditionalGeneration.forward (the exact path stock generate drives)."""
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


def gate_n1(engine, items):
    """Identity vs stock: prefill + every decode logit within 1e-3, 32 greedy
    tokens identical."""
    res = dict(desc="N1 identity: keep-all native vs stock generate path",
               samples=[], max_absdiff_prefill=0.0, max_absdiff_decode=0.0,
               token_mismatch=0, passed=True)
    for it in items:
        prep = engine.prepare(it["message"], it["ds"])
        ref_pre, ref_steps = stock_reference(engine, prep, N1_DECODE_STEPS)
        ref_toks = [int(s.argmax().item()) for s in ref_steps]
        # N+1 greedy tokens from the native side -> N decode forwards, so every
        # one of the 32 decode steps is compared logit-for-logit
        V, DS = engine.encode(prep)
        st = engine.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d",
                            full_lm_head=True)
        gen_ids, _ = engine.decode(st, N1_DECODE_STEPS + 1, ignore_eos=True)
        d_pre = float((st.logits[0] - ref_pre).abs().max().item())
        d_dec = 0.0
        mism = 0
        for n, (lg, rlg) in enumerate(zip(st.decode_logits, ref_steps)):
            d_dec = max(d_dec, float((lg[0] - rlg).abs().max().item()))
        # token stream: gen_ids = [tok0..tok32]; ref chain = tok0 (argmax
        # ref_prefill), then ref_toks = tok1..tok32
        if int(ref_pre.argmax().item()) != gen_ids[0]:
            mism += 1
        for n in range(N1_DECODE_STEPS):
            if gen_ids[1 + n] != ref_toks[n]:
                mism += 1
        ok = d_pre <= 1e-3 and d_dec <= 1e-3 and mism == 0
        res["samples"].append(dict(ds=it["ds"], idx=it["idx"],
                                   d_prefill=d_pre, d_decode=d_dec,
                                   token_mismatch=mism, ok=bool(ok)))
        res["max_absdiff_prefill"] = max(res["max_absdiff_prefill"], d_pre)
        res["max_absdiff_decode"] = max(res["max_absdiff_decode"], d_dec)
        res["token_mismatch"] += mism
        res["passed"] = bool(res["passed"] and ok)
        print(f"  [N1] {it['ds']}_{it['idx']} d_pre={d_pre:.2e} "
              f"d_dec={d_dec:.2e} mism={mism}", flush=True)
    return res


def gate_n2(engine, items):
    """K=1024 with deepstack off must differ from N1's native run."""
    res = dict(desc="N2 DeepStack injection is live (off vs on)", samples=[],
               min_absdiff=1e9, passed=True)
    for it in items:
        prep = engine.prepare(it["message"], it["ds"])
        V, DS = engine.encode(prep)
        st_on = engine.prefill(prep, V, DS, None, deepstack=True, pos="mrope3d",
                               full_lm_head=True)
        st_off = engine.prefill(prep, V, DS, None, deepstack=False, pos="mrope3d",
                                full_lm_head=True)
        d = float((st_on.logits[0] - st_off.logits[0]).abs().max().item())
        ok = d > 0.1
        res["samples"].append(dict(ds=it["ds"], idx=it["idx"], absdiff=d, ok=bool(ok)))
        res["min_absdiff"] = min(res["min_absdiff"], d)
        res["passed"] = bool(res["passed"] and ok)
        print(f"  [N2] {it['ds']}_{it['idx']} |d|={d:.4f}", flush=True)
    return res


def gate_n3(engine, items, seed=20260929):
    """K=256 arbitrary keep: kept tokens carry their full-sequence (t,h,w);
    kept text positions monotone; decode step n at prefill_max + 1 + n."""
    res = dict(desc="N3 3-D position subset + decode continuation", samples=[],
               passed=True)
    for i, it in enumerate(items):
        prep = engine.prepare(it["message"], it["ds"])
        V, DS = engine.encode(prep)
        g = torch.Generator().manual_seed(seed + i)
        keep = torch.randperm(prep["n_vis"], generator=g)[:N3_K].sort().values
        st = engine.prefill(prep, V, DS, keep, deepstack=True, pos="mrope3d")
        engine.decode(st, 8, ignore_eos=True)

        pos3d_full, _ = engine.positions_full(prep)
        keep_seq = st.keep_seq
        # (a) every kept sequence position equals its full-sequence position
        full_slice = pos3d_full[:, 0, keep_seq]          # [3, Lk]
        same = bool((st.positions_prefill[:, 0] == full_slice).all().item())
        # (b) kept visual tokens specifically
        vis_slots = (keep_seq >= prep["img_start"]) & (keep_seq < prep["img_end"])
        vis_pos_kept = st.positions_prefill[:, 0, vis_slots]
        vis_pos_full = pos3d_full[:, 0, prep["img_start"] + keep.to(keep_seq.device)]
        vis_same = bool((vis_pos_kept == vis_pos_full).all().item())
        # (c) text positions strictly monotone (w channel over kept text)
        txt_pos = st.positions_prefill[0, 0, ~vis_slots]
        mono = bool((txt_pos[1:] > txt_pos[:-1]).all().item())
        # (d) decode step n sits at prefill_max + 1 + n
        expect = [st.prefill_max_pos + 1 + n for n in range(len(st.decode_positions))]
        dec_ok = st.decode_positions == expect
        ok = same and vis_same and mono and dec_ok
        res["samples"].append(dict(ds=it["ds"], idx=it["idx"], subset_ok=same,
                                   visual_ok=vis_same, text_monotone=mono,
                                   decode_ok=bool(dec_ok), ok=bool(ok)))
        res["passed"] = bool(res["passed"] and ok)
        print(f"  [N3] {it['ds']}_{it['idx']} subset={same} vis={vis_same} "
              f"mono={mono} dec={dec_ok}", flush=True)
    return res


def with_oom_retry(fn, tries=6, wait_s=150, tag=""):
    """This host runs an Ollama service that intermittently grabs ~27 GB of
    GPU memory for a Qwen3.6-35B chat model; it unloads after ~5 min idle.
    Retry on CUDA OOM instead of failing the gate on a foreign process."""
    import time as _t
    for attempt in range(tries):
        try:
            return fn()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [oom] {tag} attempt {attempt+1}/{tries}; "
                  f"sleeping {wait_s}s (ollama keep-alive?)", flush=True)
            _t.sleep(wait_s)
    raise RuntimeError(f"OOM persists after {tries} retries: {tag}")


def gate_n4(engine, items, seed=20260929, arms=None):
    """12 samples x every arm: exact K, no dup/OOB, DS lengths, mask/position/
    cache consistency, layer_calls == 36*(1+decode_steps)."""
    arms = arms or [("identity", 1024), ("b2", 256), ("b1", 256),
                    ("divprune", 256), ("cdpruner", 256), ("hiprune", 256),
                    ("random", 256), ("b2", 64)]
    res = dict(desc="N4 invariants per arm", arms={}, passed=True)
    for name, K in arms:
        key = f"{name}|K={K}"
        rows = []
        torch.cuda.empty_cache()
        for i, it in enumerate(items):
            torch.cuda.empty_cache()
            def _one(i=it, name=name, K=K, seed=seed):
                prep = engine.prepare(i["message"], i["ds"])
                if prep["n_vis"] == 0:
                    return None
                vz = None
                if name == "hiprune":
                    V, DS, attn = engine.encode_with_importance(prep)
                elif name == "visionzip":
                    from model.baselines.visionzip import (
                        visual_forward_with_visionzip)
                    V, DS, am, ak = visual_forward_with_visionzip(
                        engine.inner.visual, prep["pv"], prep["gthw"])
                    vz = (am, ak)
                    attn = None
                else:
                    V, DS = engine.encode(prep)
                    attn = None
                text_mean, text_seq = engine.instruction_embeds(i["message"], i["ds"])
                ctx = dict(prep=prep, V=V, DS=DS, K=K, engine=engine,
                           text_mean=text_mean, text_seq=text_seq,
                           attn_list=attn, vz=vz, seed=seed)
                from model.e0_selectors import run_selector
                if name == "identity":
                    keep, prune_layers, V_sel, DS_sel = None, None, None, None
                else:
                    sel_out = run_selector(name, K, ctx)
                    if isinstance(sel_out, dict):
                        keep = sel_out.get("keep_idx")
                        prune_layers = sel_out.get("prune_layers")
                        V_sel = sel_out.get("V_sel")
                        DS_sel = sel_out.get("DS_sel")
                    else:
                        keep, prune_layers, V_sel, DS_sel = sel_out, None, None, None
                if prune_layers:
                    st = engine.run_llm_forward(prep, V, DS, keep,
                                                deepstack=True, pos="mrope3d",
                                                prune_layers=prune_layers)
                else:
                    st = engine.prefill(prep, V, DS, keep, deepstack=True,
                                        pos="mrope3d", V_sel=V_sel, DS_sel=DS_sel)
                engine.decode(st, N4_DECODE_STEPS, ignore_eos=True)
                # decode(max_new=N) performs N-1 forwards (the Nth token breaks
                # before a forward); the gate counts actual forwards
                n_fwd = len(st.decode_logits)
                return engine.invariants(prep, st, V, DS, n_decode=n_fwd)

            inv = with_oom_retry(_one, tag=f"{key}#{i}")
            if inv is None:
                continue
            inv.update(ds=it["ds"], idx=it["idx"])
            rows.append(inv)
        n_bad = sum(1 for r in rows
                    if not (r["layer_calls_ok"] and r["cache_ok"] and r["no_dup"]
                            and r["in_range"] and r["ascending"]
                            and r["n_vis_kept"] == min(K, r["n_vis_full"])
                            and all(d == min(K, r["n_vis_full"]) for d in r["ds_lengths"])))
        res["arms"][key] = dict(n=len(rows), n_bad=n_bad, rows=rows)
        res["passed"] = bool(res["passed"] and n_bad == 0 and len(rows) > 0)
        print(f"  [N4] {key}: {len(rows)} samples, {n_bad} bad", flush=True)
    return res


def gate_n5(engine, seed=20260929):
    """Legacy-mode B2 vs the archived SAGE-confirmation B2 predictions."""
    conf = json.load(open(os.path.join(common.OUTPUT_DIR, "sage_conf_B2.json")))
    records = conf["records"]
    keys = sorted(records.keys())
    idxs = common.sample_indices(len(keys), N5_SAMPLES)
    chosen = [keys[i] for i in idxs]
    items = build_items(engine.vlm,
                        [(k.rsplit("_", 1)[0], int(k.rsplit("_", 1)[1]))
                         for k in chosen])
    by_key = {f"{it['ds']}_{it['idx']}": it for it in items}
    res = dict(desc="N5 legacy-mode B2 vs archived SAGE-confirmation B2",
               samples=[], n_agree=0, n_total=0, passed=True)
    for k in chosen:
        it = by_key[k]
        out = engine.generate(it["message"], it["ds"], K=256, selector="b2",
                              deepstack=False, pos="1d", max_new_tokens=2048)
        pred = out["text"].strip()
        arch = records[k]["prediction"].strip()
        agree = pred == arch
        res["samples"].append(dict(key=k, agree=bool(agree), pred=pred, archived=arch))
        res["n_total"] += 1
        res["n_agree"] += int(agree)
        print(f"  [N5] {k} agree={agree} pred={pred!r} arch={arch!r}", flush=True)
    res["rate"] = res["n_agree"] / max(1, res["n_total"])
    res["passed"] = bool(res["rate"] >= 0.95)
    return res


def main():
    t0 = time.time()
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=2048)
    from model.native_qwen3 import NativeEngine
    engine = NativeEngine(model)

    n1_spec = ([( "TextVQA_VAL", i) for i in dev_rows("TextVQA_VAL", 6)]
               + [("DocVQA_VAL", i) for i in dev_rows("DocVQA_VAL", 5)]
               + [("OCRBench", i) for i in dev_rows("OCRBench", 5)])
    items_n1 = build_items(model, n1_spec)
    n1 = gate_n1(engine, items_n1)
    n2 = gate_n2(engine, items_n1)
    n3 = gate_n3(engine, items_n1[:6])
    n4_spec = ([("TextVQA_VAL", i) for i in dev_rows("TextVQA_VAL", 4)]
               + [("DocVQA_VAL", i) for i in dev_rows("DocVQA_VAL", 4)]
               + [("OCRBench", i) for i in dev_rows("OCRBench", 4)])
    n4 = gate_n4(engine, build_items(model, n4_spec))
    n5 = gate_n5(engine)

    gates = dict(env=dict(torch=torch.__version__,
                          transformers=__import__("transformers").__version__,
                          cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(0)),
                 n1=n1, n2=n2, n3=n3, n4=n4, n5=n5,
                 all_passed=bool(n1["passed"] and n2["passed"] and n3["passed"]
                                 and n4["passed"] and n5["passed"]),
                 elapsed_s=round(time.time() - t0, 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "e0_gates.json"), "w") as f:
        json.dump(gates, f, indent=1)
    print(f"[gates] all_passed={gates['all_passed']} "
          f"({gates['elapsed_s']}s) -> e0_gates.json", flush=True)
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
