"""
S3-A step 1 (GPU): measure the conditional marginal utility of visual tokens.

For every frozen case (s3a_cases.py) and every fixed-budget context S, this
computes the teacher-forced gold-answer loss L(y | S) under PRE-LLM delivery:
the prompt is [prefix embeds | vision embeddings of S in raster order | suffix
embeds] -- exactly the sequence the incumbent's pruned path hands the LLM --
and the answer span is appended for teacher forcing. All marginal utilities
are differences of that loss:

    ref          L(S)                                    5 per instance
    add          L(S) - L(S + {i} - {r})   budget 256    12 per context
    add_norem    L(S) - L(S + {i})         budget 257    12 per context
    leave        L(S) - L(S - {j} + {f})   budget 256    8  per context
    ri           add with 3 random removals instead of r 18 (base only)

    r = argmin over S of the frozen GDEP student score (the "lowest-value"
    token). Its identity, student score and teacher rank are recorded with
    every add measurement, and `ri` exists precisely to test whether that
    identity matters (S2-C2 found the remove side inert; re-checked here).

The utility is answer-level: gold-answer NLL, batched over the instance's
deduplicated golds (<= 6) and averaged. No hidden-state proxy is used.

Gates (run first; measurement refuses to start unless they pass):
    S3A-G1   recomputed student top-256 (S_base) == the live GDEP engine's
             select_idx (LOCAL-MLP n960 s2, topk@256) on probe instances.
    S3A-G2   direct splicing of vis[S] == what the incumbent pruner + S2-C2
             indicator map delivers (sets identical, embeddings bit-comparable).
    S3A-G3   teacher forcing reproduces greedy generation from the same prompt
             at every compared position.
    S3A-G3b  batched-gold NLL == single-gold NLL (padding is inert).
    S3A-G4   determinism: repeated measurements bit-identical.

Writes ``s3a_nll.json`` incrementally (resume by completed keys). Vision
tower outputs are cached under s3a_vis/ as bit-exact bf16 (uint16 view).
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
import common                                                   # noqa: E402
from common import eadp_model_name                              # noqa: E402
from s1_audit import OUT                                        # noqa: E402
import s3a_common as C                                          # noqa: E402
from scoring import per_sample_hits                             # noqa: E402

GEN_CAP = 64            # max generated tokens for the base predictions
G3_TOKENS = 12          # generated tokens teacher-forced in gate G3


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _load_bf16(path, dev):
    """Reload a bf16 tensor saved as a uint16 view (bit-exact)."""
    return torch.from_numpy(np.load(path)).view(torch.bfloat16).to(dev)


class S3AHarness:
    """Teacher-forced answer NLL under an explicitly-given retained set."""

    def __init__(self, model):
        self.vlm = model
        self.model = model.model                    # Qwen3VLForConditionalGeneration
        self.dev = next(model.model.parameters()).device
        self.tok = model.processor.tokenizer
        self.image_token_id = model.model.config.image_token_id
        self.emb = self.model.get_input_embeddings()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        self.eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    # ---- per-instance preparation ----------------------------------------
    def prep(self, message, ds, key):
        """Vision embeddings (disk-cached, bf16-exact) + prompt skeleton."""
        vis_dir = os.path.join(OUT, C.VIS_CACHE)
        vis_path = os.path.join(vis_dir, f"{key}.npy")
        messages = self.vlm._build_messages(message, dataset=ds)
        inputs = self.vlm._processor_inputs(messages)
        ids = inputs["input_ids"]
        pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == C.N_VIS, f"{e - s} visual tokens"

        with torch.no_grad():
            if os.path.exists(vis_path):
                vis = _load_bf16(vis_path, self.dev)
            else:
                from vlmeval.vlm.qwen3_vl.model_fixed_res import \
                    unwrap_visual_output
                pv = inputs["pixel_values"].type(self.model.visual.dtype)
                vis = unwrap_visual_output(
                    self.model.visual(pv, grid_thw=inputs["image_grid_thw"]))
                vis = vis.detach()
                os.makedirs(vis_dir, exist_ok=True)
                tmp = vis_path + ".tmp.npy"
                np.save(tmp, vis.view(torch.uint16).cpu().numpy())
                os.replace(tmp, vis_path)
                vis = vis.to(self.dev)
            emb = self.emb(ids)
        prefix, suffix = emb[0, :s], emb[0, e:]
        return dict(vis=vis, prompt_prefix=prefix, suffix=suffix,
                    n_prefix=s)

    def answer_ids(self, golds):
        out = []
        for g in golds:
            t = self.tok(str(g), add_special_tokens=False,
                         return_tensors="pt").input_ids[0]
            t = t[:C.ANSWER_TOKEN_CAP]
            if t.numel() > 0:
                out.append(t.to(self.dev))
        assert out, "instance has no non-empty gold answer"
        return out

    # ---- core forward ------------------------------------------------------
    def _prompt(self, prep, kept):
        kept_t = torch.as_tensor(sorted(int(t) for t in kept), device=self.dev)
        visk = prep["vis"].index_select(0, kept_t)
        return torch.cat([prep["prompt_prefix"], visk, prep["suffix"]], dim=0)

    @torch.no_grad()
    def nll(self, prep, kept, ans_ids):
        """mean over golds of (mean per-token NLL of that gold under `kept`)."""
        prompt = self._prompt(prep, kept)
        P = prompt.shape[0]
        Lmax = max(t.numel() for t in ans_ids)
        B, D = len(ans_ids), prompt.shape[1]
        full = torch.zeros(B, P + Lmax, D, dtype=prompt.dtype,
                           device=prompt.device)
        full[:, :P] = prompt
        am = torch.ones(B, P + Lmax, dtype=torch.long, device=self.dev)
        ids_pad = torch.zeros(B, Lmax, dtype=torch.long, device=self.dev)
        valid = torch.zeros(B, Lmax, dtype=torch.bool, device=self.dev)
        for b, t in enumerate(ans_ids):
            n = t.numel()
            full[b, P:P + n] = self.emb(t.unsqueeze(0))[0]
            ids_pad[b, :n] = t
            valid[b, :n] = True
            am[b, P + n:] = 0
        pos1 = torch.arange(P + Lmax, device=self.dev).view(1, 1, -1) \
            .expand(3, B, -1)
        out = self.model(inputs_embeds=full, attention_mask=am,
                         position_ids=pos1.contiguous(), use_cache=False,
                         return_dict=True)
        logits = out.logits[:, P - 1:P - 1 + Lmax, :].float()
        ce = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1), ids_pad, reduction="none") * valid
        per_gold = (ce.sum(1) / valid.sum(1)).tolist()
        first = ce[:, 0].tolist()
        del out, logits
        return float(np.mean(per_gold)), per_gold, first, P

    @torch.no_grad()
    def generate(self, prep, kept, max_new=GEN_CAP):
        """greedy decode under the kept set; returns raw token ids + text.

        Note: with inputs_embeds and no input_ids, HF generate returns ONLY
        the continuation (the published diag_selectors/s2c2 harness decodes
        it without slicing) -- same kwargs as the B1-side harness for like-
        for-like correctness scoring.
        """
        prompt = self._prompt(prep, kept)
        am = torch.ones(1, prompt.shape[0], dtype=torch.long, device=self.dev)
        kw = dict(self.vlm.generate_kwargs)
        kw["max_new_tokens"] = max_new
        out = self.model.generate(inputs_embeds=prompt.unsqueeze(0),
                                  attention_mask=am, do_sample=False, **kw)
        ids = out[0].tolist()
        for k, t in enumerate(ids):
            if t in self.eos:
                ids = ids[:k]
                break
        text = self.tok.decode(ids, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        return ids, text


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def run_gates(h, cases, datasets):
    rep = {}
    msgs = {}
    for c in cases[:6]:
        ds = datasets[c["ds"]]
        h.vlm.set_dump_image(ds.dump_image)
        msgs[c["key"]] = common.build_message(h.vlm, ds, c["ds"],
                                              ds.data.iloc[c["idx"]])

    # ---- G1: recomputed S_base == live GDEP engine select_idx --------------
    from m2_gdep import GDEPConfig, GDEPEngine, MODE_GDEP
    cfg = GDEPConfig(mode=MODE_GDEP, budget=256, selector="topk",
                     n_arm=C.STUDENT_N_ARM, seed=C.STUDENT_SEED, tag="S3A-G1")
    eng = GDEPEngine.from_checkpoint(cfg, model=h.vlm, max_new_tokens=8)
    g1 = []
    for c in cases[:3]:
        msg = msgs[c["key"]]
        prep_g = eng.prepare(msg, c["ds"])
        st, info = eng.prefill(prep_g)
        sel = sorted(int(t) for t in info["select_idx"])
        g1.append(dict(key=c["key"], identical=bool(sel == c["S"]),
                       symdiff=len(set(sel) ^ set(c["S"]))))
        print(f"[G1] {c['key']}: identical={g1[-1]['identical']} "
              f"symdiff={g1[-1]['symdiff']}")
        del st, prep_g
        torch.cuda.empty_cache()
    rep["G1_engine_select"] = dict(rows=g1,
                                   passed=all(r["identical"] for r in g1))

    # ---- G2: direct splice == indicator-map delivery -----------------------
    from s2b_run import GradientPruner
    from vlmeval.vlm.qwen3_vl.model_fixed_res import unwrap_visual_output
    pruner = GradientPruner(visual_token_num=256, alpha=0.5, beta=2.0,
                            visual_dim=h.vlm.pruner.visual_dim,
                            spatial_merge_size=h.vlm.pruner.spatial_merge_size,
                            selector="topk", capture=True)
    pruner.to(h.dev).eval()
    saved_pruner = h.vlm.pruner
    h.vlm.pruner = pruner
    g2 = []
    for c in cases[:2]:
        msg = msgs[c["key"]]
        p = h.prep(msg, c["ds"], c["key"])
        messages = h.vlm._build_messages(msg, dataset=c["ds"])
        inputs = h.vlm._processor_inputs(messages)
        ind = torch.zeros(C.N_VIS, dtype=torch.float32, device=h.dev)
        ind[c["S"]] = 1.0
        pruner.override = ind.unsqueeze(0)
        instr = h.vlm._get_instruction_sequence_embedding(msg, dataset=c["ds"])
        pv = inputs["pixel_values"].type(h.model.visual.dtype)
        gthw = inputs["image_grid_thw"]
        with torch.no_grad():
            vis = unwrap_visual_output(h.model.visual(pv, grid_thw=gthw))
            n_img = int(gthw.shape[0])
            pruned, sizes = pruner(vis, instr.mean(dim=1).expand(n_img, -1),
                                   instr.expand(n_img, -1, -1), gthw)
        got = pruner.last_capture.get("select_idx")
        delivered = sorted(int(t) for t in got[0].cpu().numpy())
        direct = p["vis"].index_select(
            0, torch.as_tensor(c["S"], device=h.dev))
        same_set = delivered == sorted(c["S"])
        mad = float((pruned.float() - direct.float()).abs().max()) \
            if same_set else float("nan")
        g2.append(dict(key=c["key"], same_set=bool(same_set),
                       embeds_max_abs_diff=mad))
        print(f"[G2] {c['key']}: same_set={same_set} embeds mad={mad:.3e}")
    h.vlm.pruner = saved_pruner
    rep["G2_indicator_delivery"] = dict(
        rows=g2, passed=all(r["same_set"] and r["embeds_max_abs_diff"] < 1e-2
                            for r in g2))

    # ---- G3: teacher forcing == greedy generation ---------------------------
    g3 = []
    for c in cases[:3]:
        p = h.prep(msgs[c["key"]], c["ds"], c["key"])
        ids, text = h.generate(p, c["S"])
        n = min(len(ids), G3_TOKENS)
        if n == 0:
            g3.append(dict(key=c["key"], skipped=True, text=text[:60]))
            continue
        gen = torch.tensor(ids[:n], device=h.dev)
        prompt = h._prompt(p, c["S"])
        P = prompt.shape[0]
        full = torch.cat([prompt, h.emb(gen.unsqueeze(0))[0]], dim=0) \
            .unsqueeze(0)
        am = torch.ones(1, full.shape[1], dtype=torch.long, device=h.dev)
        pos1 = torch.arange(full.shape[1], device=h.dev).view(1, 1, -1) \
            .expand(3, 1, -1).contiguous()
        with torch.no_grad():
            out = h.model(inputs_embeds=full, attention_mask=am,
                          position_ids=pos1, use_cache=False, return_dict=True)
        lg = out.logits[0, P - 1:P - 1 + n].float()
        pred = lg.argmax(-1).tolist()
        match = int(sum(int(a == b) for a, b in zip(pred, gen.tolist())))
        g3.append(dict(key=c["key"], n_gen=n, trajectory_match=match,
                       first_ok=bool(pred[0] == int(gen[0])),
                       text=text[:60]))
        print(f"[G3] {c['key']}: argmax match {match}/{n} "
              f"pred={h.tok.decode(pred[:6])!r} gen={h.tok.decode(gen.tolist()[:6])!r}")
        del out, lg
        torch.cuda.empty_cache()
    g3_live = [r for r in g3 if not r.get("skipped")]
    rep["G3_teacher_forcing"] = dict(
        rows=g3, passed=bool(g3_live) and
        all(r["first_ok"] and r["trajectory_match"] == r["n_gen"]
            for r in g3_live))

    # ---- G3b: batched golds == single gold ---------------------------------
    g3b = []
    for c in cases[:2]:
        p = h.prep(msgs[c["key"]], c["ds"], c["key"])
        ans = h.answer_ids(c["golds"])
        Lb, per, first, P = h.nll(p, c["S"], ans)
        singles = [h.nll(p, c["S"], [t])[0] for t in ans]
        worst = float(np.max(np.abs(np.array(per) - np.array(singles))))
        g3b.append(dict(key=c["key"], batched=Lb, worst_abs_diff=worst))
        print(f"[G3b] {c['key']}: batch-vs-single worst |d| = {worst:.3e}")
    # tolerance = the bf16 logit ULP at the CE scale these logits live at
    # (~8e-2 nats observed; 0.25 is a hard bound, and the same floor is
    # re-measured per instance as the `noise|floor` probe in measure_case)
    rep["G3b_batching"] = dict(rows=g3b, noise_floor_nats=0.25,
                               passed=all(r["worst_abs_diff"] < 0.25
                                          for r in g3b))

    # ---- G4: determinism ----------------------------------------------------
    c = cases[0]
    p = h.prep(msgs[c["key"]], c["ds"], c["key"])
    ans = h.answer_ids(c["golds"])
    L1, _, _, _ = h.nll(p, c["S"], ans)
    L2, _, _, _ = h.nll(p, c["S"], ans)
    rep["G4_determinism"] = dict(L1=L1, L2=L2, passed=bool(L1 == L2))
    print(f"[G4] {L1!r} == {L2!r} -> {rep['G4_determinism']['passed']}")

    rep["all_passed"] = all(rep[k]["passed"] for k in
                            ("G1_engine_select", "G2_indicator_delivery",
                             "G3_teacher_forcing", "G3b_batching",
                             "G4_determinism"))
    return rep


# ---------------------------------------------------------------------------
# main measurement
# ---------------------------------------------------------------------------
def measure_case(h, c):
    """All NLL measurements for one case; returns (records, removal map)."""
    p = h.prep(c["_msg"], c["ds"], c["key"])
    ans = h.answer_ids(c["golds"])
    s = c["_student"]
    grank = c["_grank"]

    recs = []

    def L(kept, tag, token=None, removed=None):
        Lm, per, first, P = h.nll(p, kept, ans)
        recs.append(dict(tag=tag, ctx=tag.split("|")[0], kind=tag.split("|")[1],
                         token=(int(token) if token is not None else None),
                         removed=(int(removed) if removed is not None else None),
                         removed_grank=(grank.get(int(removed))
                                        if removed is not None else None),
                         removed_sscore=(float(s[int(removed)])
                                         if removed is not None else None),
                         L=Lm, per_gold=per, first_tok=first,
                         budget=len(kept)))
        return Lm

    refs = {}
    for name, S in c["contexts"].items():
        refs[name] = L(S, f"{name}|ref")
    # noise floor: same set, same golds, re-batched one-gold-per-row. The max
    # |difference| is this instance's bf16 resolution floor for deltas.
    Lb, per, _, _ = h.nll(p, c["S"], ans)
    singles = [h.nll(p, c["S"], [t])[0] for t in ans]
    recs.append(dict(tag="noise|floor", ctx="base", kind="floor", token=None,
                     removed=None, removed_grank=None, removed_sscore=None,
                     L=float(np.max(np.abs(np.array(per) - np.array(singles)))),
                     per_gold=per, first_tok=[], budget=len(c["S"])))
    # removal token per context: student-argmin over the context members
    rem = {name: int(min(S, key=lambda t: (float(s[t]), t)))
           for name, S in c["contexts"].items()}
    filler = int(c["filler"])

    for name, S in c["contexts"].items():
        sS = set(int(t) for t in S)
        r = rem[name]
        for i in c["P"]:
            i = int(i)
            assert i not in sS, (name, i)
            L([t for t in S if t != r] + [i], f"{name}|add",
              token=i, removed=r)
        for i in c["P"]:
            L(list(S) + [int(i)], f"{name}|add_norem", token=int(i),
              removed=None)
        for j in c["J"]:
            j = int(j)
            if j not in sS:
                continue
            f = filler if filler not in sS else \
                next(t for t in range(C.N_VIS) if t not in sS)
            L([t for t in S if t != j] + [f], f"{name}|leave",
              token=j, removed=j)

    # removal-identity control on the base context
    for i, alts in (c["removal_identity"] or {}).items():
        i = int(i)
        for a in alts:
            a = int(a)
            if a == i or i in c["contexts"]["base"]:
                continue
            L([t for t in c["S"] if t != a] + [i], "base|ri",
              token=i, removed=a)
    return recs, refs, rem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="s3a_nll")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    cases_doc = json.load(open(os.path.join(OUT, C.CASES_JSON)))
    cases = cases_doc["cases"]
    backbone = {b["key"]: b for b in C.load_backbone()}
    for c in cases:
        b = backbone[c["key"]]
        c["_student"] = b["student"]
        g = b["teacher"]
        gorder = np.argsort(-g, kind="stable")
        c["_grank"] = {int(t): int(r) for r, t in enumerate(gorder)}

    out_path = os.path.join(OUT, f"{args.tag}.json")
    results = {"config": dict(cases=C.CASES_JSON, gen_cap=GEN_CAP,
                              student=f"n{C.STUDENT_N_ARM}s{C.STUDENT_SEED}",
                              w_churn=C.W_CHURN),
               "done_keys": [], "meas": {}, "preds": {}}
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev.get("config") != results["config"]:
            raise RuntimeError("existing s3a_nll.json was produced under a "
                               "different config -- refuse to mix")
        results["done_keys"] = prev.get("done_keys", [])
        results["meas"] = prev.get("meas", {})
        results["preds"] = prev.get("preds", {})
        print(f"[resume] {len(results['done_keys'])} cases already done")

    datasets = {ds: common.build_dataset(ds) for ds in C.DS_ALL}

    model = common.load_model(eadp_model_name(256, 0.5, 2.0),
                              max_new_tokens=args.max_new_tokens)
    model.model.eval()
    torch.set_grad_enabled(False)
    h = S3AHarness(model)

    if args.gates_only:
        rep = run_gates(h, cases, datasets)
        json.dump(rep, open(os.path.join(OUT, "s3a_gates.json"), "w"), indent=1)
        print(f"[gates] all_passed={rep['all_passed']}")
        sys.exit(0 if rep["all_passed"] else 1)

    rep = run_gates(h, cases, datasets)
    results["gates"] = rep
    if not rep["all_passed"]:
        print("[gates] FAILED -- refusing to measure")
        json.dump(results, open(out_path, "w"))
        sys.exit(2)

    todo = [c for c in cases if c["key"] not in results["done_keys"]]
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    for n, c in enumerate(todo):
        ds_obj = datasets[c["ds"]]
        model.set_dump_image(ds_obj.dump_image)
        c["_msg"] = common.build_message(model, ds_obj, c["ds"],
                                         ds_obj.data.iloc[c["idx"]])
        try:
            recs, refs, rem = measure_case(h, c)
            p = h.prep(c["_msg"], c["ds"], c["key"])
            ids, text = h.generate(p, c["S"])
            pred = h.vlm._post_process_response(text)
        except Exception:
            traceback.print_exc()
            print(f"[skip] {c['key']}")
            continue
        row = ds_obj.data.iloc[c["idx"]]
        hits = per_sample_hits(c["ds"], [row], [pred])
        results["meas"][c["key"]] = recs
        results["preds"][c["key"]] = dict(pred=pred, n_tokens=len(ids),
                                          hit=float(hits[0]), refs=refs,
                                          rem=rem)
        results["done_keys"].append(c["key"])
        if (n + 1) % 5 == 0 or n == len(todo) - 1:
            print(f"[{n + 1}/{len(todo)}] {c['key']}  "
                  f"L(base)={refs['base']:.3f}  hit={hits[0]:.2f}  "
                  f"{time.time() - t0:.0f}s")
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
    print(f"[saved] {out_path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
