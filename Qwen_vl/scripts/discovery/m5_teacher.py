"""
M5 (Safe Removal) step 1 -- the deletion-intervention teacher.

This builds the label the whole line rests on, and it is deliberately NOT the
gradient ranking four earlier stages used:

    d_i = L(y | S0 \\ {i}) - L(y | S0)

where L is the teacher-forced gold-answer NLL under PRE-LLM delivery -- the
same harness, the same prompt construction and the same convention S3-A used,
so a `d_i` here and a `leave`-marginal there are the same quantity measured on
a different set.  d_i > 0: deleting i hurts the answer.  d_i <= 0: free, or
better than free.

Why sampled and not exhaustive
------------------------------
|S0| = 256, so an exhaustive single-token sweep is 256 forwards per instance.
The brief asks for a strict but affordable pilot, so each instance is measured
on a stratified sample chosen to cover the axes the trimming rules act on:

    rand      32 uniform draws from S0 -- the UNBIASED view of the population
    maxred     6 highest red_s0    -- what the MAXRED rule would delete
    minred     6 lowest  red_s0    -- the contrast end
    highimp    6 highest imp
    lowimp     6 lowest  imp       -- what the LOWIMP rule would delete
    recon      6 lowest  nn4_recon -- what the RECON rule would delete

The `rand` stratum is what makes the probe's population estimate honest: the
tail strata are enriched on purpose, so every population-level metric is
computed with inverse-inclusion weights (see `m5_probe.py`), and the unweighted
numbers are reported beside them as the tail-focused view.

Group deletion control (brief §2B)
----------------------------------
Single-token effects can be individually unresolvable while a group of them is
not.  For k in {4, 8} and rule in {maxred, lowimp, random} the whole group is
deleted at once and

    D(G) = L(y | S0 \\ G) - L(y | S0)

is recorded, with a real greedy generation for each group arm when `--gen` is
given.  The point is one question only: do the single-token labels predict the
group's harm?  No Shapley values, no combinatorial search.

Usage
    python scripts/discovery/m5_teacher.py --limit 40        # pilot
    python scripts/discovery/m5_teacher.py                   # fit+val, 300
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
from m2_gdep import MODE_PRELLM, GDEPConfig, GDEPEngine, dump_json   # noqa: E402
from m5_common import (BASE_SELECTOR, BUDGET, FEATURES, N_VIS,       # noqa: E402
                       RECON_COL, RED_COL, IMP_COL, bank_items,
                       load_m5_bank, safe_features, vis_row)
from scoring import per_sample_hits                                  # noqa: E402

GEN_CAP = 64
STRATA = (("rand", 24), ("maxred", 6), ("minred", 6), ("highimp", 6),
          ("lowimp", 6), ("recon", 6))
GROUP_KS = (4, 8)
GROUP_RULES = ("maxred", "lowimp", "random")
SEED = 20260925


# ---------------------------------------------------------------------------
# harness -- teacher-forced gold NLL under an explicitly-given retained set
# ---------------------------------------------------------------------------
class TeacherHarness:
    """PRE-LLM delivery: [prefix | vis[S] | suffix] + teacher-forced answer.

    `vis` comes from the frozen bank (fp16 -> bf16, which gate G-VIS proves is
    the tower's own bf16 output bit-for-bit), so no vision pass is repeated.
    """

    def __init__(self, model):
        self.vlm = model
        self.model = model.model
        self.dev = next(model.model.parameters()).device
        self.tok = model.processor.tokenizer
        self.image_token_id = model.model.config.image_token_id
        self.emb = self.model.get_input_embeddings()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.model.config.eos_token_id
        self.eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def prep(self, message, ds, vis: torch.Tensor):
        messages = self.vlm._build_messages(message, dataset=ds)
        inputs = self.vlm._processor_inputs(messages)
        ids = inputs["input_ids"]
        pos = (ids[0] == self.image_token_id).nonzero(as_tuple=True)[0]
        s, e = int(pos[0]), int(pos[-1]) + 1
        assert e - s == N_VIS, f"{e - s} visual tokens"
        with torch.no_grad():
            vis = vis.to(self.dev)
            emb = self.emb(ids)
        return dict(vis=vis, prompt_prefix=emb[0, :s], suffix=emb[0, e:],
                    n_prefix=s)

    def answer_ids(self, golds):
        out = []
        for g in golds:
            t = self.tok(str(g), add_special_tokens=False,
                         return_tensors="pt").input_ids[0]
            t = t[:64]
            if t.numel() > 0:
                out.append(t.to(self.dev))
        assert out, "instance has no non-empty gold answer"
        return out

    def _prompt(self, prep, kept):
        kept_t = torch.as_tensor(sorted(int(t) for t in kept), device=self.dev)
        visk = prep["vis"].index_select(0, kept_t)
        return torch.cat([prep["prompt_prefix"], visk, prep["suffix"]], dim=0)

    @torch.no_grad()
    def nll(self, prep, kept, ans_ids):
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
        del out, logits
        return float(np.mean(per_gold)), per_gold, P

    @torch.no_grad()
    def nll_multi(self, prep, kept_sets, ans_id):
        """NLL of ONE gold under MANY retained sets, in a single forward.

        Every deletion of an r-token set produces a sequence of the same length
        (`kept` differs, its size does not), so the whole batch can be stacked.
        The prompt skeleton is shared and only the vision block varies, which is
        what makes the batching exact rather than an approximation.
        """
        P = None
        prompts = []
        for kept in kept_sets:
            pr = self._prompt(prep, kept)
            P = pr.shape[0] if P is None else P
            assert pr.shape[0] == P, "ragged deletion batch"
            prompts.append(pr)
        prompt = torch.stack(prompts, dim=0)                 # (B, P, D)
        B = prompt.shape[0]
        n = ans_id.numel()
        full = torch.zeros(B, P + n, prompt.shape[2], dtype=prompt.dtype,
                           device=prompt.device)
        full[:, :P] = prompt
        full[:, P:] = self.emb(ans_id.unsqueeze(0))[0]
        am = torch.ones(B, P + n, dtype=torch.long, device=self.dev)
        pos1 = torch.arange(P + n, device=self.dev).view(1, 1, -1) \
            .expand(3, B, -1)
        out = self.model(inputs_embeds=full, attention_mask=am,
                         position_ids=pos1.contiguous(), use_cache=False,
                         return_dict=True)
        logits = out.logits[:, P - 1:P - 1 + n, :].float()
        ce = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1),
            ans_id.unsqueeze(0).expand(B, -1), reduction="none")
        vals = ce.mean(dim=1).tolist()
        del out, logits, full, prompt
        return vals, P

    @torch.no_grad()
    def generate(self, prep, kept, max_new=GEN_CAP):
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
        return ids


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_tokens(X: torch.Tensor, s0: torch.Tensor, rng) -> dict:
    """Stratified sample of S0, as a dict stratum -> list of token indices.

    Deterministic given `rng`: the order within a stratum is by score, so the
    record is readable, and ties resolve by index.
    """
    s0l = s0.detach().cpu().numpy()
    out = {}
    used = set()

    def take(order, k):
        picked = []
        for t in order:
            t = int(t)
            if t in used:
                continue
            picked.append(t)
            used.add(t)
            if len(picked) == k:
                break
        return picked

    for name, k in STRATA:
        if name == "rand":
            pick = rng.choice(s0l, size=min(k, s0l.size), replace=False)
            pick = [int(t) for t in pick]
            used.update(pick)
            out[name] = sorted(pick)
            continue
        if name == "maxred":
            col = X[:, RED_COL]
            order = s0l[np.argsort(-col[s0l].detach().cpu().numpy(), kind="stable")]
        elif name == "minred":
            col = X[:, RED_COL]
            order = s0l[np.argsort(col[s0l].detach().cpu().numpy(), kind="stable")]
        elif name == "highimp":
            col = X[:, IMP_COL]
            order = s0l[np.argsort(-col[s0l].detach().cpu().numpy(), kind="stable")]
        elif name == "lowimp":
            col = X[:, IMP_COL]
            order = s0l[np.argsort(col[s0l].detach().cpu().numpy(), kind="stable")]
        elif name == "recon":
            col = X[:, RECON_COL]
            order = s0l[np.argsort(col[s0l].detach().cpu().numpy(), kind="stable")]
        else:
            raise KeyError(name)
        out[name] = sorted(take(order, k))
    return out


def group_tokens(X: torch.Tensor, s0: torch.Tensor, k: int, rule: str,
                 rng) -> list:
    """The k tokens a trimming rule would delete -- the group control's set."""
    s0l = s0.detach().cpu().numpy()
    if rule == "random":
        pick = rng.choice(s0l, size=k, replace=False)
        return sorted(int(t) for t in pick)
    col = {"maxred": RED_COL, "lowimp": IMP_COL, "recon": RECON_COL}[rule]
    vals = X[:, col][s0l].detach().cpu().numpy()
    order = np.argsort(-vals if rule == "maxred" else vals, kind="stable")
    return sorted(int(s0l[i]) for i in order[:k])


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------
def measure(h, item, bank, gen: bool):
    i = item["bank_row"]
    prep = h.prep(item["msg"], item["ds"], vis_row(bank, i))
    import s3a_common as C3
    ans = h.answer_ids(C3.golds_of(item["row"]))
    s0 = np.sort(np.asarray(bank["s0"][i], dtype=np.int64))
    # The features come from the bank, which was built by calling the SAME
    # function the live pruner calls on the SAME tensors -- the sampling strata
    # and the probe's inputs are therefore the same columns by construction,
    # and the gate below re-checks that on a sample of instances.
    X = item["_X"]

    L_ref, per_gold, P = h.nll(prep, s0.tolist(), ans)
    # ---- the noise floor, measured two ways because they are not the same ----
    # `floor_max` is the S3-A quantity: how far a single gold's NLL moves when
    # the batch composition changes.  It is large (~0.1 nats) and it is NOT the
    # resolvability of d_i, because d_i is a difference of two means computed
    # with the SAME batching, so that perturbation is common-mode and cancels.
    # `floor_mean` is the resolvability that actually applies: how far the
    # mean-over-golds moves under the same benign re-batching.  Both are
    # recorded; the thresholds are set from `floor_mean`, and `floor_max` is
    # kept so the gap between them is visible rather than assumed.
    singles = [h.nll(prep, s0.tolist(), [t])[0] for t in ans]
    floor_max = float(np.max(np.abs(np.array(per_gold) - np.array(singles))))
    floor_mean = float(abs(L_ref - float(np.mean(singles))))
    # A third, strictly weaker check: two independent recomputations of the
    # SAME mean.  If this is non-zero the harness is not deterministic at all.
    L_rep, per_rep, _ = h.nll(prep, s0.tolist(), ans)
    floor_rep = float(abs(L_rep - L_ref))

    rng = np.random.default_rng(SEED + i)
    samp = sample_tokens(X, torch.as_tensor(s0), rng)

    recs = []
    for name, toks in samp.items():
        for t in toks:
            kept = [int(x) for x in s0 if int(x) != t]
            L, _, _ = h.nll(prep, kept, ans)
            recs.append(dict(stratum=name, token=int(t), d=float(L - L_ref),
                             L=float(L),
                             red=float(X[t, RED_COL]), imp=float(X[t, IMP_COL]),
                             recon=float(X[t, RECON_COL])))

    groups = []
    for k in GROUP_KS:
        for rule in GROUP_RULES:
            G = group_tokens(X, torch.as_tensor(s0), k, rule, rng)
            kept = [int(x) for x in s0 if int(x) not in set(G)]
            L, _, _ = h.nll(prep, kept, ans)
            g = dict(k=k, rule=rule, tokens=G, D=float(L - L_ref), L=float(L))
            if gen:
                ids = h.generate(prep, kept)
                pred = h.vlm._post_process_response(
                    h.tok.decode(ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False))
                g["pred"] = pred
                g["hit"] = float(per_sample_hits(item["ds"], [item["row"]],
                                                 [pred])[0])
            groups.append(g)

    return dict(key=item["key"], ds=item["ds"], split=item["split"],
                n_golds=len(ans), n_prompt_tokens=int(P),
                L_ref=float(L_ref), floor=float(floor_mean),
                floor_mean=floor_mean, floor_max=floor_max,
                floor_rep=floor_rep, per_gold=per_gold,
                s0=s0.tolist(), singles=recs, groups=groups,
                sample_plan={k: v for k, v in samp.items()})


# ---------------------------------------------------------------------------
def measure_groups_only(h, item, bank, prior):
    """Generation for the group arms only, reusing the NLL pass's group sets.

    The brief's §2B asks for a real generation per group arm beside the NLL
    delta.  Re-running the whole 54-deletion sweep to get six generations would
    cost twenty minutes of GPU for nothing, so this re-derives the group sets
    from the same features and the same rng seed the NLL pass used and
    generates only those.  The sets are asserted equal to the recorded ones, so
    the two passes cannot describe different groups.
    """
    i = item["bank_row"]
    prep = h.prep(item["msg"], item["ds"], vis_row(bank, i))
    X = item["_X"]
    s0 = np.sort(np.asarray(bank["s0"][i], dtype=np.int64))
    rng = np.random.default_rng(SEED + i)
    # Replay the sampler's rng consumption exactly: sample_tokens draws the
    # `rand` stratum before any group is formed.
    sample_tokens(X, torch.as_tensor(s0), rng)
    out = []
    for k in GROUP_KS:
        for rule in GROUP_RULES:
            G = group_tokens(X, torch.as_tensor(s0), k, rule, rng)
            rec = next((g for g in prior["groups"]
                        if g["k"] == k and g["rule"] == rule), None)
            assert rec is not None, (k, rule)
            assert set(G) == set(int(x) for x in rec["tokens"]), \
                f"group replay drifted for k={k} {rule}"
            kept = [int(x) for x in s0 if int(x) not in set(G)]
            ids = h.generate(prep, kept)
            pred = h.vlm._post_process_response(
                h.tok.decode(ids, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False))
            # MERGE into the record the NLL pass wrote rather than replacing it.
            # The first version returned a fresh dict, so running this pass
            # silently dropped every group's `D` -- the NLL delta the whole
            # §2B control rests on -- while leaving the file looking complete.
            out.append(dict(rec, pred=pred,
                            hit=float(per_sample_hits(item["ds"], [item["row"]],
                                                      [pred])[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["fit", "val"])
    ap.add_argument("--gen", action="store_true",
                    help="also generate for the group-deletion arms")
    ap.add_argument("--gen-only", action="store_true",
                    help="generation for the group arms only, reusing an "
                         "existing NLL pass (needs --tag pointing at it)")
    ap.add_argument("--tag", default="m5_teacher")
    args = ap.parse_args()

    bank = load_m5_bank()
    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=GEN_CAP)
    model.model.eval()
    torch.set_grad_enabled(False)

    items = bank_items(model, bank, splits=tuple(args.splits))
    if args.limit:
        items = items[:args.limit]
    for it in items:
        it["_X"] = torch.from_numpy(bank["X"][it["bank_row"]].astype(np.float32))

    # ---- gate: the bank's cached state is what a live pass produces --------
    # The teacher reads the bank's vision features and the bank's feature
    # matrix instead of re-deriving them, so this re-derives both on a sample
    # and refuses to measure if they disagree.  It is the M4 gate
    # `live_s0_equals_bank_s0` widened to cover the features too.
    eng = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                       selector=BASE_SELECTOR, tag="M5BASE"))
    from instrumented import attach_pruner
    pruner = attach_pruner(model, selector=BASE_SELECTOR, capture=False)
    pruner.visual_token_num = BUDGET
    pruner.sim_mode = "rebound"
    pruner.keep_gpu = True
    eng.pruner = model.pruner = pruner
    probe = items[::max(1, len(items) // 12)][:12]
    bad = []
    for it in probe:
        i = it["bank_row"]
        prep = eng.prepare(it["msg"], it["ds"])
        live_vis = prep["vis"]
        stored_vis = vis_row(bank, i, live_vis.device)
        text_llm, text_seq = eng._instruction_embeds(prep)
        pruner(live_vis, text_llm, text_seq, prep["gthw"])
        g = pruner.last_gpu
        s0 = torch.sort(g["select_idx"][0].to(torch.long)).values
        X_live = safe_features(g, g["image_features"].float(), s0).detach()
        vis_ok = bool(torch.equal(live_vis, stored_vis))
        s0_ok = bool(np.array_equal(s0.cpu().numpy(),
                                    np.sort(np.asarray(bank["s0"][i],
                                                       dtype=np.int64))))
        x_mad = float((X_live.cpu() - it["_X"]).abs().max())
        if not (vis_ok and s0_ok and x_mad == 0.0):
            bad.append(dict(key=it["key"], vis=vis_ok, s0=s0_ok, x_mad=x_mad))
        pruner.last_gpu = {}
        del prep, live_vis, stored_vis, X_live, g
        torch.cuda.empty_cache()
    print(f"[gate] bank vs live on {len(probe)} instances: "
          f"{len(probe)-len(bad)}/{len(probe)} exact (vis, s0, features)")
    if bad:
        raise RuntimeError(f"bank disagrees with a live pass: {bad[:3]}")
    del eng, pruner
    torch.cuda.empty_cache()

    h = TeacherHarness(model)

    if args.gen_only:
        src = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
        prior = json.load(open(src))
        items = [it for it in items if it["key"] in prior["meas"]]
        t0 = time.time()
        for n, it in enumerate(items):
            try:
                prior["meas"][it["key"]]["groups"] = measure_groups_only(
                    h, it, bank, prior["meas"][it["key"]])
            except Exception:
                traceback.print_exc()
                print(f"[skip] {it['key']}")
                continue
            if (n + 1) % 10 == 0 or n == len(items) - 1:
                print(f"[gen {n+1}/{len(items)}] {time.time()-t0:.0f}s",
                      flush=True)
            dump_json(f"{args.tag}.json", prior)
        hits = [g["hit"] for k in prior["meas"].values() for g in k["groups"]
                if g.get("hit") is not None]
        print(f"[saved] {src}  {len(items)} instances, "
              f"{len(hits)} group generations, mean hit {np.mean(hits):.3f}")
        return

    out_path = os.path.join(OUTPUT_DIR, f"{args.tag}.json")
    results = dict(config=dict(splits=args.splits, strata=STRATA,
                               group_ks=GROUP_KS, group_rules=GROUP_RULES,
                               seed=SEED, gen=bool(args.gen),
                               features=list(FEATURES)),
                   done_keys=[], meas={})
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if prev.get("config") != results["config"]:
            raise RuntimeError("existing teacher file was produced under a "
                               "different config -- refuse to mix")
        results["done_keys"] = prev.get("done_keys", [])
        results["meas"] = prev.get("meas", {})
        print(f"[resume] {len(results['done_keys'])} instances already done")

    todo = [it for it in items if it["key"] not in results["done_keys"]]
    t0 = time.time()
    n_fail = 0
    for n, it in enumerate(todo):
        try:
            rec = measure(h, it, bank, args.gen)
        except Exception:
            traceback.print_exc()
            n_fail += 1
            print(f"[skip] {it['key']}")
            continue
        results["meas"][it["key"]] = rec
        results["done_keys"].append(it["key"])
        if (n + 1) % 5 == 0 or n == len(todo) - 1:
            d = np.array([s["d"] for s in rec["singles"]])
            print(f"[{n+1}/{len(todo)}] {rec['key']:22s} "
                  f"L_ref={rec['L_ref']:.3f} floor={rec['floor']:.3f} "
                  f"d p10/p50/p90 = {np.percentile(d,10):+.3f}/"
                  f"{np.percentile(d,50):+.3f}/{np.percentile(d,90):+.3f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        dump_json(f"{args.tag}.json", results)
    print(f"[saved] {out_path}  ({time.time()-t0:.0f}s, {n_fail} skipped)")


if __name__ == "__main__":
    main()
