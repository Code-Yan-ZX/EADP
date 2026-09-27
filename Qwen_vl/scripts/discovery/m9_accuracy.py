"""
M9 Phase 2/3 -- the real deferred-pruning engine and the bank150 screening.

Engine (one code path for every arm)
------------------------------------
    1. vision tower -> 1024 merged visual embeddings          (unchanged)
    2. assemble the EARLY sequence: prefix text + vis[S_early] + suffix text,
       S_early = sorted(B2-256 u reserve-32) = 288 visual tokens, positions
       RENUMBER (arange) -- the incumbent prellm convention
    3. decoder layers 0..L-1 over the early sequence (~319 rows)
       (ATTN/CMC arms additionally capture text->boundary scores at layer L-1;
        read-only hooks, forward unchanged)
    4. adjudicate the boundary = tail-r u reserve-32: keep exactly r members
       by the arm's rule; S_final = order[:256-r] u kept, |S_final| = 256
    5. compact the layer 0..L-1 KV cache and the hidden states to S_final
    6. decoder layers L..35 over the kept 291 rows, greedy generation

Arms (bank150 = the frozen test split, 150 instances):
    B2       incumbent pre-LLM EADP block8@256, no early pass (identity arm,
             gated against the stored M2 predictions 150/150)
    DB2      deferred path, adjudication keeps the whole B2 tail -> final set
             is exactly S0.  THE matched baseline: identical 288-token early
             exposure, identical schedule; the only difference from the swap
             arms is which r boundary members survive.
    DB2L0    the deferred path with L=0: zero layers before compaction, so the
             sequence is bit-for-bit the incumbent's.  ENGINE GATE: must
             reproduce B2's predictions exactly; any diff is an engine bug.
    RNDr     kept = r random boundary members (seeded per instance)
    COSr     kept = top-r by ascending cos_s0c (the corrected M6/M8 baseline)
    ATTNr    kept = top-r by att_mean, A2_last4 question aggregation, layer L-1
    CMCr     kept = top-r by cmc_mn,  A2_last4 question aggregation, layer L-1
    ORCr     kept = top-r by the P1-G2 teacher score (offline label; ceiling)

Gates (abort, never record a low macro):
    G-B2     B2 predictions == stored M2 B2 predictions 150/150 (+ hits)
    G-L0     DB2L0 predictions == B2 predictions 150/150
    G-SET    |S_final| = 256, duplicate-free, core u kept disjoint,
             kept subset of boundary
    G-NOERR  a crashed instance is never persisted as a score

Usage
    python scripts/discovery/m9_accuracy.py --arms B2 DB2 DB2L0 CMC8 ...
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
from m2_accuracy import MAX_NEW                                      # noqa: E402
from m2_gdep import (MODE_GDEP, MODE_PRELLM, POLICY_RENUMBER,         # noqa: E402
                     GDEPConfig, GDEPEngine, dump_json)
from m5_common import BUDGET, BASE_SELECTOR, bank_items              # noqa: E402
from m9_capture import EarlyLayerCapture                             # noqa: E402
from m9_common import (load_banks, TAIL_MAX, POOL)                    # noqa: E402
from scoring import per_sample_hits                                  # noqa: E402
from m6_common import DS_ORDER                                       # noqa: E402

PANEL = "bank"                    # the frozen test 150
L_PRIMARY = 4
RS = (4, 8)
SIGNAL_ARMS = ("ATTN", "CMC")     # arms that capture at layer L-1


class DeferredEngine(GDEPEngine):
    """B2 + reserve32 through the first L layers, then adjudicate to 256."""

    def __init__(self, model, L: int):
        super().__init__(model, GDEPConfig(mode=MODE_GDEP, layer=L,
                                           budget=BUDGET,
                                           pos_policy=POLICY_RENUMBER,
                                           tag="M9"))
        assert L >= 0

    # ------------------------------------------------------------ assembly --
    def _early(self, prep, bnd):
        """Assemble the 288-visual-token early sequence; return tensors and
        the bookkeeping the adjudication needs."""
        prompt, vis, (s, e) = prep["prompt"], prep["vis"], prep["vis_slice"]
        dev = prompt.device
        order = bnd["order"]
        s_early = np.sort(np.concatenate([order[:256], bnd["reserve"]]))
        se = torch.as_tensor(s_early, dtype=torch.long, device=dev)
        hidden = torch.cat([prompt[:, :s], vis[se][None], prompt[:, e:]], dim=1)
        S1 = int(hidden.shape[1])
        bpos = s + np.searchsorted(s_early, bnd["boundary"])
        assert (s_early[bpos - s] == bnd["boundary"]).all()
        q0 = s + s_early.size
        suffix_rows = torch.arange(q0, S1, device=dev)
        ids = prep["inputs"]["input_ids"][0]
        im_end_id = self.vlm.processor.tokenizer.convert_tokens_to_ids(
            "<|im_end|>")
        after = (ids == im_end_id).nonzero(as_tuple=True)[0]
        im_end = int(after[after > e].min())
        nq = im_end - (e + 1)                    # question rows: suffix 1..nq
        assert nq >= 1
        return hidden, dict(S1=S1, s=s, e=e, q0=q0, nq=nq, s_early=s_early,
                            bpos=torch.as_tensor(bpos, dtype=torch.long,
                                                 device=dev),
                            suffix_rows=suffix_rows)

    # ------------------------------------------------------- adjudication --
    def _kept(self, bnd, rule, sig48):
        kind, r = rule["kind"], rule["r"]
        order = bnd["order"]
        if kind == "DB2":
            return np.sort(order[:256])
        tail_r = order[256 - r:256]
        boundary = np.concatenate([tail_r, bnd["reserve"]])
        assert bnd["tail16"][TAIL_MAX - r:].tolist() == tail_r.tolist()
        if kind == "RND":
            rng = np.random.default_rng([rule["seed"], r,
                                         int(bnd["bank_row"])])
            kept = boundary[rng.permutation(boundary.size)[:r]]
        elif kind == "COS":
            kept = boundary[np.argsort(bnd["cos"][boundary],
                                       kind="stable")[:r]]
        elif kind == "ORC":
            kept = boundary[np.argsort(-bnd["g2"][boundary],
                                       kind="stable")[:r]]
        elif kind in SIGNAL_ARMS:
            assert sig48 is not None and sig48.shape == (TAIL_MAX + POOL,)
            sig = np.concatenate([sig48[TAIL_MAX - r:TAIL_MAX], sig48[TAIL_MAX:]])
            kept = boundary[np.argsort(-sig, kind="stable")[:r]]
        else:
            raise KeyError(kind)
        final = np.sort(np.concatenate([order[:256 - r], kept]))
        return final

    # ------------------------------------------------------------- prefill --
    @torch.no_grad()
    def prefill_deferred(self, prep, bnd, rule):
        from transformers.cache_utils import DynamicCache

        dev, L = self.dev, self.cfg.layer
        cache = DynamicCache(config=self.text.config)
        hidden, ek = self._early(prep, bnd)
        cp = torch.arange(ek["S1"], device=dev)
        pos3 = cp.view(1, 1, -1).expand(3, 1, -1)
        pos_emb = self.text.rotary_emb(hidden, pos3)
        sig48 = None
        if rule["kind"] in SIGNAL_ARMS and L > 0:
            with EarlyLayerCapture(self.text, 0, L, ek["suffix_rows"],
                                   ek["bpos"]) as cap:
                for i in range(L):
                    hidden = self.layers[i](
                        hidden, attention_mask=None, position_ids=pos3[0],
                        past_key_values=cache, cache_position=cp,
                        position_embeddings=pos_emb)
            m = cap.layers[L - 1]["att_mean" if rule["kind"] == "ATTN"
                                   else "cmc_mn"]
            sig48 = m[max(1, ek["nq"] - 3):ek["nq"] + 1].mean(0)\
                .numpy().astype(np.float64)
        else:
            for i in range(L):
                hidden = self.layers[i](
                    hidden, attention_mask=None, position_ids=pos3[0],
                    past_key_values=cache, cache_position=cp,
                    position_embeddings=pos_emb)

        final = self._kept(bnd, rule, sig48)
        assert final.size == 256 and np.unique(final).size == 256
        assert (np.diff(final) > 0).all(), "final set must be ascending"
        s, S1, q0 = ek["s"], ek["S1"], ek["q0"]
        # positions of the kept visual tokens WITHIN the early sequence
        fpos = s + np.searchsorted(ek["s_early"], final)
        assert (ek["s_early"][fpos - s] == final).all()
        keep = np.concatenate([np.arange(s), fpos, np.arange(q0, S1)])
        keep_t = torch.as_tensor(keep, dtype=torch.long, device=dev)
        assert keep_t.numel() == S1 - POOL, "keep size != early_len - 32"
        # compact the early layers' KV cache and the hidden states to S_final
        for i in range(L):
            lay = cache.layers[i]
            lay.keys = lay.keys.index_select(2, keep_t)
            lay.values = lay.values.index_select(2, keep_t)
        hidden = hidden.index_select(1, keep_t)
        n = int(hidden.shape[1])
        pos1 = torch.arange(n, device=dev)
        pos3n = pos1.view(1, 1, -1).expand(3, 1, -1)
        pos_emb = self.text.rotary_emb(hidden, pos3n)
        cpn = torch.arange(n, device=dev)
        for i in range(L, self.n_layers):
            hidden = self.layers[i](
                hidden, attention_mask=None, position_ids=pos1,
                past_key_values=cache, cache_position=cpn,
                position_embeddings=pos_emb)
        hidden = self.text.norm(hidden)
        logits = self.model.lm_head(hidden[:, -1:, :])[:, -1, :]
        info = dict(kind=rule["kind"], r=rule["r"], L=L,
                    n_vis_early=int(q0 - s), n_kept=256, context_len=n,
                    nq=ek["nq"], final_idx=final.tolist())
        state = dict(cache=cache, logits=logits, pos1=pos1,
                     n_ctx=torch.ones(1, n, dtype=torch.long, device=dev),
                     hidden_len=n)
        return state, info


def make_bnd_data(banks):
    """Per bank-row boundary bookkeeping, precomputed once."""
    out = {}
    for i, key in enumerate(banks["key"]):
        order = banks["order"][i]
        in_s0 = np.zeros(1024, bool)
        in_s0[order[:256]] = True
        dropped = np.flatnonzero(~in_s0)
        reserve = dropped[np.argsort(banks["cos"][i][dropped],
                                     kind="stable")[:POOL]]
        out[str(key)] = dict(order=order, cos=banks["cos"][i],
                             g2=banks["g2"][i], reserve=reserve,
                             tail16=order[256 - TAIL_MAX:256],
                             boundary=np.concatenate(
                                 [order[256 - TAIL_MAX:256], reserve]),
                             bank_row=i)
    return out


def run_arm(model, eng_b2, items, bnd_data, arm, L, stored_b2=None):
    """One arm over the panel.  Returns the m7-style result record."""
    import zlib
    t0 = time.time()
    preds, meta, n_fail = [], [], 0
    if arm in ("DB2L0", "DB2"):
        kind, r = "DB2", None
    elif arm == "B2":
        kind, r = None, None
    else:
        kind = arm.rstrip("0123456789")
        r = int(arm[len(kind):]) if arm[len(kind):] else None
    assert kind in (None, "DB2", "RND", "COS", "ORC", "ATTN", "CMC"), arm

    if arm == "B2":
        eng = eng_b2
    else:
        eng = DeferredEngine(model, L=0 if arm == "DB2L0" else L)
    try:
        for j, it in enumerate(items):
            try:
                if arm == "B2":
                    out = eng.run(it["msg"], it["ds"], MAX_NEW)
                    info = out["info"]
                else:
                    bnd = bnd_data[it["key"]]
                    rule = dict(kind=kind, r=r if r is not None else 8,
                                seed=int(zlib.crc32(arm.encode())) % (2**31),
                                bank_row=bnd["bank_row"])
                    prep = eng.prepare(it["msg"], it["ds"])
                    state, info = eng.prefill_deferred(prep, bnd, rule)
                    ids = eng.decode(state, MAX_NEW)
                    text = eng.vlm.processor.tokenizer.decode(
                        ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                    out = dict(prediction=eng.vlm._post_process_response(text),
                               n_decode=len(ids), info=info)
                preds.append(out["prediction"])
                meta.append(dict(key=it["key"], ds=it["ds"],
                                 n_decode=out.get("n_decode"),
                                 n_kept=info.get("n_kept"),
                                 final_idx=info.get("final_idx")))
                del out
            except Exception:
                n_fail += 1
                preds.append("")
                meta.append(dict(key=it["key"], ds=it["ds"],
                                 error=traceback.format_exc()))
            if (j + 1) % 25 == 0:
                print(f"  {j+1}/{len(items)}  {time.time()-t0:.0f}s",
                      flush=True)
        if n_fail:
            raise RuntimeError(f"{n_fail}/{len(items)} raised; first:\n"
                               + next(m["error"] for m in meta
                                      if m.get("error")))
        hits = {}
        for ds in DS_ORDER:
            sel = [i for i, it in enumerate(items) if it["ds"] == ds]
            hits[ds] = [float(x) for x in per_sample_hits(
                ds, [items[i]["row"] for i in sel],
                [preds[i] for i in sel])]
        macro = float(np.mean([np.mean(hits[ds]) for ds in DS_ORDER]))
        rec = dict(arm=arm,
                   per_benchmark={ds: dict(acc_pct=float(np.mean(hits[ds]) * 100),
                                           n=len(hits[ds]), hits=hits[ds])
                                  for ds in DS_ORDER},
                   macro_pct=macro * 100, predictions=preds,
                   per_image_meta=meta, wall_seconds=float(time.time() - t0))
        if arm == "B2" and stored_b2 is not None:
            same_p = sum(1 for it, x in zip(items, preds)
                         if stored_b2["pred"].get(it["key"]) == x)
            mine = {it["key"]: h for ds in DS_ORDER
                    for it, h in zip([i for i in items if i["ds"] == ds],
                                     hits[ds])}
            same_h = sum(1 for k, h in mine.items()
                         if abs(h - stored_b2["hit"].get(k, -1)) < 1e-9)
            rec["gate_B2_pass"] = bool(same_p == len(items)
                                       and same_h == len(items))
            rec["gate_B2_predictions"] = int(same_p)
            print(f"  [G-B2] stored-B2 agreement: {same_p}/{len(items)} "
                  f"preds, {same_h}/{len(items)} hits -> "
                  f"{'PASS' if rec['gate_B2_pass'] else 'FAIL'}")
        if arm == "DB2L0" and "B2" in RUN["arms"]:
            b2p = RUN["arms"].get("B2", {}).get("predictions")
            if b2p:
                same = sum(1 for a, b in zip(b2p, preds) if a == b)
                rec["gate_L0_pass"] = bool(same == len(items))
                rec["gate_L0_agreement"] = int(same)
                print(f"  [G-L0] DB2L0 vs B2: {same}/{len(items)} -> "
                      f"{'PASS' if rec['gate_L0_pass'] else 'FAIL'}")
        print(f"  {arm}: TextVQA {rec['per_benchmark']['TextVQA_VAL']['acc_pct']:.3f}"
              f"  DocVQA {rec['per_benchmark']['DocVQA_VAL']['acc_pct']:.3f}"
              f"  OCRBench {rec['per_benchmark']['OCRBench']['acc_pct']:.3f}"
              f"  macro {rec['macro_pct']:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        return rec
    except Exception:
        print(f"  {arm}: ERROR (not persisted as a score)", flush=True)
        return dict(arm=arm, error=traceback.format_exc())
    finally:
        if eng is not eng_b2:
            del eng
        torch.cuda.empty_cache()


RUN = {"arms": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=[
        "B2", "DB2", "DB2L0", "CMC8", "CMC4", "ATTN8", "ATTN4",
        "COS8", "COS4", "RND8", "RND4", "ORC8", "ORC4"])
    ap.add_argument("--L", type=int, default=L_PRIMARY)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test: first N instances only")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    model = common.load_model(eadp_model_name(BUDGET, 0.5, 2.0),
                              max_new_tokens=MAX_NEW)
    model.model.eval()
    torch.set_grad_enabled(False)

    banks = load_banks()
    raw = {k: banks[k] for k in ("key", "ds", "idx", "split")}
    items = bank_items(model, raw, splits=("test",))
    assert len(items) == 150
    if args.limit:
        items = items[:args.limit]
    bnd_data = make_bnd_data(banks)

    eng_b2 = GDEPEngine(model, GDEPConfig(mode=MODE_PRELLM, budget=BUDGET,
                                          selector=BASE_SELECTOR, tag="B2"))
    eng_b2.pruner.visual_token_num = BUDGET

    stored_b2 = None
    p = os.path.join(OUTPUT_DIR, "m2_accuracy.json")
    if os.path.exists(p):
        m2 = json.load(open(p))
        a = m2["arms"].get("B2")
        if a:
            hit = {ds: a["per_benchmark"][ds]["hits"] for ds in DS_ORDER}
            stored_b2 = dict(
                pred=dict(zip(m2["keys"], a["predictions"])),
                hit={k: h for ds in DS_ORDER for k, h in zip(
                    [k for k, d in zip(m2["keys"], m2["ds_order"])
                     if d == ds], hit[ds])})

    out_name = ("m9_accuracy_smoke.json" if args.limit
                else "m9_accuracy_bank.json")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    RUN["arms"] = {}
    RUN["config"] = dict(vars(args), panel=PANEL, L=args.L)
    RUN["keys"] = [i["key"] for i in items]
    RUN["ds_order"] = [i["ds"] for i in items]
    RUN["max_new_tokens"] = MAX_NEW
    if args.resume and os.path.exists(out_path):
        prev = json.load(open(out_path))
        RUN["arms"].update({k: v for k, v in prev.get("arms", {}).items()
                            if "error" not in v})
        print(f"[resume] keeping {sorted(RUN['arms'])}")

    for arm in args.arms:
        if arm in RUN["arms"]:
            print(f"[skip] {arm} already recorded")
            continue
        print(f"\n=== {arm} ===", flush=True)
        rec = run_arm(model, eng_b2, items, bnd_data, arm, args.L, stored_b2)
        RUN["arms"][arm] = rec
        dump_json(out_name, RUN)
    print(f"[saved] {out_name}")


if __name__ == "__main__":
    main()
