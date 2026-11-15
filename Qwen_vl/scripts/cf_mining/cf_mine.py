"""Native-model semantic-difference pilot: generation-based screening.

Never counts a changed output as a correct counterfactual answer. Each image
gets its own vision encoding and B2 selection. Resume from persisted records.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time
from decimal import Decimal, InvalidOperation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "discovery"))
import common
import torch
from model.native_qwen3 import NativeEngine

OUT = Path(common.QWEN_ROOT) / "outputs" / "cf_mining"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def normalize(value):
    s = str(value).strip().lower().strip("'\"`")
    s = re.sub(r"[\s,$£€]", "", s)
    s = s.replace("−", "-")
    # Numeric canonicalization changes representation, never the value.
    suffix = "%" if s.endswith("%") else ""
    z = s[:-1] if suffix else s
    try:
        return str(Decimal(z).normalize()) + suffix
    except InvalidOperation:
        return s


def correct(pred, answer):
    return normalize(pred) == normalize(answer)


def message(image, question):
    return [dict(type="image", value=str(image)),
            dict(type="text", value=question + "\nAnswer with the value only.")]


@torch.no_grad()
def run_image(eng, image, question, answer, cache_path, max_tokens=24):
    msg = message(image, question)
    prep = eng.prepare(msg, "DocVQA_VAL")
    V, DS = eng.encode(prep)
    tm, ts = eng.instruction_embeds(msg, "DocVQA_VAL")
    ctx = dict(engine=eng, prep=prep, V=V, DS=DS,
               text_mean=tm, text_seq=ts)
    keep = eng.select("b2", 256, ctx)
    assert V.shape[0] == 1024 and keep.numel() == 256
    predictions = {}
    for label, indices in (("teacher", None), ("student", keep)):
        st = eng.prefill(prep, V, DS, indices)
        gen, pred = eng.decode(st, max_tokens)
        predictions[label] = pred
        predictions[label + "_truncated"] = len(gen) >= max_tokens
        del st
    rec = dict(ids=prep["ids"].cpu(), pos3d=eng.positions_full(prep)[0].cpu(),
               V=V.cpu(), DS=[x.cpu() for x in DS], keep=keep.cpu(),
               gthw=prep["gthw"].cpu(), img_start=prep["img_start"],
               img_end=prep["img_end"], n_vis=prep["n_vis"], answer=answer,
               image=str(image), question=question)
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, cache_path)
    del rec, prep, V, DS, ctx, tm, ts
    return predictions


def summarize(records):
    stats = {}
    for split in sorted({r["split"] for r in records}):
        rr = [r for r in records if r["split"] == split]
        vv = [r for r in rr if r["teacher_valid"]]
        stats[split] = dict(attempted=len(rr), teacher_valid=len(vv),
                            blind=sum(r["blind"] for r in vv),
                            strict_collapse=sum(r["collapse"] for r in vv),
                            student_both_correct=sum(r["student_both_correct"] for r in vv))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(OUT / "manifest.json"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--source-limit", type=int, default=32)
    ap.add_argument("--invariant-limit", type=int, default=8)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    path = OUT / "mining.json"
    result = json.loads(path.read_text()) if path.exists() else dict(
        config=vars(args), records=[], sources=[], started=time.time(),
        model=common.BASELINE_MODEL, selector="b2", token_budget=256,
        resolution=1024, native_deepstack=True, native_mrope=True,
        seed=20261108, stage="running")
    done = {r["pair_id"] for r in result["records"]}
    records = manifest["records"][:args.limit]
    model = common.load_model(common.BASELINE_MODEL, max_new_tokens=args.max_tokens)
    eng = NativeEngine(model)
    t0 = time.time()
    for i, item in enumerate(records):
        if item["pair_id"] in done:
            continue
        p = item["pair_id"]
        row = dict(item)
        row["split"] = "fit" if item["split"] == "train" else item["split"]
        for side in ("original", "edited"):
            cache = OUT / "cache" / f"{p}_{side}.pt"
            pred = run_image(eng, item[side + "_image"], item["question"],
                             item[side + "_answer"], cache, args.max_tokens)
            row["cache_" + side] = str(cache)
            for key, value in pred.items():
                row[side + "_" + key + ("" if key.endswith("truncated") else "_prediction")] = value
        row["teacher_valid"] = all(correct(row[s + "_teacher_prediction"],
                                                  row[s + "_answer"])
                                          for s in ("original", "edited"))
        row["student_both_correct"] = all(correct(row[s + "_student_prediction"],
                                                       row[s + "_answer"])
                                               for s in ("original", "edited"))
        row["blind"] = row["teacher_valid"] and not row["student_both_correct"]
        row["collapse"] = (row["blind"] and
                           normalize(row["original_student_prediction"]) ==
                           normalize(row["edited_student_prediction"]))
        result["records"].append(row)
        result["summary"] = summarize(result["records"])
        result["updated"] = time.time()
        atomic_json(path, result)
        print(f"[mine {i+1}/{len(records)}] {p}: teacher_valid={row['teacher_valid']} "
              f"blind={row['blind']} collapse={row['collapse']} "
              f"answers={item['original_answer']}->{item['edited_answer']} "
              f"teacher={row['original_teacher_prediction']!r}->{row['edited_teacher_prediction']!r} "
              f"student={row['original_student_prediction']!r}->{row['edited_student_prediction']!r} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
    # Untouched real pages: held out by source image, never used in mining fit.
    src_done = {r["pair_id"] for r in result["sources"]}
    candidates = [r for r in records if r["split"] not in ("fit", "train")][:args.source_limit]
    for item in candidates:
        if item["pair_id"] in src_done:
            continue
        cache = OUT / "cache" / f"{item['pair_id']}_source.pt"
        pp = run_image(eng, item["source_image"], item["question"],
                       item["original_answer"], cache, args.max_tokens)
        row = dict(pair_id=item["pair_id"], split=item["split"],
                   answer=item["original_answer"], cache=str(cache), **pp)
        row["teacher_correct"] = correct(pp["teacher"], row["answer"])
        row["student_correct"] = correct(pp["student"], row["answer"])
        result["sources"].append(row)
        atomic_json(path, result)
    result["summary"] = summarize(result["records"])
    # Unrelated-field candidates are accepted only after BOTH full-teacher
    # answers match an existing label. A sibling question is not automatically
    # invariant merely because its string omits the edited value.
    result.setdefault("invariants", [])
    inv_done = {r["pair_id"] for r in result["invariants"]}
    inv_candidates = [r for r in records if r["split"] not in ("fit", "train")
                      and r.get("invariant_questions")][:args.invariant_limit]
    for item in inv_candidates:
        if item["pair_id"] in inv_done:
            continue
        q = item["invariant_questions"][0]
        row = dict(pair_id=item["pair_id"], split=item["split"],
                   answer=q["answers"][0], answers=q["answers"], question=q["question"])
        for side in ("original", "edited"):
            cache = OUT / "cache" / f"{item['pair_id']}_invariant_{side}.pt"
            pred = run_image(eng, item[side + "_image"], q["question"],
                             q["answers"][0], cache, args.max_tokens)
            row["cache_" + side] = str(cache)
            row[side + "_teacher_prediction"] = pred["teacher"]
            row[side + "_student_prediction"] = pred["student"]
        for who in ("teacher", "student"):
            row[who + "_both_correct"] = all(
                any(correct(row[side + "_" + who + "_prediction"], a)
                    for a in row["answers"]) for side in ("original", "edited"))
        result["invariants"].append(row)
        atomic_json(path, result)
    result["stage"] = "complete"
    result["completed"] = time.time()
    result["wall_seconds_this_run"] = time.time() - t0
    atomic_json(path, result)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
