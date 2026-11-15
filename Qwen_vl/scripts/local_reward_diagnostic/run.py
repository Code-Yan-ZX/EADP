"""Resumable, inference-only official-head action-pool diagnostic."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "Qwen_vl/outputs/local_reward_diagnostic"
MODEL = Path("/media/disk2/YZX/research/model/FireRed-OCR")
HEAD = ROOT / "Qwen_vl/third_party/LayoutLite/checkpoints/firered_600.pt"
QA_SUFFIX = "\nAnswer with the answer only, without explanation."
SEEDS = [2026110801, 2026110802, 2026110803, 2026110804]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "detach"):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(type(obj).__name__)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=plain)+"\n")
    tmp.replace(path)


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=plain)+"\n")
        f.flush()
        os.fsync(f.fileno())


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()] if Path(path).exists() else []


def qa_exact(text, answers):
    from reward_metrics import normalize
    value = normalize(text).strip("`*\"'").rstrip(",:;")
    return any(value == normalize(a).strip("`*\"'").rstrip(",:;") for a in answers)


def smoke(runner, args):
    """Outside-panel first three old binding source pages, fixed in advance."""
    source = json.loads((ROOT/"Qwen_vl/outputs/binding_oracle/data/combined_manifest.json").read_text())
    rows, first_gate = [], None
    for i, item in enumerate(source["records"][:3]):
        page = runner.encode(item["source_image"])
        if i == 0:
            first_gate = runner.identity_gate(page)
            atomic_json(OUT/"identity_gate.json", first_gate)
            print("IDENTITY_GATE", json.dumps(first_gate, default=plain), flush=True)
            assert first_gate["pass_all"], "Native all-token identity gate failed"
        for field in item["fields"][:2]:
            prompt = field["question"]+QA_SUFFIX
            qa_page = runner.re_prompt(page, prompt)
            text, info = runner.generate(qa_page, None, max_new_tokens=32)
            rows.append(dict(image=item["source_image"], question_id=field["question_id"],
                             question=field["question"], answers=field["answers_original"],
                             text=text, info=info, exact=qa_exact(text, field["answers_original"])))
            print("QA_SMOKE", len(rows), repr(text[:150]), rows[-1]["exact"], flush=True)
        del page
        gc.collect()
    good = sum(r["exact"] and not r["info"]["truncated"] for r in rows)
    decision = dict(rows=rows, correct=good, total=len(rows),
                    run_secondary_qa=good >= 4 and len(rows) == 6 and not any(r["info"]["truncated"] for r in rows),
                    threshold="at least 4/6 exact, nontruncated full-token QA on fixed outside-panel pages",
                    prompt_suffix=QA_SUFFIX, max_new_tokens=32,
                    inference_only=True, identity_gate=first_gate)
    atomic_json(OUT/"smoke.json", decision)
    print("SMOKE_COMPLETE", good, "/", len(rows), "secondary_qa", decision["run_secondary_qa"], flush=True)


def panel(runner, args):
    manifest_path = OUT/"data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"], "Data manifest must be completely frozen before panel generation"
    items = [r for r in manifest["items"] if r["role"] == "primary"]
    assert len(items) == 32 and len({r["parent_id"] for r in items}) == 32
    # Reserves are not consumed in this pilot; never replace compressed failures.
    smoke_result = json.loads((OUT/"smoke.json").read_text())
    assert smoke_result["identity_gate"]["pass_all"], "Identity gate must pass before panel"
    assert json.loads((OUT/"batch_gates.json").read_text())["pass_all"], "Real stock-generation and batching gate must pass"
    run_qa = smoke_result["run_secondary_qa"]
    outputs = OUT/"predictions.jsonl"
    existing = load_records(outputs)
    done = {(r["doc_id"], r["kind"], r["arm"], r.get("question_id")) for r in existing}
    runpath = OUT/"run_manifest.json"
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in [
        manifest_path, HERE/"run.py", HERE/"layoutlite_runner.py", HERE/"batch_runner.py", HERE/"reward_metrics.py",
        ROOT/"docs/local_reward_diagnostic_prereg.md", HEAD]}
    if runpath.exists():
        previous = json.loads(runpath.read_text())
        assert previous["frozen_hashes"] == fingerprints, "Frozen source/data/checkpoint changed on resume"
    else:
        import torch, transformers, Levenshtein
        atomic_json(runpath, dict(started_unix=time.time(), frozen_hashes=fingerprints,
            model_dir=str(MODEL), model_revision="7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50",
            model_safetensors_sha256=sha(MODEL/"model.safetensors"),
            layoutlite_commit="7e398a2ff4f23f0cb836e36e0e81d21d379d8fc2",
            torch=torch.__version__, transformers=transformers.__version__, Levenshtein=Levenshtein.__version__,
            mask_seeds=SEEDS, mask_temperature=1, keep_ratio=0.5,
            ocr_prompt=runner.ocr_prompt, generation="greedy, native EOS; full serial, five compressed actions same-page batch5",
            max_new_tokens=args.max_tokens, qa_max_new_tokens=32, secondary_qa=run_qa,
            page_ids=[r["doc_id"] for r in items], planned_pages=len(items), planned_ocr_outputs=32*6,
            inference_only=True, reserve_policy="fixed primary32; no replacements, reserve16 unused",
            scope="official head/backbone; exact-K diagnostic actions, not official clustered inference reproduction"))
    started = time.time()
    for at, item in enumerate(items):
        doc = item["doc_id"]
        arms_done = sum((doc, "ocr", arm, None) in done for arm in ["full", "topk", *[f"gumbel_{s}" for s in SEEDS]])
        if arms_done == 6 and (not run_qa or all((doc,"qa",arm,q["question_id"]) in done
                for arm in ["full","topk", *[f"gumbel_{s}" for s in SEEDS]] for q in item["qa"])):
            print("SKIP_COMPLETE", at+1, doc, flush=True)
            continue
        page = runner.encode(item["image_path"])
        from layoutlite_runner import diagnostic_masks
        masks = diagnostic_masks(page["scores"], keep_ratio=.5, seeds=SEEDS)
        # Normalize descriptive runner names into the frozen order.
        names = ["topk", *[f"gumbel_{s}" for s in SEEDS]]
        assert len(masks) == 5
        choices = [("full", None)] + [(name, mask["keep_idx"]) for name, mask in zip(names, masks)]
        audit = dict(doc_id=doc, n_vis=len(page["scores"]),
                     official_scores=page["scores"].detach().cpu().tolist(),
                     masks=[dict(name=name, **{k:v for k,v in mask.items() if k != "name"}) for name,mask in zip(names,masks)])
        atomic_json(OUT/"masks"/(doc+".json"), audit)
        compressed_outputs=None
        for arm, keep in choices:
            key = (doc, "ocr", arm, None)
            if key not in done:
                before = time.time()
                if arm == "full":
                    text, info = runner.generate(page, keep, max_new_tokens=args.max_tokens)
                else:
                    if compressed_outputs is None:
                        from batch_runner import generate_many
                        compressed_outputs=dict(zip(names,generate_many(runner,page,[m["keep_idx"] for m in masks],max_new_tokens=args.max_tokens)))
                    text,info=compressed_outputs[arm]
                append(outputs, dict(doc_id=doc, parent_id=item["parent_id"], kind="ocr", arm=arm,
                                     text=text, info=info, elapsed_seconds=info["elapsed_seconds"],
                                     elapsed_scope=info.get("elapsed_scope","one serial output wall time")))
                done.add(key)
                print("OCR", at+1, "/32", doc, arm, "N", info["n_vis"],
                      "tokens",info.get("generated_tokens"), "truncated",info["truncated"],
                      "sec",round(time.time()-before,2), flush=True)
            if run_qa:
                for question in item["qa"]:
                    qid = question["question_id"]
                    key = (doc,"qa",arm,qid)
                    if key in done:
                        continue
                    qpage = runner.re_prompt(page, question["question"]+QA_SUFFIX)
                    text, info = runner.generate(qpage, keep, max_new_tokens=32)
                    append(outputs, dict(doc_id=doc,parent_id=item["parent_id"],kind="qa",arm=arm,
                        question_id=qid, text=text, info=info, exact=qa_exact(text,question["answers"])))
                    done.add(key)
        del page, masks
        gc.collect()
        print("PAGE_COMPLETE", at+1,"/32", "elapsed_sec",round(time.time()-started,1),flush=True)
    print("PANEL_COMPLETE", "records",len(load_records(outputs)),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from layoutlite_runner import LayoutLiteRunner
    runner = LayoutLiteRunner(model_dir=str(MODEL),head_path=str(HEAD),device="cuda")
    if args.smoke:
        smoke(runner,args)
    else:
        panel(runner,args)


if __name__ == "__main__":
    main()
