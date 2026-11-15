"""Offline mechanism checks on validation pages, never held-out test pages.

These analyses report whether the proposed mechanisms are present. They do
not choose checkpoints/seeds, remove test questions, or stop another experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import atomic_json, load_protocol, sha256
from .build_cache import load_torch, seed_for, sufficient
from . import runtime


def jaccard(a, b):
    return float((a & b).sum()) / max(1, int((a | b).sum()))


def crossed_mask(a, b, seed):
    """Retain common members and mix equal-size exclusive portions at fixed K."""
    if a.dtype != torch.bool or b.dtype != torch.bool or a.shape != b.shape or int(a.sum()) != int(b.sum()):
        raise ValueError("Need two boolean same-budget masks")
    only_a = (a & ~b).nonzero().flatten()
    only_b = (b & ~a).nonzero().flatten()
    if len(only_a) < 2:
        raise ValueError("Too few distinct members to form a nontrivial crossed set")
    gen = torch.Generator().manual_seed(seed)
    cut = len(only_a) // 2
    result = a & b
    result[only_a[torch.randperm(len(only_a), generator=gen)[:cut]]] = True
    result[only_b[torch.randperm(len(only_b), generator=gen)[:len(only_b)-cut]]] = True
    if int(result.sum()) != int(a.sum()) or torch.equal(result, a) or torch.equal(result, b):
        raise AssertionError("Crossed selection is not a distinct fixed-K set")
    return result


def run_diagnostics(run_dir):
    run = Path(run_dir).resolve()
    p = load_protocol(run)
    digest = sha256(run / "protocol.json")
    index = json.loads((run / "cache/index.json").read_text())
    if not index.get("complete") or index["protocol_sha256"] != digest:
        raise RuntimeError("Mechanism checks need a complete frozen training cache")
    manifest = json.loads((run / "data/manifest.json").read_text())
    target = run / "diagnostics/mechanisms.json"
    result = json.loads(target.read_text()) if target.exists() else dict(
        protocol_sha256=digest, cache_index_sha256=sha256(run / "cache/index.json"),
        complete=False, query=[], crossed_sets=[], bank_inventory=[],
        scope="offline validation diagnostics; not deployable QA or a strict upper bound")
    if result["protocol_sha256"] != digest or result["cache_index_sha256"] != sha256(run / "cache/index.json"):
        raise RuntimeError("Mechanism diagnostic inputs changed")
    if result.get("complete"):
        return result
    if not result["bank_inventory"]:
        for page_id, info in index["pages"].items():
            if sha256(info["path"]) != info["sha256"]:
                raise RuntimeError("Training cache modified")
            cache = load_torch(info["path"])
            for q in cache["questions"]:
                good = q["bank_success"].nonzero().flatten().tolist()
                overlaps = [jaccard(q["bank_masks"][a], q["bank_masks"][b])
                            for offset, a in enumerate(good) for b in good[offset+1:]]
                result["bank_inventory"].append(dict(page_id=page_id, role=info["role"],
                    item_id=q["item_id"], successful_masks=len(good),
                    candidate_masks=len(q["bank_masks"]), minimum_success_jaccard=min(overlaps) if overlaps else None))
        atomic_json(target, result)
    eng = runtime.load_engine(p["model_path"])
    done_query = {row["item_id"] for row in result["query"]}
    done_cross = {row["item_id"] for row in result["crossed_sets"]}
    for page in manifest["splits"]["val"]:
        cache = load_torch(index["pages"][page["page_id"]]["path"])
        questions = page["questions"]
        for qi, question in enumerate(questions):
            if question["item_id"] in done_query and question["item_id"] in done_cross:
                continue
            qcache = cache["questions"][qi]
            if qcache["item_id"] != question["item_id"]:
                raise RuntimeError("Question cache order differs from frozen manifest")
            ctx = runtime.prepare(eng, page, question, p["max_question_tokens"])
            if not torch.equal(torch.stack((ctx["V"], *ctx["DS"])).to(torch.bfloat16).cpu(), cache["features"]):
                raise RuntimeError("Diagnostic fresh vision differs from frozen training cache")
            partial = load_torch(qcache["source_partial_path"])
            if question["item_id"] not in done_query:
                other = cache["questions"][(qi + 1) % len(questions)]
                other_mask = other["bank_masks"][other["bank_names"].index("magnitude_v")]
                prediction = runtime.generate(eng, ctx, other_mask.nonzero().flatten(), p["max_new_tokens"])
                prediction["anls"] = runtime.anls(prediction["text"], question["answers"])
                own_index = qcache["bank_names"].index("magnitude_v")
                own = partial["results"][own_index]
                result["query"].append(dict(item_id=question["item_id"], parent_id=page["parent_id"],
                    swapped_from_item_id=other["item_id"], own_anls=own["anls"],
                    swapped_anls=prediction["anls"], swapped=prediction,
                    mask_jaccard=jaccard(qcache["bank_masks"][own_index], other_mask)))
                atomic_json(target, result)
            if question["item_id"] not in done_cross:
                good = qcache["bank_success"].nonzero().flatten().tolist()
                pairs = [(jaccard(qcache["bank_masks"][a], qcache["bank_masks"][b]), a, b)
                         for offset, a in enumerate(good) for b in good[offset+1:]]
                row = dict(item_id=question["item_id"], parent_id=page["parent_id"])
                if not pairs or min(pairs)[0] > p["cross_max_jaccard"]:
                    row.update(eligible=False, reason="no sufficiently distinct successful pair")
                else:
                    overlap, a, b = min(pairs)
                    mixed = crossed_mask(qcache["bank_masks"][a], qcache["bank_masks"][b],
                        seed_for(p["seed"], "cross", question["item_id"]))
                    prediction = runtime.generate(eng, ctx, mixed.nonzero().flatten(), p["max_new_tokens"])
                    prediction["anls"] = runtime.anls(prediction["text"], question["answers"])
                    row.update(eligible=True, pair_indices=[a, b], pair_jaccard=overlap,
                        pair_anls=[float(qcache["bank_anls"][a]), float(qcache["bank_anls"][b])],
                        crossed=prediction, crossed_success=sufficient(partial["full"], prediction,
                            p["success_min_anls"], p["success_tolerance"]))
                result["crossed_sets"].append(row)
                atomic_json(target, result)
            print(f"mechanism validation {question['item_id']}", flush=True)
    query = result["query"]
    crossed = [row for row in result["crossed_sets"] if row["eligible"]]
    result["summary"] = dict(
        query_questions=len(query),
        query_own_minus_swapped_anls=100*sum(row["own_anls"]-row["swapped_anls"] for row in query)/max(1,len(query)),
        distinct_success_pairs=len(crossed),
        crossed_failures=sum(not row["crossed_success"] for row in crossed),
        multi_success_fit_questions=sum(row["role"]=="fit" and row["successful_masks"]>=2 for row in result["bank_inventory"]),
        selection_rule="report only; all pretrained arms/seeds still run; no test-based selection")
    result["complete"] = True
    atomic_json(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostics(args.run_dir)


if __name__ == "__main__":
    main()
