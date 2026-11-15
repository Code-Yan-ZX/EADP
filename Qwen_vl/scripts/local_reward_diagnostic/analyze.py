"""Frozen-parameter reward ranking analysis with document-level bootstrap."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

import Levenshtein
import numpy as np

from reward_metrics import local_rewards, normalize, question_coverage, teacher_fragments
from run import OUT, atomic_json, load_records, SEEDS, sha

ARMS = ["topk", *[f"gumbel_{s}" for s in SEEDS]]
METRICS = ["global_raw", "global_normalized", "span_mean", "span_length_weighted", "tail20"]
CONTROLS = METRICS[:-1]
EPS = 1e-12


def bootstrap(values, seed=20261108):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci95=[None, None], n_documents=0)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws,[.025,.975]).tolist(), n_documents=len(values))


def compare_pairs(pages, near_tie=True, q_key="Q"):
    documents = []
    raw = {metric: Counter() for metric in METRICS}
    pair_rows = []
    for page in pages:
        eligible = []
        for a, b in itertools.combinations(ARMS, 2):
            left, right = page["arms"][a], page["arms"][b]
            q_delta = left[q_key] - right[q_key]
            if abs(q_delta) <= EPS:
                continue
            if near_tie and abs(left["rewards"]["global_raw"]-right["rewards"]["global_raw"]) > .01:
                continue
            record = dict(doc_id=page["doc_id"],a=a,b=b,qa_delta=q_delta, concordance={})
            for metric in METRICS:
                difference = left["rewards"][metric] - right["rewards"][metric]
                score = .5 if abs(difference)<=EPS else float(difference*q_delta > 0)
                record["concordance"][metric] = score
                raw[metric]["tied" if score == .5 else "correct" if score == 1 else "reversed"] += 1
            eligible.append(record)
            pair_rows.append(record)
        if eligible:
            documents.append(dict(doc_id=page["doc_id"],pairs=len(eligible),
                concordance={m:float(np.mean([r["concordance"][m] for r in eligible])) for m in METRICS}))
    concordance = {m: dict(**bootstrap([d["concordance"][m] for d in documents]),
                           raw_pair_counts=dict(raw[m])) for m in METRICS}
    incremental = {m: bootstrap([d["concordance"]["tail20"]-d["concordance"][m] for d in documents]) for m in CONTROLS}
    return dict(near_global_tie=near_tie, contributing_documents=len(documents),
                discordant_pairs=len(pair_rows), concordance=concordance,
                tail_increment=incremental, document_scores=documents,pair_scores=pair_rows)


def selection(pages, q_key="Q"):
    choices, differences = {}, {}
    for metric in METRICS:
        rows = []
        for page in pages:
            # max returns the first arm in the registered order on an exact tie.
            best_value=max(page["arms"][name]["rewards"][metric] for name in ARMS)
            arm=next(name for name in ARMS if page["arms"][name]["rewards"][metric]>=best_value-EPS)
            rows.append(dict(doc_id=page["doc_id"],selected=arm,Q=page["arms"][arm][q_key]))
        choices[metric] = dict(**bootstrap([r["Q"] for r in rows]),rows=rows)
    for control in CONTROLS:
        differences[control] = bootstrap([a["Q"]-b["Q"] for a,b in zip(choices["tail20"]["rows"],choices[control]["rows"])])
    return dict(by_reward=choices,tail_increment=differences,
        deterministic_topk=bootstrap([p["arms"]["topk"][q_key] for p in pages]),
        action_pool_oracle=bootstrap([max(p["arms"][a][q_key] for a in ARMS) for p in pages]))


def diagnostic(pages):
    return dict(near_tie=compare_pairs(pages,True),all_discordant=compare_pairs(pages,False),best_mask=selection(pages))


def verdict(primary, sensitivity, complete, gates_pass=True):
    if not complete:
        return "PENDING_INCOMPLETE_PANEL"
    if not gates_pass:
        return "INADEQUATE_INTERFACE"
    near = primary["near_tie"]
    if near["contributing_documents"] < 12 or near["discordant_pairs"] < 24:
        return "HOLD_INSUFFICIENT_INFORMATION"
    gains = near["tail_increment"]
    reverse = sensitivity["reverse_alignment"]["near_tie"]["tail_increment"]
    if any(reverse[c]["mean"] is None or
           (gains[c]["mean"] > 0) != (reverse[c]["mean"] > 0) or
           (gains[c]["mean"] < 0) != (reverse[c]["mean"] < 0) for c in CONTROLS):
        return "ALIGNMENT_SENSITIVE_HOLD"
    if any(gains[c]["mean"] <= 0 for c in ["span_mean","span_length_weighted"]):
        return "STOP_SPECIAL_TAIL_MECHANISM"
    if all(gains[c]["ci95"][0] > 0 for c in CONTROLS):
        return "SIGNAL_FOR_FURTHER_STUDY"
    return "HOLD_UNRESOLVED"


def prepare(manifest, records, width=64, reverse=False):
    population=[r for r in manifest["items"] if r["role"]=="primary"]
    assert len(population)==32 and len({r["doc_id"] for r in population})==32 and len({r["parent_id"] for r in population})==32
    index = {}
    for record in records:
        if record["kind"] != "ocr":
            continue
        key = (record["doc_id"],record["arm"])
        if key in index:
            raise ValueError(f"duplicate output {key}")
        index[key] = record
    pages, excluded, pending, all_details = [], [], [], []
    total_targets = teacher_targets = 0
    for item in [r for r in manifest["items"] if r["role"] == "primary"]:
        doc = item["doc_id"]
        missing = [arm for arm in ["full",*ARMS] if (doc,arm) not in index]
        if missing:
            pending.append(dict(doc_id=doc,missing=missing))
            continue
        full = index[doc,"full"]
        teacher = full["text"]
        qs = item["qa"]
        eligible = [q for q in qs if question_coverage(teacher,q["answers"])]
        total_targets += len(qs)
        teacher_targets += len(eligible)
        detail = dict(doc_id=doc, parent_id=item["parent_id"], total_annotated_QA=len(qs),
                      eligible_annotated_QA=len(eligible),eligible_question_ids=[q["question_id"] for q in eligible],
                      full_truncated=full["info"]["truncated"],
                      teacher_coverage=[question_coverage(teacher,q["answers"]) for q in qs],
                      compressed_coverage={}, fragments=[],
                      source_QA_total=len(item["source_qa"]),
                      source_QA_full_covered=sum(question_coverage(teacher,q["answers"]) for q in item["source_qa"]),
                      source_QA_compressed_covered={arm:sum(question_coverage(index[doc,arm]["text"],q["answers"]) for q in item["source_qa"]) for arm in ARMS})
        normalized, offsets = teacher_fragments(teacher,width)
        detail["fragments"] = [dict(start=a,end=b,text=normalized[a:b]) for a,b in offsets]
        arm_values = {}
        for arm in ARMS:
            rec = index[doc,arm]
            covered = [question_coverage(rec["text"],q["answers"]) for q in eligible]
            all_covered = [question_coverage(rec["text"],q["answers"]) for q in qs]
            detail["compressed_coverage"][arm] = dict(eligible=covered,all=all_covered)
            if not teacher.strip():
                continue
            arm_values[arm] = dict(Q=float(np.mean(covered)) if eligible else None,
                all_Q=float(np.mean(all_covered)), joint_Q=float(all(covered)) if eligible else None,
                retained=sum(covered),n_eligible=len(covered),all_retained=sum(all_covered),n_all=len(qs),
                truncated=rec["info"]["truncated"],
                rewards=local_rewards(teacher,rec["text"],width=width,reverse_alignment=reverse))
        detail["arm_values"] = arm_values
        all_details.append(detail)
        reason = "truncated_full_reference" if full["info"]["truncated"] else "empty_teacher" if not teacher.strip() else "fewer_than_5_teacher_fragments" if len(offsets)<5 else "no_full_reference_covered_QA" if not eligible else None
        if reason:
            excluded.append(dict(doc_id=doc,reason=reason,total_QA=len(qs),full_covered_QA=len(eligible)))
            continue
        pages.append(dict(doc_id=doc,parent_id=item["parent_id"],arms=arm_values,
                          n_targets=len(eligible),all_untruncated=all(not v["truncated"] for v in arm_values.values())))
    return pages,dict(excluded=excluded,pending=pending,all_details=all_details,
                      total_annotated_QA=total_targets,full_reference_covered_QA=teacher_targets)


def simple_rates(pages):
    return {arm:dict(individual_retained=sum(p["arms"][arm]["retained"] for p in pages),
                    eligible_targets=sum(p["arms"][arm]["n_eligible"] for p in pages),
                    page_macro_Q=float(np.mean([p["arms"][arm]["Q"] for p in pages])) if pages else None,
                    all_targets_retained_pages=sum(p["arms"][arm]["joint_Q"] for p in pages),
                    pages=len(pages)) for arm in ARMS}


def unfiltered_rates(details):
    """Keep all selected numeric aliases and all source-page QA visible.

    Source QA includes ambiguous/common strings and is descriptive only.
    """
    n_numeric=sum(d["total_annotated_QA"] for d in details)
    n_source=sum(d["source_QA_total"] for d in details)
    return dict(selected_numeric_QA_denominator=n_numeric,
        selected_numeric_full_covered=sum(d["eligible_annotated_QA"] for d in details),
        selected_numeric_compressed_covered={a:sum(sum(d["compressed_coverage"][a]["all"]) for d in details) for a in ARMS},
        all_source_QA_denominator=n_source,all_source_full_covered=sum(d["source_QA_full_covered"] for d in details),
        all_source_compressed_covered={a:sum(d["source_QA_compressed_covered"][a] for d in details) for a in ARMS},
        all_source_warning="Includes unfiltered, potentially ambiguous or common answer strings; descriptive only, not primary or actual QA accuracy.")


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def report(result):
    y = result["yield"]
    near = result["primary"]["near_tie"]
    lines = ["# LayoutLite local-reward diagnostic", "",f"Verdict: **{result['verdict']}**", "",
        "Inference only. External target is literal coverage of original human DocVQA answers in OCR text, not actual QA accuracy, field binding, or a trained improvement.","",
        f"Frozen primary pages: 32. OCR outputs: {y['ocr_outputs']}/192. Pending pages: {len(y['pending'])}.",
        f"Selected annotated numeric QA: {y['total_annotated_QA']}; covered by full OCR: {y['full_reference_covered_QA']}. Valid reward-ranking pages: {y['valid_pages']}.",
        f"Full references truncated: {y['full_truncated']}; compressed outputs truncated: {y['compressed_truncated']}; excluded pages: {len(y['excluded'])}.","",
        "## Registered near-global-tie ranking", "",
        f"Contributing independent documents: {near['contributing_documents']}; Q-discordant pairs within global-ratio difference <=0.01: {near['discordant_pairs']}. Gates: >=12 documents and >=24 pairs.","",
        "| Reward | Document-macro concordance | Tail minus reward [95% CI] |", "|---|---:|---:|"]
    for m in METRICS:
        val=near["concordance"][m]
        delta=near["tail_increment"].get(m)
        difference="—" if delta is None else f"{fmt(delta['mean'])} [{fmt(delta['ci95'][0])}, {fmt(delta['ci95'][1])}]"
        lines.append(f"| {m} | {fmt(val['mean'])} | {difference} |")
    lines += ["", "## Coverage of fixed actions", "",
              "| Action | Retained eligible answer evidence | All-targets-retained pages |", "|---|---:|---:|"]
    for arm,v in result["action_rates"].items():
        lines.append(f"| {arm} | {v['individual_retained']}/{v['eligible_targets']} | {int(v['all_targets_retained_pages'])}/{v['pages']} |")
    u=result["unfiltered_coverage"]
    lines += ["",f"All selected numeric QA (without teacher filtering): full {u['selected_numeric_full_covered']}/{u['selected_numeric_QA_denominator']}; compressed " + ", ".join(f"{a} {v}/{u['selected_numeric_QA_denominator']}" for a,v in u['selected_numeric_compressed_covered'].items())+".",
        f"All original source-page QA strings (unfiltered descriptive proxy): full {u['all_source_full_covered']}/{u['all_source_QA_denominator']}; compressed "+", ".join(f"{a} {v}/{u['all_source_QA_denominator']}" for a,v in u['all_source_compressed_covered'].items())+"."]
    lines += ["", "## Reward-selected actions (same five actions, no training)","",
              "| Reward | Page-macro evidence coverage |", "|---|---:|"]
    for m,v in result["primary"]["best_mask"]["by_reward"].items():
        lines.append(f"| {m} | {fmt(v['mean'])} |")
    lines += ["", "## Fixed sensitivities", "",
              "| Analysis | Documents / near-tie pairs | Tail minus span mean | Tail minus length mean |", "|---|---:|---:|---:|"]
    for name,v in result["sensitivities"].items():
        n=v["near_tie"]
        lines.append(f"| {name} | {n['contributing_documents']} / {n['discordant_pairs']} | {fmt(n['tail_increment']['span_mean']['mean'])} | {fmt(n['tail_increment']['span_length_weighted']['mean'])} |")
    lines += ["", "## Limits", "",
        "- Official released FireRed-OCR/score head; stock Transformers 4.57.6 execution adapted from upstream 5.6 code. All-token native interface gates are recorded separately.",
        "- Exact per-page 50% masks probe equal-cost actions. This does not reproduce the original globally calibrated clustered allocation rule, nor the project's 8B/25%-retention setting.",
        "- The panel is compact numeric DocVQA TRAIN pages, selected without compressed outcomes. Fresh source parents do not establish unseen-layout generalization.",
        "- Full OCR can contain errors; literal answer presence can miss formatting changes or credit a value in the wrong field. Provided OCR is a silver reference, not human whole-page gold.",
        "- No reward, seed, tie window or tail fraction was fitted to this panel. A ranking signal would still require controlled training and real-QA evaluation.", ""]
    return "\n".join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--partial",action="store_true");args=ap.parse_args()
    manifest=json.loads((OUT/"data/manifest.json").read_text())
    records=load_records(OUT/"predictions.jsonl")
    pages,yield_info=prepare(manifest,records)
    ocr=[r for r in records if r["kind"]=="ocr"]
    complete=len(ocr)==192 and not yield_info["pending"]
    if not complete and not args.partial:
        raise SystemExit(f"Panel incomplete: {len(ocr)}/192 OCR outputs; no final analysis")
    primary=diagnostic(pages)
    sensitivities={}
    for width in [32,128]:
        other,_=prepare(manifest,records,width=width)
        sensitivities[f"width_{width}"]=diagnostic(other)
    reverse_pages,_=prepare(manifest,records,reverse=True)
    sensitivities["reverse_alignment"]=diagnostic(reverse_pages)
    sensitivities["all_untruncated"]=diagnostic([p for p in pages if p["all_untruncated"]])
    changed=total=0
    changed_scores={m:0 for m in METRICS}
    changed_rankings={m:0 for m in METRICS}
    total_rankings=0
    for standard,reverse in zip(pages,reverse_pages):
        assert standard["doc_id"]==reverse["doc_id"]
        for arm in ARMS:
            total+=1
            changed+=abs(standard["arms"][arm]["rewards"]["tail20"]-reverse["arms"][arm]["rewards"]["tail20"])>EPS
            for metric in METRICS:
                changed_scores[metric]+=int(abs(standard["arms"][arm]["rewards"][metric]-reverse["arms"][arm]["rewards"][metric])>EPS)
        for a,b in itertools.combinations(ARMS,2):
            total_rankings+=1
            for metric in METRICS:
                def order(page):
                    difference=page["arms"][a]["rewards"][metric]-page["arms"][b]["rewards"][metric]
                    return 0 if abs(difference)<=EPS else 1 if difference>0 else -1
                changed_rankings[metric]+=int(order(standard)!=order(reverse))
    auxiliary=[]
    for item in [r for r in manifest["items"] if r["role"]=="primary"]:
        full=next((r for r in ocr if r["doc_id"]==item["doc_id"] and r["arm"]=="full"),None)
        if full:
            reference=json.loads(Path(item["provided_ocr_path"]).read_text())
            text="\n".join(v["text"] for v in reference["lines"].values())
            auxiliary.append(dict(doc_id=item["doc_id"],silver_ocr_teacher_ratio=float(Levenshtein.ratio(normalize(text),normalize(full["text"])))))
    yield_info.update(valid_pages=len(pages),ocr_outputs=len(ocr),full_truncated=sum(r["info"]["truncated"] for r in ocr if r["arm"]=="full"),
        compressed_truncated=sum(r["info"]["truncated"] for r in ocr if r["arm"]!="full"))
    result=dict(schema_version=1,complete=complete,primary=primary,sensitivities=sensitivities,
        yield_=None, action_rates=simple_rates(pages),alignment_sensitivity=dict(changed_tail_scores=changed,total_masks=total,
            changed_scores_by_metric=changed_scores,changed_rankings_by_metric=changed_rankings,total_mask_pairs=total_rankings),
        provided_ocr_silver_reference=auxiliary,score_tie_epsilon=EPS,bootstrap_seed=20261108,bootstrap_draws=10000,
        interpretation="evidence-coverage proxy; inference-only; not actual QA or trained method")
    result["yield"]=yield_info;result.pop("yield_")
    result["analysis_source_sha256"]=sha(Path(__file__))
    result["data_manifest_sha256"]=sha(OUT/"data/manifest.json")
    gate_paths=[OUT/"identity_gate.json",OUT/"batch_gates.json"]
    result["interface_gates"]={p.name:json.loads(p.read_text()) if p.exists() else {"pass_all":False,"missing":True} for p in gate_paths}
    result["unfiltered_coverage"]=unfiltered_rates(yield_info["all_details"])
    result["verdict"]=verdict(primary,sensitivities,complete,all(g["pass_all"] for g in result["interface_gates"].values()))
    name="analysis_partial" if not complete else "analysis"
    atomic_json(OUT/(name+".json"),result)
    (OUT/(name+".md")).write_text(report(result))
    print(json.dumps({k:result[k] for k in ["complete","verdict"]},indent=2))
    print(f"valid pages {len(pages)}; near-tie documents {primary['near_tie']['contributing_documents']}; pairs {primary['near_tie']['discordant_pairs']}")


if __name__=="__main__":main()
