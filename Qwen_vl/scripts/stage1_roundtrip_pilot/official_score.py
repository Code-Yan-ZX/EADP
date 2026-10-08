"""CPU-only scoring for live LLaVA predictions under repaired protocols.

GQA/VizWiz require complete unique GT coverage. POPE requires the complete
as-asked question set (random=2910 is the user-approved 2026-10-08 version),
and pairs GT by (image, question), never by historical file order.
Use a separate *.official.score.json output to preserve archived scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from rescore_official_20261008 import (
    _pope_key, gqa_official_score, pope_parse, vizwiz_official_score,
)
from vizwiz_official_normalization import VizWizNormalizer

PROTOCOL = "official-20261008-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_predictions(pred_file, expected_ids):
    """Check real lines before any dictionary can erase duplicates."""
    ids, seen, duplicates, invalid_text = [], set(), [], []
    for line in open(pred_file):
        row = json.loads(line)
        qid = str(row["question_id"])
        ids.append(qid)
        if qid in seen:
            duplicates.append(qid)
        seen.add(qid)
        value = row.get("text")
        if value is None or not str(value).strip() or str(value).strip().upper().startswith("FAILED"):
            invalid_text.append(qid)
    expected = set(map(str, expected_ids))
    errors = dict(n_lines=len(ids), n_expected=len(expected),
                  n_duplicate=len(duplicates), n_unknown=len(seen - expected),
                  n_missing=len(expected - seen), n_invalid_text=len(invalid_text))
    if len(ids) != len(expected) or any(errors[k] for k in
            ("n_duplicate", "n_unknown", "n_missing", "n_invalid_text")):
        raise ValueError(f"prediction integrity gate failed: {errors}")
    return errors


def pope_live_score(pred_file, questions, annotation_dir):
    """Pair by identity and handle a valid all-'no' model with F1=0."""
    by_id = {str(q["question_id"]): q for q in questions}
    predictions = [json.loads(line) for line in open(pred_file)]
    results = {}
    for cat in ("random", "popular", "adversarial"):
        gt = [json.loads(line) for line in open(
            Path(annotation_dir) / f"coco_pope_{cat}.json")]
        keys = [_pope_key(g["image"], g["text"]) for g in gt]
        if len(keys) != len(set(keys)) or any(g["label"] not in ("yes", "no") for g in gt):
            raise ValueError(f"POPE invalid or duplicate GT keys: {cat}")
        labels = dict(zip(keys, (g["label"] for g in gt)))
        seen, pairs = set(), []
        for row in predictions:
            q = by_id[str(row["question_id"])]
            if q["category"] != cat:
                continue
            key = _pope_key(q["image"], q["text"])
            if key in seen or key not in labels:
                raise ValueError(f"POPE duplicate or unmatched prediction key: {cat}: {key}")
            seen.add(key)
            pairs.append((int(pope_parse(row["text"]) == "yes"), int(labels[key] == "yes")))
        TP = sum(p == 1 and g == 1 for p, g in pairs)
        FP = sum(p == 1 and g == 0 for p, g in pairs)
        TN = sum(p == 0 and g == 0 for p, g in pairs)
        FN = sum(p == 0 and g == 1 for p, g in pairs)
        precision = TP / (TP + FP) if TP + FP else 0.0
        recall = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results[cat] = dict(f1=round(100 * f1, 3), acc=round(100 * (TP + TN) / len(pairs), 3),
                            precision=round(100 * precision, 3), recall=round(100 * recall, 3),
                            yes_ratio=round(sum(p for p, _ in pairs) / len(pairs), 3),
                            TP=TP, FP=FP, TN=TN, FN=FN, unmatched=0,
                            n_pred=len(pairs), n_gt=len(gt), n_gt_without_pred=len(set(keys) - seen))
    results["average_f1"] = round(sum(results[c]["f1"] for c in
                                    ("random", "popular", "adversarial")) / 3, 3)
    return results


def score(dataset, result_file, gt_file=None, question_file=None, annotation_dir=None):
    inputs = {"predictions": {"path": str(result_file), "sha256": sha256(result_file)}}
    if dataset == "pope":
        if not question_file or not annotation_dir:
            raise ValueError("POPE requires --question-file and --annotation-dir")
        questions = [json.loads(line) for line in open(question_file)]
        qids = [str(q["question_id"]) for q in questions]
        if len(qids) != len(set(qids)):
            raise ValueError("duplicate question IDs in POPE question file")
        if {q["category"] for q in questions} != {"random", "popular", "adversarial"}:
            raise ValueError("POPE requires all three categories")
        integrity = checked_predictions(result_file, qids)
        results = pope_live_score(result_file, questions, annotation_dir)
        inputs["questions"] = {"path": str(question_file), "sha256": sha256(question_file)}
        for cat in ("random", "popular", "adversarial"):
            path = Path(annotation_dir) / f"coco_pope_{cat}.json"
            inputs[f"gt_{cat}"] = {"path": str(path), "sha256": sha256(path)}
        result = dict(summary=dict(average_f1=results["average_f1"], metric="three-category mean F1",
                                   n=integrity["n_lines"], complete=True), categories=results,
                      integrity=integrity)
    else:
        if not gt_file:
            raise ValueError(f"{dataset} requires --gt-file")
        gt = json.load(open(gt_file))
        expected = range(len(gt)) if dataset == "vizwiz" else gt.keys()
        integrity = checked_predictions(result_file, expected)
        if dataset == "vizwiz":
            class Stub:
                def getImgs(self):
                    return []
            summary, per = vizwiz_official_score(
                result_file, gt_file, VizWizNormalizer(Stub(), Stub()))
        else:
            summary, per = gqa_official_score(result_file, gt_file)
        if not summary["complete"]:
            raise ValueError(f"{dataset} GT coverage gate failed: {summary}")
        inputs["gt"] = {"path": str(gt_file), "sha256": sha256(gt_file)}
        result = dict(summary=summary, per_question=per, integrity=integrity)
    result.update(dataset=dataset, protocol=PROTOCOL, inputs=inputs)
    return result


def main(dataset=None):
    ap = argparse.ArgumentParser(description=__doc__)
    if dataset is None:
        ap.add_argument("--dataset", required=True, choices=("vizwiz", "gqa", "pope"))
    ap.add_argument("--result-file", required=True)
    ap.add_argument("--gt-file")
    ap.add_argument("--question-file")
    ap.add_argument("--annotation-dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if Path(args.out).resolve() == Path(args.result_file).resolve():
        ap.exit(1, "Scoring output must be separate from raw predictions.\n")
    try:
        result = score(dataset or args.dataset, args.result_file, args.gt_file,
                       args.question_file, args.annotation_dir)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        ap.exit(1, f"SCORING REJECTED: {exc}\n")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, target)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
