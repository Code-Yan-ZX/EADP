"""Offline re-scoring under OFFICIAL protocols (audit round 2026-10-08).

Operates ONLY on the snapshot at audit_rescore_20261008/raw_snapshot/;
originals are never touched.  Output goes to audit_rescore_20261008/rescored/.

Fixes applied (audit items #2 #3 #4):
- VizWiz (#3): official leave-one-out accuracy.  Normalization + min(1, n/3)
  loop ported verbatim from the official VizWiz API vqaEval.py
  (vizwiz.org API.zip, PythonEvaluationTools/vqaEvaluation/vqaEval.py;
  archived as scripts_snapshot/vizwiz_official_vqaEval.py).  The previous
  scorer used min(1, match/3) without leave-one-out (match=3 gave 1.0, official 0.9).
- POPE (#2): official eval_pope.py logic verbatim; headline metric = average
  F1 over random/popular/adversarial (user decision A: F1 becomes the main
  table column).  Predictions matched to GT by question_id (the official
  script zips by file order; both variants are computed and compared).
- GQA (#4): exact-match lowercase, PLUS the completeness/uniqueness checks
  the old scorer lacked (silent skip of unknown question_id, duplicate ids,
  wrong denominator).
"""
import argparse
import importlib.util
import json
import os
import sys

SNAP = "/media/disk2/YZX/research/audit_rescore_20261008"
OUT = os.path.join(SNAP, "rescored")

# ---------------------------------------------------------------- VizWiz ----

def load_official_eval():
    """Instantiate the official VQAEval with stub VQA objects to reuse its
    normalization methods verbatim."""
    src_path = os.path.join(SNAP, "scripts_snapshot", "vizwiz_official_vqaEval.py")
    src = open(src_path).read()
    # The caption-metric section imports pycocoevalcap relatively; it is not
    # used by the accuracy path.  Neutralize those imports for standalone load.
    lines = src.splitlines(keepends=True)
    out = []
    for ln in lines:
        if ln.lstrip().startswith("from .pycocoevalcap"):
            out.append("# neutralized for standalone load (caption metric only): " + ln)
        else:
            out.append(ln)
    src = "".join(out)
    spec = importlib.util.spec_from_file_location("vizwiz_official_vqaEval", src_path)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, src_path, "exec"), mod.__dict__)

    class _Stub:
        def getImgs(self):
            return []

    return mod.VQAEval(_Stub(), _Stub(), n=2)


def vizwiz_official_score(pred_file, gt_file, ev):
    """Official accuracy: for each of the 10 GT answers i, hold it out,
    acc_i = min(1, (# of the other 9 matching the normalized prediction)/3),
    average the 10 values.  vqaEval.py lines 96-104.
    N05 gate (audit 2026-10-08): real line count / duplicate / unknown ids
    counted BEFORE any dict; final score only emitted when the prediction
    id set exactly covers the 4319 validation questions."""
    # GT is val.json (list); predictions' question_id = index in this list,
    # same mapping the generation pipeline and old scorer used (verified by
    # prompt == gt[i]["question"] on arm LRMAIN025).
    gt = {i: q for i, q in enumerate(json.load(open(gt_file)))}
    gt_ids = set(gt.keys())
    raw_ids, dup_ids, unknown_ids = [], [], []
    texts = {}
    for line in open(pred_file):
        d = json.loads(line)
        qid = int(d["question_id"])
        raw_ids.append(qid)
        if qid in texts:
            dup_ids.append(qid)
        if qid not in gt_ids:
            unknown_ids.append(qid)
        texts[qid] = str(d["text"])
    n_lines = len(raw_ids)
    n_unique = len(set(raw_ids))
    missing_ids = sorted(gt_ids - set(raw_ids))
    complete = (n_lines == len(gt_ids) and n_unique == len(gt_ids)
                and not dup_ids and not unknown_ids and not missing_ids)
    per, n, c = [], 0, 0.0
    if complete:
        for qid in gt_ids:
            resAns = texts[qid].replace("\n", " ").replace("\t", " ").strip()
            resAns = ev.processPunctuation(resAns)
            resAns = ev.processDigitArticle(resAns)
            answers = gt[qid]["answers"]
            gtAcc = []
            for i, _ in enumerate(answers):
                other = [item for j, item in enumerate(answers) if i != j]
                matching = [item for item in other if item["answer"] == resAns]
                gtAcc.append(min(1, float(len(matching)) / 3))
            avg = float(sum(gtAcc)) / len(gtAcc)
            c += avg
            n += 1
            per.append({"question_id": qid, "acc": avg})
    return {"n_lines": n_lines, "n_unique_ids": n_unique,
            "n_gt_ids": len(gt_ids),
            "n_dup": len(dup_ids), "n_unknown": len(unknown_ids),
            "n_missing": len(missing_ids),
            "complete": complete,
            "accuracy": round(100 * c / n, 2) if complete else None,
            "metric": "official VizWiz leave-one-out VQA accuracy "
                      "(vqaEval.py:96-104); hard gate: pred ids == 4319 "
                      "val ids (N05 fix)"}, per


def old_vizwiz_compare(arm, per_new):
    old_path = os.path.join(SNAP, "raw_snapshot", "anchorzip_p3",
                            "vizwiz" if not arm.startswith("next") else "next_vizwiz",
                            f"{arm}.score.json")
    if not os.path.exists(old_path):
        return None
    old = json.load(open(old_path))
    old_per = {q["question_id"]: q["acc"] for q in old.get("per_question", [])}
    diffs = [abs(old_per.get(q["question_id"], -1) - q["acc"]) for q in per_new]
    changed = sum(1 for d in diffs if d > 1e-9)
    return {"old_accuracy": old["summary"]["accuracy"], "questions_changed": changed,
            "n_compared": len(diffs)}


# ------------------------------------------------------------------ POPE ----

def pope_parse(text):
    """Verbatim from eval_pope.py: first sentence, comma strip, no/not/no => no."""
    if text.find(".") != -1:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split(" ")
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def pope_metrics(pairs):
    """pairs: list of (pred 0/1, label 0/1) -- verbatim confusion logic."""
    TP = FP = TN = FN = 0
    for pred, label in pairs:
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 0 and label == 1:
            FN += 1
    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    yes_ratio = sum(p for p, _ in pairs) / len(pairs)
    return {"f1": round(100 * f1, 3), "acc": round(100 * acc, 3),
            "precision": round(100 * precision, 3), "recall": round(100 * recall, 3),
            "yes_ratio": round(yes_ratio, 3), "TP": TP, "FP": FP, "TN": TN, "FN": FN}


def _pope_key(image, text):
    return (image, text.split("\n")[0].strip().replace("imange", "image"))


def pope_official_score(pred_file, qfile, gt_dir):
    """POPE re-scoring.

    Primary pairing: per-question (image, question-text) key.  Rationale
    (audit finding, 2026-10-08): llava_pope_test.jsonl's random block has
    2910 questions of which 144 carry the old POPE 'imange' typo, while the
    GT file on disk (byte-identical to the current RUCAIBox/POPE release)
    has 3000 corrected lines -- so the official eval_pope.py order-zip
    pairing misaligns the random category.  The (image, text) keys are
    unique on both sides, giving an unambiguous pairing.  The historical
    order-zip metrics are also reported to quantify the discrepancy."""
    questions = {q["question_id"]: q for q in
                 (json.loads(l) for l in open(qfile))}
    answers = [json.loads(l) for l in open(pred_file)]
    results, f1s = {}, []
    for cat in ("random", "popular", "adversarial"):
        gt_lines = [json.loads(l) for l in
                    open(os.path.join(gt_dir, f"coco_pope_{cat}.json"))]
        gt_by_key = {_pope_key(g["image"], g["text"]): g["label"] for g in gt_lines}
        cur = [a for a in answers
               if questions[a["question_id"]]["category"] == cat]
        gt_keys = [_pope_key(g["image"], g["text"]) for g in gt_lines]
        if len(set(gt_keys)) != len(gt_keys):
            results[cat] = {"ERROR": "duplicate GT (image,text) keys"}
            continue
        pred_keys = [_pope_key(questions[a["question_id"]]["image"],
                               questions[a["question_id"]]["text"]) for a in cur]
        if len(set(pred_keys)) != len(pred_keys):
            results[cat] = {"ERROR": "duplicate pred (image,text) keys"}
            continue
        missing_gt = sorted(set(gt_keys) - set(pred_keys))
        pairs, unmatched = [], 0
        for a in cur:
            q = questions[a["question_id"]]
            label = gt_by_key.get(_pope_key(q["image"], q["text"]))
            if label is None:
                unmatched += 1
                continue
            pairs.append((0 if pope_parse(a["text"]) == "no" else 1,
                          0 if label == "no" else 1))
        m = pope_metrics(pairs)
        m["unmatched"] = unmatched
        m["n_pred"] = len(cur)
        m["n_gt"] = len(gt_lines)
        m["n_gt_without_pred"] = len(missing_gt)  # set-identity gap (N02)
        results[cat] = m
        f1s.append(m["f1"])
        # historical order-zip (what the in-round eval_pope.py runs did)
        h = [p for p in zip(cur, gt_lines)]
        hist = pope_metrics([(0 if pope_parse(a["text"]) == "no" else 1,
                              0 if g["label"] == "no" else 1) for a, g in h])
        results[cat + "_historical_order_zip"] = hist
    results["average_f1"] = round(sum(f1s) / len(f1s), 3)
    hist_f1 = [results[c + "_historical_order_zip"]["f1"]
               for c in ("random", "popular", "adversarial")]
    results["average_f1_historical_order_zip"] = round(sum(hist_f1) / len(hist_f1), 3)
    return results


# ------------------------------------------------------------------- GQA ----

def gqa_official_score(pred_file, gt_file):
    """N03 fix (audit 2026-10-08): count REAL lines and duplicate ids BEFORE
    building the dict (dict construction silently overwrote duplicates, so
    the old check could never fail).  Gate: pred id set must EQUAL the GT id
    set exactly; on any violation no final accuracy is emitted."""
    gt = json.load(open(gt_file))
    gt_ids = set(gt.keys())
    raw_ids, dup_ids, unknown_ids = [], [], []
    texts = {}
    for ln, line in enumerate(open(pred_file)):
        d = json.loads(line)
        qid = str(d["question_id"])
        raw_ids.append(qid)
        if qid in texts:
            dup_ids.append(qid)
        if qid not in gt_ids:
            unknown_ids.append(qid)
        texts[qid] = str(d["text"]).strip().lower()
    n_lines = len(raw_ids)
    n_unique = len(set(raw_ids))
    missing_ids = sorted(gt_ids - set(raw_ids))
    complete = (n_lines == len(gt_ids) and n_unique == len(gt_ids)
                and not dup_ids and not unknown_ids and not missing_ids)
    per, c = [], 0
    if complete:
        for qid in gt_ids:  # fixed GT denominator
            ok = str(gt[qid]["answer"]).strip().lower() == texts[qid]
            c += ok
            per.append({"question_id": qid, "correct": bool(ok)})
    return {"n_lines": n_lines, "n_unique_ids": n_unique,
            "n_gt_ids": len(gt_ids),
            "dup_ids": dup_ids[:50], "n_dup": len(dup_ids),
            "unknown_ids": unknown_ids[:50], "n_unknown": len(unknown_ids),
            "missing_ids_count": len(missing_ids),
            "complete": complete,
            "accuracy": round(100 * c / len(gt_ids), 2) if complete else None,
            "metric": "exact-match lowercase-strip, GQA testdev; "
                      "hard gate: pred id set == GT id set (N03 fix)"}, per


def _gqa_pred_map(pred_file):
    m = {}
    for line in open(pred_file):
        d = json.loads(line)
        m[str(d["question_id"])] = str(d["text"]).strip().lower()
    return m



def model_of(d):
    return "next" if d.startswith("next") else "v15"


# ------------------------------------------------------------------- SQA ----

def sqa_check(jsonl_path, qfile):
    """N01 gate (audit 2026-10-08): completeness/identity check for SQA
    prediction files.  Verifies real line count, unique question ids, id-set
    equality with the question file, IMG subset count, and FAILED-style
    answers.  No rescoring here -- the official eval_science_qa is the
    scorer; this is the reuse/generation gate."""
    q = json.load(open(qfile))
    q_ids = [x["id"] for x in q]
    q_id_set = set(q_ids)
    n_img = sum(1 for x in q if "image" in x)
    raw_ids, dups, bad_text = [], 0, 0
    texts = set()
    for line in open(jsonl_path):
        d = json.loads(line)
        qid = d["question_id"]
        raw_ids.append(qid)
        if qid in texts:
            dups += 1
        texts.add(qid)
        t = str(d.get("text", "")).strip()
        if not t or t.upper().startswith("FAILED"):
            bad_text += 1
    n_lines = len(raw_ids)
    unique = len(texts)
    unknown = sum(1 for i in raw_ids if i not in q_id_set)
    missing = len(q_id_set - texts)
    return {"n_lines": n_lines, "n_unique": unique, "n_question_file": len(q_ids),
            "n_img_questions": n_img,
            "dup_ids": dups, "unknown_ids": unknown,
            "missing_ids": missing,
            "failed_or_empty_text": bad_text,
            "complete": (n_lines == len(q_ids) and unique == len(q_ids)
                         and dups == 0 and unknown == 0 and missing == 0)}

# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="vizwiz,pope,gqa")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    os.makedirs(OUT, exist_ok=True)
    summary = {}

    if "vizwiz" in tasks:
        ev = load_official_eval()
        gt = os.path.join(SNAP, "raw_snapshot", "gt", "vizwiz_val.json")
        rows = {}
        for d in ("vizwiz", "next_vizwiz"):
            for f in sorted(os.listdir(os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d))):
                if not f.endswith(".jsonl"):
                    continue
                arm = f[:-6]
                res, per = vizwiz_official_score(
                    os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d, f), gt, ev)
                res["vs_old_scorer"] = old_vizwiz_compare(arm, per)
                rows[f"{d}/{arm}"] = res
                json.dump({"summary": res, "per_question": per},
                          open(os.path.join(OUT, f"vizwiz_{model_of(d)}_{arm}_official.json"), "w"))
        summary["vizwiz"] = rows

    if "pope" in tasks:
        rows = {}
        gt_dir = os.path.join(SNAP, "raw_snapshot", "gt")
        qfile = os.path.join(gt_dir, "llava_pope_test.jsonl")
        for d in ("pope", "next_pope"):
            for f in sorted(os.listdir(os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d))):
                if not f.endswith(".jsonl") or os.path.getsize(
                        os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d, f)) == 0:
                    continue
                arm = f[:-6]
                n_lines = sum(1 for _ in open(os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d, f)))
                if n_lines != 8910:
                    rows[f"{d}/{arm}"] = {"INCOMPLETE": n_lines}
                    continue
                res = pope_official_score(
                    os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d, f), qfile, gt_dir)
                rows[f"{d}/{arm}"] = res
                json.dump(res, open(os.path.join(OUT, f"pope_{model_of(d)}_{arm}_official.json"), "w"))
        summary["pope"] = rows

    if "gqa" in tasks:
        gt = os.path.join(SNAP, "raw_snapshot", "gt", "gqa_testdev_balanced_questions.json")
        rows = {}
        for d in ("gqa", "next_gqa"):
            for f in sorted(os.listdir(os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d))):
                if not f.endswith(".jsonl"):
                    continue
                arm = f[:-6]
                res, per = gqa_official_score(
                    os.path.join(SNAP, os.path.join("raw_snapshot", "anchorzip_p3"), d, f), gt)
                rows[f"{d}/{arm}"] = res
                json.dump({"summary": res, "per_question": per},
                          open(os.path.join(OUT, f"gqa_{model_of(d)}_{arm}_official.json"), "w"))
        summary["gqa"] = rows

    if "sqa" in tasks:
        rows = {}
        qfile = ("/media/disk2/YZX/research/EADP_amp/LLaVA/playground/"
                 "data/eval/scienceqa/llava_test_CQM-I.json")
        base = os.path.join(SNAP, "raw_snapshot", "anchorzip_p3")
        # live CQMI outputs live next to the snapshot source tree
        live = ("/media/disk2/YZX/research/EADP_amp/LLaVA/playground/"
                "data/eval/anchorzip_p3")
        for d in ("sqa", "next_sqa"):
            for f in sorted(os.listdir(os.path.join(live, d))):
                if not f.endswith("_CQMI_vicuna.jsonl"):
                    continue
                arm = f[:-len("_CQMI_vicuna.jsonl")]
                res = sqa_check(os.path.join(live, d, f), qfile)
                rows[f"{d}/{arm}"] = res
                json.dump(res, open(os.path.join(
                    OUT, f"sqa_{model_of(d)}_{arm}_CQMI_check.json"), "w"))
        summary["sqa"] = rows

    json.dump(summary, open(os.path.join(OUT, "rescore_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
