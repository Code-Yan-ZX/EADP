"""MME official acc+acc_plus with GT from the verified evaluation archive.

The locally reconstructed MME_Benchmark_release_version contains 162 wrong
labels. Never use it for scoring. The archive's Your_Results and LaVIN GT
agree on all 2374 questions and match the checksum-locked VLMEvalKit TSV.
Keep perception and cognition separate: the EADP LLaVA table uses perception.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import zipfile

GT_ARCHIVE_SHA256 = "b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee"
QUESTION_SHA256 = "9f67707cb72d2a71df955d0dce84337f2f22eafe2ed473ca3e148c187ac360c5"
PERCEPTION = ("existence", "count", "position", "color", "posters", "celebrity",
              "scene", "landmark", "artwork", "OCR")
COGNITION = ("commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning")
PROTOCOL = "mme-canonical-gt-20261008-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_gt(archive):
    if sha256(archive) != GT_ARCHIVE_SHA256:
        raise ValueError("unverified MME GT archive SHA256")
    copies = []
    with zipfile.ZipFile(archive) as z:
        for folder in ("Your_Results", "LaVIN"):
            gt = {}
            for category in PERCEPTION + COGNITION:
                member = f"eval_tool/{folder}/{category}.txt"
                for line in z.read(member).decode("utf-8").splitlines():
                    image, question, answer, *_ = line.split("\t")
                    key = category, Path(image).stem, question
                    if key in gt or answer.lower() not in ("yes", "no"):
                        raise ValueError(f"invalid canonical MME GT: {key}")
                    gt[key] = answer.lower()
            copies.append(gt)
    if copies[0] != copies[1] or len(copies[0]) != 2374:
        raise ValueError("canonical MME GT copies disagree or have incomplete coverage")
    groups = Counter((c, image) for c, image, question in copies[0])
    if any(n != 2 for n in groups.values()):
        raise ValueError("MME requires exactly two GT questions per image")
    return copies[0]


def question_key(question_id, prompt, gt):
    category = question_id.split("/")[0]
    image = Path(question_id).stem
    question = prompt.replace("Answer the question using a single word or phrase.", "").strip()
    if "Please answer yes or no." not in question:
        question += " Please answer yes or no."
    key = category, image, question
    if key not in gt:
        key = category, image, question.replace(" Please answer yes or no.", "  Please answer yes or no.")
    if key not in gt:
        raise ValueError(f"question has no canonical GT: {key}")
    return key


def checked_records(result_file, question_file, gt):
    if sha256(question_file) != QUESTION_SHA256:
        raise ValueError("unverified MME question-file SHA256")
    questions = [json.loads(line) for line in open(question_file)]
    expected = {(q["question_id"], q["text"]) for q in questions}
    expected_gt = {question_key(*key, gt) for key in expected}
    if len(expected) != 2374 or expected_gt != set(gt):
        raise ValueError("MME questions do not cover canonical GT")
    rows = [json.loads(line) for line in open(result_file)]
    seen = set()
    for row in rows:
        key = row["question_id"], row["prompt"]
        if key in seen or key not in expected:
            raise ValueError(f"duplicate or unknown MME prediction identity: {key}")
        seen.add(key)
        answer = row.get("text")
        if not isinstance(answer, str) or not answer.strip() or answer.strip().upper().startswith("FAILED"):
            raise ValueError(f"invalid MME prediction text: {key}")
        if "\n" in answer or "\t" in answer:
            raise ValueError(f"MME conversion requires a single line without tabs: {key}")
    if seen != expected:
        raise ValueError(f"incomplete MME predictions: {len(seen)}/2374")
    return rows


def parse_prediction(answer):
    # Exact official calculation.py semantics (first four characters).
    answer = answer.lower()
    if answer in ("yes", "no"):
        return answer
    return "yes" if "yes" in answer[:4] else "no" if "no" in answer[:4] else "other"


def score_rows(rows, gt):
    categories, per_question = defaultdict(list), []
    for row in rows:
        category, image, question = question_key(row["question_id"], row["prompt"], gt)
        prediction = parse_prediction(row["text"])
        record = dict(category=category, image=image, question=question,
                      question_id=row["question_id"], prompt=row["prompt"],
                      gt=gt[category, image, question], prediction=prediction,
                      correct=prediction == gt[category, image, question])
        categories[category].append(record)
        per_question.append(record)
    scores = {}
    for category in PERCEPTION + COGNITION:
        entries = categories[category]
        images = defaultdict(list)
        for entry in entries:
            images[entry["image"]].append(entry["correct"])
        if any(len(pair) != 2 for pair in images.values()):
            raise ValueError(f"incomplete MME image pairs: {category}")
        correct = sum(e["correct"] for e in entries)
        both_correct = sum(all(pair) for pair in images.values())
        acc = correct / len(entries)
        acc_plus = both_correct / len(images)
        scores[category] = dict(n=len(entries), n_images=len(images), correct=correct,
                                both_correct=both_correct, acc=acc, acc_plus=acc_plus,
                                score=100 * (acc + acc_plus),
                                other=sum(e["prediction"] == "other" for e in entries))
    perception = sum(scores[c]["score"] for c in PERCEPTION)
    cognition = sum(scores[c]["score"] for c in COGNITION)
    return dict(summary=dict(perception=perception, cognition=cognition,
                             total=perception + cognition, n=len(rows), complete=True,
                             metric="category sum of 100*(accuracy+paired-image accuracy)"),
                categories=scores, per_question=per_question)


def converted_lines(rows, gt):
    """Group by image so the original official calculator's two-line chunks hold."""
    by_category = defaultdict(lambda: defaultdict(list))
    for row in rows:
        category, image, question = question_key(row["question_id"], row["prompt"], gt)
        by_category[category][image].append("\t".join(
            (image + ".txt", question, gt[category, image, question], row["text"])) + "\n")
    return {c: "".join(line for pair in images.values() for line in pair)
            for c, images in by_category.items()}


def score(result_file, question_file, gt_archive):
    gt = canonical_gt(gt_archive)
    rows = checked_records(result_file, question_file, gt)
    result = score_rows(rows, gt)
    result.update(protocol=PROTOCOL, dataset="mme", inputs={
        name: dict(path=str(path), sha256=sha256(path)) for name, path in
        (("predictions", result_file), ("questions", question_file), ("gt_archive", gt_archive))})
    return result, converted_lines(rows, gt)


def write_converted(directory, converted):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Converted directory must be new or empty; archived results are preserved.")
    directory.mkdir(parents=True, exist_ok=True)
    for category, lines in converted.items():
        (directory / f"{category}.txt").write_text(lines)


def convert_main():
    """Compatibility entry for the original MME --experiment converter."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--results-dir", help="Use a new directory when the historical experiment exists")
    ap.add_argument("--gt-archive", default="eval_tool.zip")
    ap.add_argument("--question-file", default="llava_mme.jsonl")
    args = ap.parse_args()
    predictions = Path("answers") / f"{args.experiment}.jsonl"
    directory = args.results_dir or Path("eval_tool/answers") / args.experiment
    try:
        gt = canonical_gt(args.gt_archive)
        rows = checked_records(predictions, args.question_file, gt)
        write_converted(directory, converted_lines(rows, gt))
        # A manifest makes the label source explicit on any later score reuse.
        (Path(directory) / "canonical_gt_provenance.json").write_text(json.dumps(dict(
            protocol=PROTOCOL, n=len(rows), gt_archive_sha256=sha256(args.gt_archive),
            questions_sha256=sha256(args.question_file), predictions_sha256=sha256(predictions),
        ), indent=1) + "\n")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        ap.exit(1, f"MME CONVERSION REJECTED: {exc}\n")
    print(f"Canonical GT conversion: {directory}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-file", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--gt-archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--converted-dir")
    args = ap.parse_args()
    if Path(args.out).resolve() == Path(args.result_file).resolve():
        ap.exit(1, "Score output must be separate from predictions.\n")
    try:
        result, converted = score(args.result_file, args.question_file, args.gt_archive)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        ap.exit(1, f"MME SCORING REJECTED: {exc}\n")
    if args.converted_dir:
        try:
            write_converted(args.converted_dir, converted)
        except ValueError as exc:
            ap.exit(1, f"MME SCORING REJECTED: {exc}\n")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(temporary, target)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
