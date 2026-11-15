"""Freeze 16 unused DocVQA reserve pages, first existing numeric QA per page.

CPU-only: selection never reads model predictions.  Previous predictions are
read afterwards solely to audit that no selected parent entered the old panel.
Original images, answer aliases and the source manifest are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image

QWEN = Path(__file__).resolve().parents[2]
SOURCE = QWEN / "outputs/local_reward_diagnostic/data/manifest.json"
OUT = QWEN / "outputs/catchup_rerank/data/manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source):
    source = Path(source).resolve()
    raw = json.loads(source.read_text())
    if raw.get("complete") is not True:
        raise ValueError("Source data manifest is incomplete")
    # Source order and qa[0] are the selection rule. Do not sort, teacher-filter,
    # select a different question or replace a page after model evaluation.
    reserves = [item for item in raw["items"] if item["role"] == "reserve"]
    if len(reserves) != 16:
        raise ValueError(f"Expected all 16 reserve pages, found {len(reserves)}")
    items = []
    for rank, item in enumerate(reserves):
        qa = item["qa"][0]
        answers = qa["answers"]
        if not answers or not all(isinstance(x, str) for x in answers):
            raise ValueError("Original human answer aliases must be nonempty strings")
        if not any(sum(c.isdigit() for c in answer) >= 2 for answer in answers):
            raise ValueError("Source qa[0] violates the existing numeric-panel scope")
        image = Path(item["image_path"]).resolve()
        image_hash = sha256(image)
        if image_hash != item["source_sha256"]:
            raise ValueError(f"Source image hash changed: {image}")
        with Image.open(image) as picture:
            image_size = list(picture.size)
        if image_size != item["source_image_size"]:
            raise ValueError(f"Source image dimensions changed: {image}")
        selected = dict(
            item_id=item["doc_id"] + "_q" + str(qa["question_id"]),
            doc_id=item["doc_id"], parent_id=item["parent_id"],
            source_document_id=item["parent_id"], dataset="DocVQA_TRAIN",
            image_path=str(image), image_sha256=image_hash,
            source_sha256=image_hash, source_image_size=image_size,
            question_id=qa["question_id"], question=qa["question"],
            answers=list(answers), selected_qa_index=0,
            source_role="reserve", selection_rank=rank,
            source_selection_rank=item["selection_rank"],
            provided_ocr_word_count=item["provided_ocr_word_count"],
            source_qa_count=len(item.get("source_qa", [])),
            source_numeric_qa_count=len(item["qa"]),
            annotation_repository_revision=item["qa_repository_revision"],
            source_parquet=item["source_parquet"],
            source_parquet_row=item["source_parquet_row"],
            source_parquet_sha256=item["source_parquet_sha256"],
            selected_source_qa=qa,
            region_warning="Provided-OCR boxes are silver localization only; not selection inputs or human field boundaries.",
        )
        items.append(selected)

    parents = [item["parent_id"] for item in items]
    if len(set(parents)) != 16 or len({x["image_sha256"] for x in items}) != 16:
        raise ValueError("Reserve pages are not independent source parents/images")
    excluded = set(raw["provenance"]["excluded_parent_documents"])
    if excluded & set(parents):
        raise ValueError("Reserve parents overlap previous exclusion pool")
    # This audit is separate from selection, and fails rather than replacing a
    # selected sample. No answer, correctness, score or compression result is read.
    predictions = QWEN / "outputs/local_reward_diagnostic/predictions.jsonl"
    old_rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    overlap = set(parents) & {row["parent_id"] for row in old_rows}
    if overlap:
        raise ValueError(f"Reserve parents unexpectedly appeared in previous panel: {sorted(overlap)}")

    return dict(
        schema_version=1, complete=True,
        provenance=dict(
            source_manifest=str(source), source_manifest_sha256=sha256(source),
            original_dataset=raw["provenance"]["dataset"],
            original_repository=raw["provenance"]["repository"],
            original_repository_revision=raw["provenance"]["repository_revision"],
            original_split="TRAIN; exploratory within-project pilot, not a held-out benchmark or unseen-layout claim",
            selection="All role=reserve pages in original manifest order; qa[0] exactly; no model-output selection or replacement.",
            source_panel_seed=raw["provenance"]["seed"],
            exclusion_policy=raw["provenance"]["exclusion_policy"],
            excluded_parent_counts=raw["provenance"]["excluded_parent_counts"],
            source_predictions_audited=str(predictions),
            source_predictions_sha256=sha256(predictions),
            source_prediction_rows=len(old_rows),
            source_prediction_roles=dict(Counter(row.get("kind", "unknown") for row in old_rows)),
            source_prediction_parent_overlap=sorted(overlap),
            evidence_ground_truth="Original human DocVQA QA answer variants; score actual generated answers. No OCR transcript coverage proxy.",
        ),
        protocol=dict(
            design="Exploratory pilot: 16 independent source pages, one preselected numeric QA each",
            keep_all_samples=True, teacher_correct_filter=False,
            failures_and_truncations="Retain and report every output, including full-model errors and truncated answers.",
            planned_input_tokens=1024, planned_visual_budget=256,
            planned_candidate_budget=512,
            planned_arms=["full", "rres512", "baseline", "dense_full", "dense_2k", "ltc_2k"],
            runner_freeze_required="Runner must freeze exact baseline, reranking formulas, candidate construction, prompt, generation cap, model and code hashes before new QA generation.",
        ),
        counts=dict(pages=16, independent_source_documents=16, questions=16,
                    original_answer_aliases=sum(len(item["answers"]) for item in items)),
        data_validation=dict(independent_parents=True, unique_original_images=True,
                             all_source_image_hashes_verified=True,
                             all_image_dimensions_verified=True,
                             prior_exclusion_parent_overlap=0,
                             previous_ocr_panel_parent_overlap=0,
                             one_first_qa_per_parent=True),
        items=items, records=items,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = build(args.source)
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_text() != encoded:
        raise ValueError("Refusing to change an existing frozen manifest; use a separate --out")
    if not args.out.exists():
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.out)
    print(json.dumps(dict(manifest=str(args.out.resolve()), sha256=sha256(args.out),
                          counts=manifest["counts"], validation=manifest["data_validation"]), indent=2))


if __name__ == "__main__":
    main()
