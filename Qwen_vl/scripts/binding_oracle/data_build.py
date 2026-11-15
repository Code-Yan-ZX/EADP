"""CPU-only QA-grounded, two-value swaps on real DocVQA pages.

Candidates are selected using annotations/OCR/geometry only, before any
teacher or compressed-model prediction. Both boxes are rerendered in both
members with a single common font scale. Neither box is a gold field polygon:
value boxes are QA-grounded OCR matches; context boxes are Tesseract lines.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

QWEN = Path(__file__).resolve().parents[2]
OUT = QWEN / "outputs/binding_oracle/data"
NUMERIC = re.compile(r"^\s*[$€£]?\s*\d[\d,.]*\s*%?\s*$")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def key(value):
    value = re.sub(r"[\s$€£,%]", "", str(value))
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return value


def answers(value):
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        value = [str(value)]
    return [str(v).strip() for v in value]


def split_for(document, seed):
    value = int(hashlib.sha256(f"binding:{seed}:{document}".encode()).hexdigest()[:8], 16)
    return "dev" if value % 10 < 6 else "confirm"


def xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def projected_region(box, size, side=1024):
    width, height = size
    square = max(width, height)
    left, top = (square - width) // 2, (square - height) // 2
    scale = side / square
    x0, y0, x1, y1 = xyxy(box)
    edges = [(x0 + left) * scale, (y0 + top) * scale,
             (x1 + left) * scale, (y1 + top) * scale]
    active, centers = [], []
    for row in range(32):
        for col in range(32):
            token = [32 * col, 32 * row, 32 * (col + 1), 32 * (row + 1)]
            if overlaps(edges, token):
                active.append(row * 32 + col)
            cx, cy = 32 * col + 16, 32 * row + 16
            if edges[0] <= cx < edges[2] and edges[1] <= cy < edges[3]:
                centers.append(row * 32 + col)
    return dict(xyxy=edges, overlap_token_indices=active, center_token_indices=centers)


def ocr(source, cache, args):
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    if args.reuse_ocr:
        prior = Path(args.reuse_ocr) / cache.name
        if prior.exists():
            payload = json.loads(prior.read_text())
            if "lines" in payload:
                atomic_json(cache, payload)
                return payload
    env = dict(os.environ, OMP_THREAD_LIMIT="1", TESSDATA_PREFIX=args.tessdata)
    env["LD_LIBRARY_PATH"] = args.library_path + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run([args.tesseract, str(source), "stdout", "-l", "eng", "tsv"],
                            env=env, capture_output=True, text=True, timeout=90)
    if result.returncode:
        return dict(words=[], error=result.stderr[-500:])
    words, lines = [], {}
    for item in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
        line_id = ":".join(item[k] for k in ("page_num", "block_num", "par_num", "line_num"))
        box = [int(item[k]) for k in ("left", "top", "width", "height")]
        if item["level"] == "4":
            lines[line_id] = dict(bbox=box, words=[])
        text = (item.get("text") or "").strip()
        if item["level"] == "5" and text:
            word = dict(text=text, confidence=float(item["conf"]), bbox=box, line_id=line_id)
            words.append(word)
            lines.setdefault(line_id, dict(bbox=box, words=[]))["words"].append(text)
    payload = dict(source=str(source), words=words, lines=lines, engine="Tesseract 4.1.1 / eng / default PSM3")
    atomic_json(cache, payload)
    return payload


def numeric_questions(frame):
    result = defaultdict(list)
    for row_idx, row in frame.iterrows():
        accepted = answers(row.answer)
        numeric = [a for a in accepted if NUMERIC.fullmatch(a)]
        if not numeric or len({key(a) for a in numeric}) != 1:
            continue
        preferred = numeric[0]
        value_key = key(preferred)
        # Do not change an explicit numeric premise in a question.
        question_numbers = {key(n) for n in re.findall(r"\d[\d,.]*", row.question)}
        if value_key in question_numbers:
            continue
        result[row.image_path].append(dict(row_idx=int(row_idx), question_id=int(row["index"]),
                                          question=row.question, answers=accepted,
                                          answer=preferred, numeric_key=value_key,
                                          doc_id=int(row.docId), source_document_id=str(row.ucsf_document_id)))
    return result


def unit_kind(field, word):
    q = field["question"].lower()
    if "%" in q or "%" in word["text"] or any("%" in a for a in field["answers"]) or re.search(r"percent|percentage", q):
        return "percent"
    if (any(s in word["text"] for s in "$€£") or any(any(s in a for s in "$€£") for a in field["answers"])
            or re.search(r"\b(amount|cost|expense|expenses|income|price|salary|salaries|funds|profit|revenue|sales|budget|remuneration)\b", q)):
        return "currency"
    if re.search(r"\b(year|when|born|date)\b", q):
        return "year_or_date"
    if re.search(r"\b(code|serial|case|suite|box|extension|ext|page|figure)\b", q):
        return "identifier"
    return "number"


def render_quality(words, font_path=FONT):
    digits = [sum(c.isdigit() for c in word["text"]) for word in words]
    reasons = []
    if max(digits) / max(1, min(digits)) > 2:
        reasons.append("digit_count_ratio_above_2")
    if any(word["bbox"][3] > word["bbox"][2] and count <= 2 for word, count in zip(words, digits)):
        reasons.append("vertical_or_too_narrow_short_value_box")
    font = ImageFont.truetype(font_path, 100)
    glyphs = [font.getbbox(word["text"]) for word in words]
    scale = min(min(word["bbox"][2] / (bb[2] - bb[0]), word["bbox"][3] / (bb[3] - bb[1]))
                for word in words for bb in glyphs)
    ratio = min(max(1, round((bb[3] - bb[1]) * scale)) / word["bbox"][3]
                for word in words for bb in glyphs)
    if ratio < .6:
        reasons.append("rendered_height_retention_below_0.6")
    return reasons, dict(digit_counts=digits, digit_count_ratio=max(digits) / max(1, min(digits)),
                         minimum_rendered_height_retention=ratio, common_font_scale=scale)


def choose_pair(fields, payload, size, rejection_log=None):
    matched, seen = [], set()
    for field in fields:
        matches = [w for w in payload["words"] if NUMERIC.fullmatch(w["text"]) and key(w["text"]) == field["numeric_key"]]
        if len(matches) != 1:
            continue
        match = matches[0]
        # Provided DocVQA OCR uses null / "Low", not a numeric score. Keep
        # null explicit, reject Low, and never invent a numeric confidence.
        confidence_ok = (match.get("raw_confidence") != "Low"
                         if match.get("source_engine") == "provided DocVQA OCR"
                         else match["confidence"] >= 55)
        if not confidence_ok or min(match["bbox"][2:]) < 6:
            continue
        word = matches[0]
        if tuple(word["bbox"]) in seen:
            continue
        seen.add(tuple(word["bbox"]))
        matched.append((field, word))
    proposals = []
    for (a, wa), (b, wb) in itertools.combinations(matched, 2):
        if a["numeric_key"] == b["numeric_key"] or overlaps(xyxy(wa["bbox"]), xyxy(wb["bbox"])):
            continue
        # Percentage and nonpercentage values are not interchangeable units.
        ua, ub = unit_kind(a, wa), unit_kind(b, wb)
        reasons, metrics = render_quality([wa, wb])
        if ua != ub:
            reasons.append("incompatible_annotated_unit_categories")
        if reasons:
            if rejection_log is not None:
                rejection_log.append(dict(question_ids=[a["question_id"], b["question_id"]],
                                          texts=[wa["text"], wb["text"]], boxes=[wa["bbox"], wb["bbox"]],
                                          reasons=reasons, quality_metrics=metrics, unit_categories=[ua, ub]))
            continue
        # Source annotations remain valid questions after swapping the values.
        qa_numbers = {key(n) for n in re.findall(r"\d[\d,.]*", a["question"])}
        qb_numbers = {key(n) for n in re.findall(r"\d[\d,.]*", b["question"])}
        if b["numeric_key"] in qa_numbers or a["numeric_key"] in qb_numbers:
            continue
        la, lb = payload["lines"][wa["line_id"]]["bbox"], payload["lines"][wb["line_id"]]["bbox"]
        line_disjoint = not overlaps(xyxy(la), xyxy(lb))
        ma, mb = projected_region(wa["bbox"], size), projected_region(wb["bbox"], size)
        mask_disjoint = not bool(set(ma["overlap_token_indices"]) & set(mb["overlap_token_indices"]))
        ta = set(re.findall(r"[a-z]+", a["question"].lower()))
        tb = set(re.findall(r"[a-z]+", b["question"].lower()))
        similarity = len(ta & tb) / max(1, len(ta | tb))
        digits = [sum(c.isdigit() for c in w["text"]) for w in (wa, wb)]
        score = (10 * mask_disjoint + 4 * line_disjoint + 2 * (ua == ub)
                 + 2 * (digits[0] == digits[1]) + min(min(digits), 4) + similarity)
        proposals.append((score, a, wa, b, wb))
    return max(proposals, key=lambda x: (x[0], -x[1]["question_id"], -x[3]["question_id"])) if proposals else None


def render(source, words, old_path, swapped_path, font_path):
    image = Image.open(source).convert("RGB")
    pixels = np.asarray(image)
    patches = []
    for word in words:
        x, y, w, h = word["bbox"]
        region = [max(0, x - 1), max(0, y - 1), min(image.width, x + w + 1), min(image.height, y + h + 1)]
        x0, y0, x1, y1 = region
        patch = pixels[y0:y1, x0:x1]
        border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
        bg = tuple(int(v) for v in np.median(border, axis=0))
        brightness = patch.mean(-1)
        fg = tuple(int(v) for v in np.median(patch[brightness <= np.percentile(brightness, 15)], axis=0))
        if np.mean(bg) - np.mean(fg) < 30:
            fg = (20, 20, 20)
        patches.append(dict(region=region, background=bg, ink=fg))
    font = ImageFont.truetype(font_path, 100)
    tokens = [word["text"] for word in words]
    glyph_boxes = [font.getbbox(text) for text in tokens]
    scales = [min(word["bbox"][2] / (bb[2] - bb[0]), word["bbox"][3] / (bb[3] - bb[1]))
              for word in words for bb in glyph_boxes]
    scale = min(scales)
    for path, values in ((old_path, tokens), (swapped_path, list(reversed(tokens)))):
        member = image.copy()
        draw = ImageDraw.Draw(member)
        for word, patch, token in zip(words, patches, values):
            x, y, w, h = word["bbox"]
            bb = font.getbbox(token)
            tw, th = max(1, round((bb[2] - bb[0]) * scale)), max(1, round((bb[3] - bb[1]) * scale))
            glyph = Image.new("RGB", (bb[2] - bb[0], bb[3] - bb[1]), patch["background"])
            ImageDraw.Draw(glyph).text((-bb[0], -bb[1]), token, font=font, fill=patch["ink"])
            glyph = glyph.resize((tw, th), Image.Resampling.LANCZOS)
            x0, y0, x1, y1 = patch["region"]
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=patch["background"])
            member.paste(glyph, (x + (w - tw) // 2, y + (h - th) // 2))
        member.save(path, compress_level=2)
    return dict(font=font_path, common_font_scale=scale, patches=patches,
                all_boxes_rerendered_in_both_members=True, label_boxes_untouched=True)


def record_for(image_id, fields, payload, frame, args, rejection_log=None):
    source = Path(args.image_dir) / image_id
    with Image.open(source) as im:
        size = list(im.size)
    proposal = choose_pair(fields, payload, size, rejection_log)
    if proposal is None:
        return None
    _, a, wa, b, wb = proposal
    pair_id = f"swap_{a['question_id']}_{b['question_id']}"
    images = Path(args.out) / "images"
    old_path, new_path = images / (pair_id + "_original.png"), images / (pair_id + "_swapped.png")
    rendering = render(source, [wa, wb], old_path, new_path, args.font)
    result_fields = []
    for i, (own, word, other) in enumerate(((a, wa, b), (b, wb, a))):
        line = payload["lines"][word["line_id"]]
        field = {k: own[k] for k in ("row_idx", "question_id", "question")}
        field.update(field_id=f"f{i}", original_answer=own["answer"], swapped_answer=other["answer"],
                     edited_answer=other["answer"], answers_original=own["answers"], answers_swapped=other["answers"],
                     answers=own["answers"], value_bbox=word["bbox"], context_line_bbox=line["bbox"],
                     ocr_text=word["text"], ocr_confidence=word["confidence"], ocr_line_id=word["line_id"],
                     ocr_line_text=" ".join(line["words"]),
                     value_region_1024=projected_region(word["bbox"], size),
                     context_line_region_1024=projected_region(line["bbox"], size))
        result_fields.append(field)
    width, height = size
    square = max(size)
    source_qa = [dict(question_id=int(row["index"]), row_idx=int(idx), question=row.question, answers=answers(row.answer))
                 for idx, row in frame[frame.image_path == image_id].iterrows()]
    line_disjoint = not overlaps(xyxy(result_fields[0]["context_line_bbox"]), xyxy(result_fields[1]["context_line_bbox"]))
    value_disjoint = not bool(set(result_fields[0]["value_region_1024"]["overlap_token_indices"]) & set(result_fields[1]["value_region_1024"]["overlap_token_indices"]))
    return dict(pair_id=pair_id, dataset="DocVQA_VAL", source_document_id=a["source_document_id"], doc_id=a["doc_id"],
                image_id=image_id, source_image=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_image=str(old_path.resolve()), swapped_image=str(new_path.resolve()), edited_image=str(new_path.resolve()),
                source_image_size=size, split=split_for(a["source_document_id"], args.seed), fields=result_fields,
                source_qa=source_qa, original_numeric_multiset=sorted([wa["text"], wb["text"]]),
                swapped_numeric_multiset=sorted([wb["text"], wa["text"]]),
                rendering=rendering, eligible_for_line_oracle=line_disjoint, value_overlap_masks_disjoint=value_disjoint,
                render_quality=render_quality([wa, wb], args.font)[1],
                source_to_letterbox1024=dict(square_side=square, pad_left=(square - width) // 2,
                                             pad_top=(square - height) // 2, scale=1024 / square,
                                             box_convention="pixel-edge xywh source; projected xyxy edges; no expansion",
                                             token_grid=[32, 32], token_size=[32, 32],
                                             overlap_definition="positive intersection area with token square; raster index row*32+col",
                                             center_definition="token center inside half-open projected region; can legitimately be empty"),
                provenance=dict(qa_source=args.tsv, localization="unique Tesseract numeric token matches of existing QA answers",
                                value_region_status="QA-grounded OCR value box; not a gold semantic field boundary",
                                context_region_status="Tesseract OCR line proposal; not a gold semantic field boundary",
                                selection="annotation/OCR/geometry only; no full or compressed model prediction used",
                                teacher_status="not_run; must verify two questions on both image members"))


def montage(records, path):
    records = records[:12]
    panel = Image.new("RGB", (1200, 220 * len(records)), (248, 248, 248))
    draw = ImageDraw.Draw(panel)
    for i, record in enumerate(records):
        fields = record["fields"]
        title = f"{record['pair_id']} {record['split']}  {fields[0]['original_answer']} <-> {fields[1]['original_answer']}"
        draw.text((6, i * 220 + 3), title, fill="black")
        for j, f in enumerate(fields):
            draw.text((6, i * 220 + 20 + j * 15), f"f{j}: {f['question'][:150]}", fill="black")
        for k, side in enumerate(("source_image", "original_image", "swapped_image")):
            with Image.open(record[side]) as image:
                for j, field in enumerate(fields):
                    x, y, w, h = field["value_bbox"]
                    patch = image.crop((max(0, x - 60), max(0, y - 20), min(image.width, x + w + 60), min(image.height, y + h + 20)))
                    patch.thumbnail((390, 72))
                    panel.paste(patch, (k * 400 + 4, i * 220 + 63 + j * 77))
    panel.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="/media/disk2/YZX/LMUData/DocVQA_VAL.tsv")
    ap.add_argument("--image-dir", default="/media/disk2/YZX/LMUData/images/DocVQA_VAL")
    ap.add_argument("--prior-manifest", default=str(QWEN / "outputs/cf_mining/manifest.json"))
    ap.add_argument("--reuse-ocr", default=str(OUT / "ocr"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--target", type=int, default=60)
    ap.add_argument("--preview", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--font", default=FONT)
    ap.add_argument("--tesseract", default="/tmp/eadp_cf_tools/root/usr/bin/tesseract")
    ap.add_argument("--tessdata", default="/tmp/eadp_cf_tools/root/usr/share/tesseract-ocr/4.00/tessdata")
    ap.add_argument("--library-path", default="/tmp/eadp_cf_tools/root/usr/lib/x86_64-linux-gnu")
    args = ap.parse_args()
    out = Path(args.out)
    for directory in (out, out / "images", out / "ocr"):
        directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.tsv, sep="\t", usecols=["index", "docId", "ucsf_document_id", "image_path", "question", "answer"])
    prior = json.loads(Path(args.prior_manifest).read_text())
    excluded_images = {row["image_id"] for row in prior["records"]}
    excluded_documents = set(str(x) for x in frame[frame.image_path.isin(excluded_images)].ucsf_document_id)
    eligible = frame[~frame.ucsf_document_id.astype(str).isin(excluded_documents)]
    questions = numeric_questions(eligible)
    questions = {image: rows for image, rows in questions.items() if len({r['numeric_key'] for r in rows}) >= 2 and (Path(args.image_dir) / image).exists()}
    rng = random.Random(args.seed)
    image_ids = sorted(questions)
    rng.shuffle(image_ids)
    # Prior OCR is localization-only. It contains no model predictions.
    first, rest = [], []
    for image_id in image_ids:
        cached = QWEN / "outputs/cf_mining/ocr" / (image_id + ".json")
        if cached.exists():
            payload = json.loads(cached.read_text())
            pseudo = dict(payload, lines={})
            for number, word in enumerate(pseudo["words"]):
                word["line_id"] = str(number)
                pseudo["lines"][str(number)] = dict(bbox=word["bbox"], words=[word["text"]])
            with Image.open(Path(args.image_dir) / image_id) as image:
                if choose_pair(questions[image_id], pseudo, image.size):
                    first.append(image_id)
                else:
                    rest.append(image_id)
        else:
            rest.append(image_id)
    print(f"fresh source documents excluded={len(excluded_documents)}; potential pages={len(image_ids)}; preview localization pages={len(first)}", flush=True)
    records, used_pages, failures, quality_rejections = [], set(), [], []
    started = time.time()
    provenance = dict(config=vars(args), source_split="DocVQA_VAL internal source-document partitions; not official train/test",
                      excluded_prior_source_documents=sorted(excluded_documents), selection_uses_model_outputs=False,
                      region_warning="annotated-value/OCR-line oracles, never gold semantic field polygons")

    def save(name, complete):
        seen = {}
        for record in records:
            assert record["source_document_id"] not in excluded_documents
            assert seen.setdefault(record["source_document_id"], record["split"]) == record["split"]
            assert record["original_numeric_multiset"] == record["swapped_numeric_multiset"]
        summary = dict(pairs=len(records), unique_pages=len({r['image_id'] for r in records}),
                       unique_documents=len({r['source_document_id'] for r in records}),
                       splits=dict(Counter(r['split'] for r in records)),
                       line_disjoint_pairs=sum(r['eligible_for_line_oracle'] for r in records),
                       value_token_masks_disjoint=sum(r['value_overlap_masks_disjoint'] for r in records),
                       pages_ocr_attempted=len(used_pages), wall_seconds=time.time() - started)
        atomic_json(out / name, dict(schema_version=2, complete=complete, provenance=provenance, counts=summary,
                                    records=records, exclusions=failures, candidate_quality_rejections=quality_rejections))

    for phase in (first, rest):
        payloads = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(ocr, Path(args.image_dir) / image_id, out / "ocr" / (image_id + ".json"), args): image_id for image_id in phase}
            for n, future in enumerate(as_completed(futures), 1):
                image_id = futures[future]
                try:
                    payloads[image_id] = future.result()
                except Exception as exc:
                    payloads[image_id] = dict(words=[], error=str(exc))
                if n % 40 == 0:
                    print(f"OCR {n}/{len(phase)}", flush=True)
        for image_id in phase:
            if len(records) >= args.target:
                break
            used_pages.add(image_id)
            row = record_for(image_id, questions[image_id], payloads[image_id], frame, args, quality_rejections)
            if row:
                records.append(row)
                print(f"[pair {len(records)}] {row['pair_id']} {row['image_id']}", flush=True)
                if len(records) == args.preview:
                    save("manifest_preview.json", False)
                    montage(records, out / "montage_preview.png")
                    print(f"PREVIEW READY: {out / 'manifest_preview.json'}", flush=True)
            else:
                failures.append(dict(image_id=image_id, reason="no two distinct unique confident numeric OCR matches with nonoverlapping boxes and compatible units"))
        if len(records) >= args.target:
            break
    save("manifest.json", True)
    montage(records, out / "montage_preview.png")
    print(f"FINAL READY: {out / 'manifest.json'} pairs={len(records)}", flush=True)


if __name__ == "__main__":
    main()
