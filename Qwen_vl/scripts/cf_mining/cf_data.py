"""Build auditable one-digit counterfactual pairs from real DocVQA pages.

The original and changed answer tokens are BOTH rendered into the same OCR
box, with the same font/background treatment.  The unmodified real page is
retained separately for natural-image evaluation.  This is a pilot generator:
OCR is only a localization proposal; downstream full-model verification must
still establish that both members answer the intended question correctly.

All OCR runs on CPU.  No model weights or system packages are changed.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

QWEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = QWEN_ROOT / "outputs" / "cf_mining"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
NUMERIC = re.compile(r"^[\s$€£(+-]*\d[\d,.]*[%)]?\s*$")


def numeric_key(value: str) -> str:
    """Keep decimal precision; ignore cosmetic currency/thousands wrappers."""
    return re.sub(r"[\s$€£,%()]", "", str(value).strip())


def parse_answers(value):
    try:
        values = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        values = [str(value)]
    return [str(v).strip() for v in values]


def stable_split(doc_id, seed=20261108):
    # All questions and pages of a source document stay in the same split.
    u = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()[:8], 16) % 10
    return "train" if u < 6 else "validation" if u < 8 else "test"


def ocr_image(image_path, cache_path, binary, tessdata, library_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    env["TESSDATA_PREFIX"] = tessdata
    env["LD_LIBRARY_PATH"] = library_path + ":" + env.get("LD_LIBRARY_PATH", "")
    p = subprocess.run([binary, str(image_path), "stdout", "-l", "eng", "tsv"],
                       capture_output=True, text=True, env=env, timeout=90)
    if p.returncode:
        return {"error": p.stderr[-500:], "words": []}
    words = []
    for row in csv.DictReader(io.StringIO(p.stdout), delimiter="\t"):
        text = row.get("text", "").strip()
        if not text or int(row["level"]) != 5:
            continue
        words.append(dict(text=text, confidence=float(row["conf"]),
                          bbox=[int(row[k]) for k in ("left", "top", "width", "height")]))
    result = {"image_path": str(image_path), "words": words}
    cache_path.write_text(json.dumps(result, ensure_ascii=False))
    return result


def changed_token(token, seed):
    digits = [i for i, c in enumerate(token) if c.isdigit()]
    # Prefer a non-leading digit, and ensure the rendered change is real.
    candidates = digits[1:] or digits
    i = candidates[seed % len(candidates)]
    digit = (int(token[i]) + 1 + (seed // 7) % 8) % 10
    if i == digits[0] and digit == 0:
        digit = 1
        if token[i] == "1":
            digit = 2
    new = token[:i] + str(digit) + token[i + 1:]
    assert sum(a != b for a, b in zip(token, new)) == 1
    return new, {"character_index": i, "old": token[i], "new": str(digit)}


def edited_answer_alias(answer, old_token, change):
    """Apply the same digit edit while retaining answer units/sign wrappers."""
    digit_number = sum(c.isdigit() for c in old_token[:change["character_index"]])
    digit_positions = [i for i, c in enumerate(answer) if c.isdigit()]
    if len(digit_positions) != sum(c.isdigit() for c in old_token):
        return None
    at = digit_positions[digit_number]
    if answer[at] != change["old"]:
        return None
    return answer[:at] + change["new"] + answer[at + 1:]


def render_pair(source, bbox, old, new, out_old, out_new, font_path):
    image = Image.open(source).convert("RGB")
    x, y, w, h = bbox
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(image.width, x + w + 1), min(image.height, y + h + 1)
    pixels = np.asarray(image)
    patch = pixels[y0:y1, x0:x1]
    # Border estimates scanned-paper color; darkest pixels estimate text ink.
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    bg = tuple(int(v) for v in np.median(border, axis=0))
    brightness = patch.mean(-1)
    ink_pixels = patch[brightness <= np.percentile(brightness, 15)]
    ink = tuple(int(v) for v in np.median(ink_pixels, axis=0))
    if np.mean(bg) - np.mean(ink) < 30:
        ink = (20, 20, 20)
    # Same glyph scale for both members, selected jointly to fit their box.
    font = ImageFont.truetype(font_path, 100)
    measurements = [font.getbbox(t) for t in (old, new)]
    maxw = max(b[2] - b[0] for b in measurements)
    maxh = max(b[3] - b[1] for b in measurements)
    scale = min(w / maxw, h / maxh)
    # Supersampling and resize reproduce subpixel text at original dimensions.
    for token, dest in ((old, out_old), (new, out_new)):
        b = font.getbbox(token)
        tw = max(1, round((b[2] - b[0]) * scale))
        th = max(1, round((b[3] - b[1]) * scale))
        layer = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), bg)
        ImageDraw.Draw(layer).text((-b[0], -b[1]), token, font=font, fill=ink)
        layer = layer.resize((tw, th), Image.Resampling.LANCZOS)
        member = image.copy()
        ImageDraw.Draw(member).rectangle((x0, y0, x1 - 1, y1 - 1), fill=bg)
        member.paste(layer, (x + (w - tw) // 2, y + (h - th) // 2))
        member.save(dest, compress_level=2)
    return dict(font=font_path, background=list(bg), ink=list(ink),
                image_size=list(image.size), patch_bbox=[x0, y0, x1, y1])


def build(args):
    out = Path(args.out).resolve()
    (out / "ocr").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    data = pd.read_csv(args.tsv, sep="\t", usecols=["question", "answer", "image_path", "index", "docId"])
    candidates = []
    for row_idx, row in data.iterrows():
        answers = parse_answers(row.answer)
        numeric = [a for a in answers if NUMERIC.fullmatch(a) and sum(c.isdigit() for c in a) >= 2]
        if not numeric:
            continue
        # One concrete numeric answer concept; case/currency variants are OK.
        keys = {numeric_key(a) for a in numeric}
        if len(keys) != 1:
            continue
        source = Path(args.image_dir) / row.image_path
        if not source.exists():
            continue
        key = next(iter(keys))
        if key in numeric_key(row.question):
            continue
        candidates.append(dict(row_idx=int(row_idx), question_id=int(row["index"]),
                               doc_id=int(row.docId), image_id=row.image_path,
                               question=row.question, answers=answers, answer_key=key,
                               original_answer=numeric[0], source_image=str(source)))
    # Deterministic shuffled scan avoids beginning-of-TSV source/style bias.
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[:args.max_scan]
    unique_images = {c["image_id"]: c["source_image"] for c in candidates}
    print(f"OCR {len(unique_images)} unique pages for {len(candidates)} numeric questions", flush=True)
    results = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ocr_image, source, out / "ocr" / (image_id + ".json"),
                               args.tesseract, args.tessdata, args.library_path): image_id
                   for image_id, source in unique_images.items()}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            image_id = futures[future]
            try:
                results[image_id] = future.result()
            except Exception as exc:
                results[image_id] = {"words": [], "error": str(exc)}
            if n % 25 == 0:
                print(f"OCR {n}/{len(unique_images)} {time.time() - t0:.1f}s", flush=True)
    records, rejected = [], {}
    for c in candidates:
        words = results[c["image_id"]]["words"]
        matches = [w for w in words if numeric_key(w["text"]) == c["answer_key"]
                   and NUMERIC.fullmatch(w["text"])]
        reason = None
        if len(matches) != 1:
            reason = "nonunique_or_missing_ocr_match"
        elif matches[0]["confidence"] < args.min_confidence:
            reason = "low_ocr_confidence"
        elif min(matches[0]["bbox"][2:]) < 6:
            reason = "box_too_small"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        word = matches[0]
        edited, change = changed_token(word["text"], c["question_id"] + args.seed)
        edited_answers = [edited_answer_alias(a, word["text"], change) for a in c["answers"]]
        edited_answers = [a for a in edited_answers if a is not None]
        edited_target = edited_answer_alias(c["original_answer"], word["text"], change)
        assert edited_target is not None and edited_answers
        # Existing occurrences of the new value could make an edited question ambiguous.
        if any(numeric_key(w["text"]) == numeric_key(edited) for w in words):
            rejected["edited_value_already_present"] = rejected.get("edited_value_already_present", 0) + 1
            continue
        pair_id = f"docvqa_{c['question_id']}_{c['row_idx']}"
        old_path = out / "images" / (pair_id + "_original.png")
        new_path = out / "images" / (pair_id + "_edited.png")
        if args.existing_only:
            if not old_path.exists() or not new_path.exists():
                continue
            # Snapshot already completed pairs while a long generation run saves
            # remaining pages.  No PNGs are rewritten by this read-only option.
            try:
                with Image.open(old_path) as image:
                    image.verify()
                with Image.open(new_path) as image:
                    image.verify()
            except (OSError, SyntaxError):
                # Another process may still be writing the latest pair.
                continue
            with Image.open(old_path) as image:
                render = dict(font=args.font, image_size=list(image.size),
                              existing_pair_snapshot=True)
        else:
            render = render_pair(c["source_image"], word["bbox"], word["text"], edited,
                                 old_path, new_path, args.font)
        invariant = []
        for sibling_idx, sibling in data[data.image_path == c["image_id"]].iterrows():
            ans = parse_answers(sibling.answer)
            if int(sibling_idx) != c["row_idx"] and all(c["answer_key"] not in numeric_key(a) for a in ans):
                if c["answer_key"] not in numeric_key(sibling.question):
                    invariant.append(dict(row_idx=int(sibling_idx), question_id=int(sibling["index"]),
                                          question=sibling.question, answers=ans))
        c.pop("answer_key")
        c.update(pair_id=pair_id, split=stable_split(c["doc_id"], args.seed),
                 dataset="DocVQA_VAL", original_image=str(old_path), edited_image=str(new_path),
                 edited_answer=re.sub(r"[\s$€£,]", "", edited_target),
                 edited_answers=edited_answers, bbox=word["bbox"], ocr_text=word["text"],
                 edited_text=edited, ocr_confidence=word["confidence"], changed_digit=change,
                 rendering=render, invariant_questions=invariant)
        records.append(c)
        if len(records) >= args.target_pairs:
            break
    counts = dict(scanned_questions=len(candidates), scanned_images=len(unique_images),
                  pairs=len(records), unique_documents=len({r["doc_id"] for r in records}),
                  unique_images=len({r["image_id"] for r in records}), rejected=rejected,
                  splits={s: sum(r["split"] == s for r in records) for s in ("train", "validation", "test")})
    # Assert all source images/documents are disjoint across groups.
    for field in ("image_id", "doc_id"):
        seen = {}
        for r in records:
            assert seen.setdefault(r[field], r["split"]) == r["split"]
    manifest = dict(schema_version=1, purpose="CPU-localized candidates; require full-model pair validation",
                    source_split="DocVQA_VAL (pilot partitions only; not benchmark training)",
                    config=vars(args), counts=counts, records=records)
    name = "manifest_partial.json" if args.existing_only else "manifest.json"
    temporary_manifest = out / (name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary_manifest.replace(out / name)
    print(json.dumps(counts, indent=2), flush=True)
    print(f"Manifest: {out / name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="/media/disk2/YZX/LMUData/DocVQA_VAL.tsv")
    ap.add_argument("--image-dir", default="/media/disk2/YZX/LMUData/images/DocVQA_VAL")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tesseract", default="/tmp/eadp_cf_tools/root/usr/bin/tesseract")
    ap.add_argument("--tessdata", default="/tmp/eadp_cf_tools/root/usr/share/tesseract-ocr/4.00/tessdata")
    ap.add_argument("--library-path", default="/tmp/eadp_cf_tools/root/usr/lib/x86_64-linux-gnu")
    ap.add_argument("--font", default=FONT)
    ap.add_argument("--max-scan", type=int, default=600)
    ap.add_argument("--target-pairs", type=int, default=180)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-confidence", type=float, default=55)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--existing-only", action="store_true")
    build(ap.parse_args())
