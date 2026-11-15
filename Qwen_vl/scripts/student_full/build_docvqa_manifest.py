"""Freeze all local VLMEvalKit DocVQA_VAL rows and audit student data overlap.

Run from any directory. No model, benchmark predictions or answer-based filtering.
"""
from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv')
DATA = ROOT / 'Qwen_vl/outputs/student_distill/data'
OUT = ROOT / 'Qwen_vl/outputs/student_full'
SCRIPT = Path(__file__).resolve()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_hash(raw):
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert('RGB')
        h = hashlib.sha256(struct.pack('>II', *im.size) + im.tobytes()).hexdigest()
        return h, list(im.size)


def dump_new(path, value):
    if path.exists():
        raise FileExistsError(f'refusing to overwrite {path}')
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as f:
            json.dump(value, f, ensure_ascii=False, indent=1)
            f.write('\n')
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'docvqa_val_manifest.json'
    audit_path = OUT / 'docvqa_val_data_audit.json'
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError('Output already exists; preserve frozen artifacts.')
    csv.field_size_limit(sys.maxsize)
    source_sha_before = sha_file(SRC)
    by_idx, rows, pixel_by_bytes = {}, [], {}
    with open(SRC, encoding='utf8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        expected = {'question', 'answer', 'image', 'index', 'image_path', 'ucsf_document_id', 'ucsf_document_page_no'}
        if not expected <= set(reader.fieldnames or []):
            raise ValueError(f'unexpected TSV schema: {reader.fieldnames}')
        for pos, r in enumerate(reader):
            idx = str(r['index'])
            if idx in by_idx:
                raise ValueError(f'duplicate source index {idx}')
            data = r['image']
            if len(data) > 50:
                raw = base64.b64decode(data, validate=True)
                sha = hashlib.sha256(raw).hexdigest()
                if sha not in pixel_by_bytes:
                    pixel_by_bytes[sha] = pixel_hash(raw)
                pix, size = pixel_by_bytes[sha]
                value = {'image_sha256': sha, 'pixel_sha256_rgb_dimensions': pix, 'image_size': size}
            else:
                value = {'ref': data}
            by_idx[idx] = value
            answers = ast.literal_eval(r['answer'])
            if not isinstance(answers, list) or not all(isinstance(a, str) for a in answers):
                raise ValueError(f'bad answer list at source row {pos}')
            rows.append({'source_row': pos, 'index': idx, 'parent_id': r['ucsf_document_id'],
                         'image_path': r['image_path'], 'source_page_number': r['ucsf_document_page_no'],
                         'question': r['question'], 'answers': answers})
    for row in rows:
        key = row['index']
        seen = set()
        while 'ref' in by_idx[key]:
            if key in seen:
                raise ValueError(f'cyclic image reference {key}')
            seen.add(key)
            key = by_idx[key]['ref']
            if key not in by_idx:
                raise ValueError(f'unresolved image reference {key}')
        row.update(by_idx[key])
        row['image_source_index'] = key
    groups = defaultdict(list)
    for row in rows:
        groups[row['image_path']].append(row)
    inconsistent = {p: sorted({r['pixel_sha256_rgb_dimensions'] for r in rr})
                    for p, rr in groups.items() if len({r['pixel_sha256_rgb_dimensions'] for r in rr}) != 1}
    if inconsistent:
        raise ValueError(f'image_path mapped to different pixel arrays: {inconsistent}')
    if len(rows) != 5349:
        raise ValueError(f'expected complete 5349 rows, got {len(rows)}')
    val_parents = {r['parent_id'] for r in rows}
    val_shas = {r['image_sha256'] for r in rows}
    val_pixels = {r['pixel_sha256_rgb_dimensions'] for r in rows}
    val_ids = {r['index'] for r in rows}
    split_audit = {}
    for role in ('fit', 'val', 'test', 'dev'):
        p = DATA / f'{role}_manifest.json'
        src = json.loads(p.read_text(encoding='utf8'))['records']
        train_pixels = set()
        for rec in src:
            img = Path(rec['image_path'])
            raw = img.read_bytes()
            if hashlib.sha256(raw).hexdigest() != rec['image_sha256']:
                raise ValueError(f'{role} image SHA changed: {img}')
            train_pixels.add(pixel_hash(raw)[0])
        split_audit[role] = {
            'manifest_path': str(p), 'manifest_sha256': sha_file(p), 'count': len(src),
            'overlap_parent_ids': sorted(val_parents & {str(x['parent_id']) for x in src}),
            'overlap_image_sha256': sorted(val_shas & {x['image_sha256'] for x in src}),
            'overlap_rgb_pixel_sha256': sorted(val_pixels & train_pixels),
            'overlap_question_ids': sorted(val_ids & {str(x['question_id']) for x in src}),
        }
    source_sha_after = sha_file(SRC)
    if source_sha_before != source_sha_after:
        raise RuntimeError('source TSV changed during manifest creation')
    prov = {'source_tsv': str(SRC), 'source_tsv_sha256': source_sha_before,
            'builder_path': str(SCRIPT), 'builder_sha256': sha_file(SCRIPT),
            'image_sha256_semantics': 'decoded source image bytes; reference rows inherit their indexed source image',
            'pixel_sha256_semantics': 'SHA256(big-endian uint32 width,height followed by PIL RGB bytes)',
            'policy': 'all source rows, no model- or answer-based filtering',
            'note': 'exact pixels and byte hashes do not detect all re-encoding, cropping, or pretraining exposure'}
    manifest = {'schema_version': 1, 'complete': True, 'provenance': prov, 'records': rows}
    dump_new(manifest_path, manifest)
    audit = {'complete': True, 'manifest_path': str(manifest_path), 'manifest_sha256': sha_file(manifest_path),
             'provenance': prov, 'num_rows': len(rows), 'unique_indices': len(by_idx),
             'unique_parent_ids': len(val_parents), 'unique_image_paths': len(groups),
             'unique_image_byte_sha256': len(val_shas), 'unique_rgb_pixel_sha256': len(val_pixels),
             'inconsistent_image_path_pixels': inconsistent, 'all_image_references_resolved': True,
             'student_data_overlap': split_audit}
    dump_new(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
