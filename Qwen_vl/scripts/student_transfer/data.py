"""Materialize immutable, unfiltered transfer benchmarks without loading a model.

The current student's fit/validation pages are used only for contamination
audits. All official rows remain in ``items``; subset memberships are metadata.
Images follow the local VLMEvalKit PIL decode/default-save convention.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

from PIL import Image, __version__ as PILLOW_VERSION

QWEN = Path(__file__).resolve().parents[2]
DATASETS = ('DocVQA_VAL', 'TextVQA_VAL', 'OCRBench')
EXPECTED_COUNTS = {'DocVQA_VAL': 5349, 'TextVQA_VAL': 5000, 'OCRBench': 1000}
DEFAULT_SOURCE_ROOT = Path('/media/disk2/YZX/LMUData')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert('RGB')
    return hashlib.sha256(struct.pack('>II', *rgb.size) + rgb.tobytes()).hexdigest()


def dhash64(image: Image.Image) -> int:
    pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def atomic_new(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
            stream.write('\n')
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _progress(message):
    print('[student-transfer data] ' + message, file=sys.stderr, flush=True)


def parse_answers(value: str, dataset: str, index: str) -> list[str]:
    answers = ast.literal_eval(value)
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
        raise ValueError(f'Invalid answer annotation {dataset}:{index}')
    return answers


def materialize_image(raw: bytes, suffix: str, output_dir: Path) -> dict:
    """Match VLMEvalKit, including its default JPEG re-encoding.

    Source and materialized hashes are both recorded. In particular, JPEG
    source bytes are not silently substituted for the PIL-saved input.
    """
    formats = Image.registered_extensions()
    if suffix not in formats:
        raise ValueError(f'Unsupported source image suffix: {suffix}')
    with Image.open(io.BytesIO(raw)) as image:
        source_pixel = pixel_sha256(image)
        size = list(image.size)
        perceptual_hash = dhash64(image)
        if image.mode in ('RGBA', 'P', 'LA'):
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format=formats[suffix])
    saved = buffer.getvalue()
    digest = hashlib.sha256(saved).hexdigest()
    path = output_dir / 'images' / (digest + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest:
            raise ValueError(f'Existing materialized image changed: {path}')
    else:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(saved)
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
    with Image.open(io.BytesIO(saved)) as image:
        saved_pixel = pixel_sha256(image)
    return dict(image_path=str(path.resolve()), image_sha256=digest,
                pixel_sha256=saved_pixel, source_image_sha256=hashlib.sha256(raw).hexdigest(),
                source_pixel_sha256=source_pixel, image_size=size,
                dhash64=f'{perceptual_hash:016x}')


def _training_references(source_run: Path):
    path = source_run / 'data/manifest.json'
    raw = json.loads(path.read_text(encoding='utf-8'))
    refs, seen = [], set()
    split_counts = {}
    for split in ('fit', 'val'):
        pages = raw['splits'][split]
        split_counts[split] = dict(pages=len(pages), questions=sum(len(p['questions']) for p in pages))
        for page in pages:
            image_path = Path(page['image_path'])
            if sha256(image_path) != page['image_sha256']:
                raise ValueError(f'Student {split} image changed: {image_path}')
            with Image.open(image_path) as image:
                pixels = pixel_sha256(image)
                if page.get('pixel_sha256') and pixels != page['pixel_sha256']:
                    raise ValueError(f'Student {split} pixels changed: {image_path}')
                ref = dict(split=split, parent_id=str(page['parent_id']),
                           page_id=str(page['page_id']), image_path=str(image_path),
                           image_sha256=page['image_sha256'], pixel_sha256=pixels,
                           image_size=list(image.size), dhash64=f'{dhash64(image):016x}')
            key = (split, ref['pixel_sha256'])
            if key in seen:
                raise ValueError('Duplicate source student fit/val page')
            refs.append(ref)
            seen.add(key)
    return refs, dict(manifest_path=str(path.resolve()), manifest_sha256=sha256(path),
                      split_counts=split_counts, references=refs)


def _history_coverage(history_root: Path, dataset: str):
    """Read identity columns only; never select by past predictions or scores."""
    prefix = history_root / 'eadp/Qwen3-VL-8B-EADP-256-a0.5-b2.0'
    path = prefix / f'Qwen3-VL-8B-EADP-256-a0.5-b2.0_{dataset}.xlsx'
    if not path.exists():
        return set(), dict(evidence_path=None, coverage='unknown',
                          note='No configured historical full-prediction identity file; false flags do not prove no exposure.')
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book.active.iter_rows(values_only=True)
        header = next(rows)
        column = list(header).index('index')
        identities = {str(int(row[column])) if isinstance(row[column], (int, float)) else str(row[column])
                      for row in rows if row[column] is not None}
    finally:
        book.close()
    return identities, dict(evidence_path=str(path.resolve()), evidence_sha256=sha256(path),
                           source_indices=sorted(identities), coverage='source index present in prior EADP prediction table',
                           note='Project observation, not current student training; no score-based selection.')


def _resolve_metadata(by_index: dict, index: str, stack=None):
    stack = set() if stack is None else stack
    if index not in by_index:
        raise ValueError(f'Unresolved image index reference {index}')
    value = by_index[index]
    if 'ref' not in value:
        return value
    if index in stack:
        raise ValueError(f'Cyclic image index reference {index}')
    resolved = _resolve_metadata(by_index, value['ref'], stack | {index})
    by_index[index] = resolved
    return resolved


def _mark_overlap(item: dict, refs: list[dict]):
    reasons, near = [], []
    source_pixels = {item['pixel_sha256'], item['source_pixel_sha256']}
    for ref in refs:
        exact = []
        if item['dataset'] == 'DocVQA_VAL' and item['parent_id'] == ref['parent_id']:
            exact.append('source_parent')
        if ref['pixel_sha256'] in source_pixels:
            exact.append('rgb_pixels')
        if ref['image_sha256'] in {item['image_sha256'], item['source_image_sha256']}:
            exact.append('image_bytes')
        if exact:
            reasons.append(dict(split=ref['split'], parent_id=ref['parent_id'],
                                page_id=ref['page_id'], matches=exact))
        width, height = item['image_size']
        rw, rh = ref['image_size']
        distance = (int(item['dhash64'], 16) ^ int(ref['dhash64'], 16)).bit_count()
        if distance <= 3 and abs((width / height) / (rw / rh) - 1) <= .03:
            near.append(dict(split=ref['split'], parent_id=ref['parent_id'],
                             page_id=ref['page_id'], dhash_hamming=distance))
    item['training_overlap'] = bool(reasons)
    item['training_overlap_reasons'] = reasons
    item['near_duplicate_suspected'] = bool(near)
    item['near_duplicate_matches'] = near


def _build_dataset(name, source_path, output_dir, refs, history_root):
    source_sha = sha256(source_path)
    history_ids, history = _history_coverage(history_root, name)
    rows, images = [], {}
    csv.field_size_limit(sys.maxsize)
    with source_path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'index', 'image', 'question', 'answer'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected benchmark TSV schema: {name}')
        for position, row in enumerate(reader):
            index = str(row['index'])
            if index in images:
                raise ValueError(f'Duplicate source index {name}:{index}')
            payload = row['image']
            suffix = Path(row.get('image_path') or f'{index}.jpg').suffix.lower()
            if len(payload) > 50:
                metadata = materialize_image(base64.b64decode(payload, validate=True), suffix, output_dir)
                metadata['image_source_index'] = index
                images[index] = metadata
            elif payload:
                images[index] = {'ref': payload}
            else:
                raise ValueError(f'Empty embedded image or reference at {name}:{index}')
            question = row['question']
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f'Empty original question {name}:{index}')
            rows.append(dict(dataset=name, item_id=f'{name}_{index}', source_id=index,
                             source_row=position, question=question,
                             answers=parse_answers(row['answer'], name, index),
                             category=row.get('category', ''),
                             source_image_path=row.get('image_path', ''),
                             parent_id=str(row.get('ucsf_document_id') or ''),
                             source_page_number=row.get('ucsf_document_page_no', ''),
                             history_exposed=index in history_ids))
            if (position + 1) % 500 == 0:
                _progress(f'{name}: {position + 1} original rows materialized')
    if len(rows) != EXPECTED_COUNTS[name]:
        raise ValueError(f'{name}: expected {EXPECTED_COUNTS[name]} complete rows, got {len(rows)}')
    for row in rows:
        row.update(_resolve_metadata(images, row['source_id']))
        if not row['parent_id']:
            # TextVQA and OCRBench TSVs have no trustworthy source-document ID.
            # Cluster exact source rasters across repeated question rows.
            row['parent_id'] = f"{name}:raster:{row['source_pixel_sha256']}"
        _mark_overlap(row, refs)
    if sha256(source_path) != source_sha:
        raise RuntimeError(f'Benchmark TSV changed during construction: {source_path}')
    clean = [r['item_id'] for r in rows if not r['training_overlap']]
    clean_near = [r['item_id'] for r in rows if not r['training_overlap'] and not r['near_duplicate_suspected']]
    novel = [r['item_id'] for r in rows if not r['training_overlap'] and not r['near_duplicate_suspected'] and not r['history_exposed']]
    return dict(items=rows, complete=True, source_tsv=str(source_path.resolve()),
                source_tsv_sha256=source_sha, num_items=len(rows),
                num_unique_images=len({r['pixel_sha256'] for r in rows}),
                num_unique_parents=len({r['parent_id'] for r in rows}),
                training_overlap_count=sum(r['training_overlap'] for r in rows),
                near_duplicate_suspected_count=sum(r['near_duplicate_suspected'] for r in rows),
                history_exposed_count=sum(r['history_exposed'] for r in rows),
                category_counts=dict(Counter(r['category'] for r in rows)),
                decontaminated_item_ids=clean, conservative_item_ids=clean_near,
                novel_item_ids=novel, historical_exposure=history)


def verify_manifest(manifest: dict):
    if manifest.get('schema') != 1 or not manifest.get('complete'):
        raise ValueError('Incomplete or unsupported transfer data manifest')
    provenance = manifest['provenance']
    for path, expected in provenance['source_files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Frozen data source changed: {path}')
    seen = {}
    for dataset in manifest['datasets'].values():
        for item in dataset['items']:
            path, expected = item['image_path'], item['image_sha256']
            if path in seen and seen[path] != expected:
                raise ValueError('Conflicting materialized input hashes')
            seen[path] = expected
    for path, expected in seen.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Frozen benchmark input changed: {path}')
    return manifest


def build_manifest(output_dir: Path, source_run: Path, datasets=DATASETS,
                   *, source_root=DEFAULT_SOURCE_ROOT, history_root=None) -> dict:
    output_dir, source_run = Path(output_dir).resolve(), Path(source_run).resolve()
    datasets = tuple(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in DATASETS for d in datasets):
        raise ValueError('Datasets must be unique supported benchmark names')
    source_root = Path(source_root).resolve()
    history_root = Path(history_root or QWEN / 'outputs').resolve()
    path = output_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if tuple(manifest['datasets']) != datasets or manifest['source_run'] != str(source_run):
            raise ValueError('Existing manifest belongs to another data configuration')
        if manifest['provenance']['source_root'] != str(source_root):
            raise ValueError('Existing manifest uses a different benchmark source root')
        return verify_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs, training = _training_references(source_run)
    result = {}
    for name in datasets:
        result[name] = _build_dataset(name, source_root / (name + '.tsv'), output_dir, refs, history_root)
        _progress(f'{name}: complete {result[name]["num_items"]}, '
                  f'training overlap {result[name]["training_overlap_count"]}, '
                  f'near-duplicate suspects {result[name]["near_duplicate_suspected_count"]}')
    sources = {str(Path(__file__).resolve()): sha256(Path(__file__)),
               training['manifest_path']: training['manifest_sha256']}
    sources.update({r['image_path']: r['image_sha256'] for r in refs})
    for dataset in result.values():
        sources[dataset['source_tsv']] = dataset['source_tsv_sha256']
        history = dataset['historical_exposure']
        if history['evidence_path']:
            sources[history['evidence_path']] = history['evidence_sha256']
    manifest = dict(schema=1, complete=True, source_run=str(source_run), datasets=result,
                    student_training=training,
                    provenance=dict(source_root=str(source_root), history_root=str(history_root),
                        source_files=sources, pillow_version=PILLOW_VERSION,
                        policy='Every official source row retained; no model prediction, answer value, or teacher-success filtering.',
                        image_materialization='VLMEvalKit: PIL open; RGBA/P/LA to RGB; default PIL save in original image_path extension, otherwise .jpg.',
                        pixel_hash='SHA256(big-endian uint32 width,height + PIL RGB pixels)',
                        exact_overlap='Current student fit or validation source parent (DocVQA only), source/materialized bytes, or source/materialized RGB raster.',
                        near_duplicate='64-bit grayscale dHash at 9x8, Hamming <=3, aspect ratio within 3%, against student fit+val pages; flags only, no official-row exclusion.',
                        subsets='decontaminated excludes exact training overlap; conservative additionally excludes dHash suspects; novel additionally excludes documented prior project prediction-table indices.',
                        limitations='Exact hashes miss recompression/cropping. dHash is a heuristic with false positives/negatives and does not certify all near duplicates absent. Prior project observation differs from student training. Base-model pretraining exposure is unknown.'))
    atomic_new(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', '--out', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    manifest = build_manifest(args.output_dir, args.source_run, args.datasets,
                              source_root=args.source_root, history_root=args.history_root)
    print(json.dumps({name: {k: value[k] for k in ('num_items', 'num_unique_images',
                      'num_unique_parents', 'training_overlap_count',
                      'near_duplicate_suspected_count', 'history_exposed_count')}
                      for name, value in manifest['datasets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
