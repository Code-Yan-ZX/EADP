"""Freeze every local row of ten additional benchmarks, without model loading.

The existing DocVQA-trained students are never adapted here.  Original answer,
option, category, and grouping fields are retained for deterministic scoring.
HallusionBench's image-free control questions remain genuine text-only inputs.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import string
import sys

from PIL import __version__ as PILLOW_VERSION
from pandas import __version__ as PANDAS_VERSION
from pandas._libs.parsers import STR_NA_VALUES

from scripts.student_transfer import data as transfer_data
from scripts.student_transfer.data import (
    DEFAULT_SOURCE_ROOT, QWEN, _history_coverage, _mark_overlap,
    _resolve_metadata, _training_references, atomic_new, dhash64,
    materialize_image, pixel_sha256, sha256,
)

DATASETS = ('AI2D_TEST', 'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11',
            'MMStar', 'RealWorldQA', 'ChartQA_TEST', 'HallusionBench',
            'MME', 'InfoVQA_VAL', 'POPE')
EXPECTED_COUNTS = dict(zip(DATASETS, (3088, 4876, 4876, 1500, 765,
                                     2500, 1129, 2374, 2801, 5127)))
MCQ_DATASETS = frozenset(DATASETS[:5])
IMAGE_FIELDS = ('image_path', 'image_sha256', 'pixel_sha256',
                'source_image_sha256', 'source_pixel_sha256', 'image_size',
                'dhash64', 'image_source_index')
PUBLIC_METADATA_POLICY = ('Public options/hint match current pandas read_csv default '
                          'NA semantics, including nonempty NA/None strings. '
                          'Raw metadata, questions, answers and images remain unchanged.')


def _progress(message):
    print('[student-broad data] ' + message, file=sys.stderr, flush=True)


def parse_answers(value: str, dataset: str, index: str) -> list[str]:
    """Preserve literal lists and plain scalar labels/numbers without eval."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Empty answer annotation {dataset}:{index}')
    value = value.strip()
    if dataset == 'ChartQA_TEST':
        # VLMEvalKit treats only successfully parsed lists as alternatives.
        # "[Gambia, Niger]" is one literal answer; "[2014, 2016]" is a
        # numeric list, whose elements relaxed_correctness casts to strings.
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]
        if isinstance(parsed, list):
            if not parsed:
                raise ValueError(f'Invalid answer annotation {dataset}:{index}')
            return [str(answer) for answer in parsed]
        return [value]
    if value.startswith(('[', '(', '"', "'")):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f'Invalid answer annotation {dataset}:{index}') from error
        if isinstance(parsed, str):
            answers = [parsed]
        elif isinstance(parsed, (list, tuple)):
            answers = list(parsed)
        else:
            raise ValueError(f'Invalid answer annotation {dataset}:{index}')
    else:
        answers = [value]
    if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
        raise ValueError(f'Invalid answer annotation {dataset}:{index}')
    return answers


def _row_metadata(name: str, row: dict, position: int, history_ids: set) -> dict:
    index = str(row['index'])
    if not row['question'].strip():
        raise ValueError(f'Empty original question {name}:{index}')
    metadata = {k: v for k, v in row.items() if k != 'image'}
    answers = parse_answers(row['answer'], name, index)
    options = {letter: row[letter] for letter in string.ascii_uppercase
               if row.get(letter, '') not in STR_NA_VALUES}
    warnings = []
    if name in MCQ_DATASETS:
        for answer in answers:
            if answer in row and row[answer] in STR_NA_VALUES:
                # The released TSV has two empty correct A options. Keep the
                # raw blank and answer in metadata; VLMEvalKit's default NaN
                # semantics omit empty choices from the rendered options.
                warnings.append(('empty_correct_option:' if not row[answer]
                                 else 'missing_correct_option:') + answer)
    if name in MCQ_DATASETS and (not options or any(
            a not in options and not any(w.endswith(':' + a) for w in warnings)
            for a in answers)):
        raise ValueError(f'Missing MCQ option or invalid label {name}:{index}')
    item = dict(dataset=name, item_id=f'{name}_{index}', source_id=index,
                source_index=index, index=index, source_row=position,
                question=row['question'], answers=answers, options=options,
                hint=row.get('hint', '') if row.get('hint', '') not in STR_NA_VALUES else '',
                category=row.get('category', ''),
                source_image_path=row.get('image_path', ''),
                parent_id='', source_page_number='',
                history_exposed=index in history_ids, metadata=metadata,
                annotation_warnings=warnings)
    for key in ('l2-category', 'l2_category', 'split', 'abcLabel', 'bench',
                'answer_type', 'evidence', 'operation/reasoning'):
        if key in row:
            item[key] = row[key]
    if name.startswith('MMBench_'):
        item['g_index'] = str(int(index) % 1_000_000)
    elif name == 'MME':
        item['mme_image_group'] = row['category'] + ':' + row['image_path']
    elif name == 'HallusionBench':
        parts = index.split('_')
        if len(parts) != 6:
            raise ValueError(f'Unexpected HallusionBench index {index}')
        item.update(set_id=parts[3], figure_id=parts[4], question_id=parts[5])
    return item


def _finish_item(item: dict, images: dict, refs: list[dict]):
    item.update(_resolve_metadata(images, item['source_id']))
    if not item['has_image']:
        # Group a figure's associated yes/no controls together as in the
        # Hallusion official aggregate; no synthetic image is introduced.
        item['parent_id'] = (f"HallusionBench:{item['l2-category']}:"
                             f"set:{item['set_id']}")
        item.update(training_overlap=False, training_overlap_reasons=[],
                    near_duplicate_suspected=False, near_duplicate_matches=[])
        return
    if item['dataset'] == 'HallusionBench':
        item['parent_id'] = (f"HallusionBench:{item['l2-category']}:"
                             f"set:{item['set_id']}")
    elif item['dataset'] == 'MME':
        item['parent_id'] = 'MME:' + item['mme_image_group']
    else:
        # Clustering exact source rasters keeps repeated prompts/options on
        # the same image together; MMBench circular question IDs also survive.
        item['parent_id'] = f"{item['dataset']}:raster:{item['source_pixel_sha256']}"
    _mark_overlap(item, refs)


def _build_dataset(name, source_path, output_dir, refs, history_root):
    source_sha = sha256(source_path)
    history_ids, history = _history_coverage(history_root, name)
    rows, images = [], {}
    # Deduplicate identical embedded rasters before expensive PIL work.
    # The same payload saved with different extensions is a different input.
    embedded = {}
    csv.field_size_limit(sys.maxsize)
    with source_path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'index', 'image', 'question', 'answer'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected benchmark TSV schema: {name}')
        source_fields = list(reader.fieldnames)
        for position, row in enumerate(reader):
            index = str(row['index'])
            if index in images:
                raise ValueError(f'Duplicate source index {name}:{index}')
            payload = row['image']
            if len(payload) > 64:
                suffix = Path(row.get('image_path') or f'{index}.jpg').suffix.lower()
                key = (hashlib.sha256(payload.encode('ascii')).hexdigest(), suffix)
                if key not in embedded:
                    embedded[key] = materialize_image(
                        base64.b64decode(payload, validate=True), suffix, output_dir)
                images[index] = dict(embedded[key], has_image=True, image_source_index=index)
            elif payload:
                images[index] = {'ref': payload}
            elif name == 'HallusionBench' and not row.get('image_path'):
                images[index] = dict.fromkeys(IMAGE_FIELDS)
                images[index]['has_image'] = False
            else:
                raise ValueError(f'Empty image outside official image-free controls: {name}:{index}')
            rows.append(_row_metadata(name, row, position, history_ids))
            if (position + 1) % 500 == 0:
                _progress(f'{name}: {position + 1} original rows materialized')
    if len(rows) != EXPECTED_COUNTS[name]:
        raise ValueError(f'{name}: expected {EXPECTED_COUNTS[name]} complete rows, got {len(rows)}')
    # Repeated image references get the original anchor's source index.
    for item in rows:
        _finish_item(item, images, refs)
    if sha256(source_path) != source_sha:
        raise RuntimeError(f'Benchmark TSV changed during construction: {source_path}')
    clean = [r['item_id'] for r in rows if not r['training_overlap']]
    conservative = [r['item_id'] for r in rows
                    if not r['training_overlap'] and not r['near_duplicate_suspected']]
    conservative_set = set(conservative)
    return dict(items=rows, complete=True, source_tsv=str(source_path.resolve()),
                source_tsv_sha256=source_sha, source_fields=source_fields,
                num_items=len(rows), num_image_items=sum(r['has_image'] for r in rows),
                num_text_only_items=sum(not r['has_image'] for r in rows),
                num_unique_images=len({r['pixel_sha256'] for r in rows if r['has_image']}),
                num_unique_parents=len({r['parent_id'] for r in rows}),
                training_overlap_count=sum(r['training_overlap'] for r in rows),
                near_duplicate_suspected_count=sum(r['near_duplicate_suspected'] for r in rows),
                history_exposed_count=sum(r['history_exposed'] for r in rows),
                annotation_warning_count=sum(bool(r['annotation_warnings']) for r in rows),
                category_counts=dict(Counter(r['category'] for r in rows)),
                decontaminated_item_ids=clean, conservative_item_ids=conservative,
                novel_item_ids=[r['item_id'] for r in rows if r['item_id'] in conservative_set
                                and not r['history_exposed']]
                    if history['coverage'] != 'unknown' else [],
                historical_exposure=history)


def verify_manifest(manifest: dict):
    if manifest.get('schema') != 1 or not manifest.get('complete'):
        raise ValueError('Incomplete or unsupported broad benchmark manifest')
    for path, expected in manifest['provenance']['source_files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Frozen data source changed: {path}')
    seen = {}
    for dataset in manifest['datasets'].values():
        for item in dataset['items']:
            if not item['has_image']:
                if item['dataset'] != 'HallusionBench' or any(item[k] is not None for k in IMAGE_FIELDS):
                    raise ValueError('Invalid original image-free control')
                continue
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
        raise ValueError('Datasets must be unique supported benchmark names; local aliases are not new datasets')
    source_root = Path(source_root).resolve()
    history_root = Path(history_root or QWEN / 'outputs').resolve()
    path = output_dir / 'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (tuple(manifest['datasets']) != datasets or manifest['source_run'] != str(source_run)
                or manifest['provenance']['source_root'] != str(source_root)
                or manifest['provenance']['history_root'] != str(history_root)):
            raise ValueError('Existing manifest belongs to another data configuration')
        return verify_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    refs, training = _training_references(source_run)
    result = {}
    for name in datasets:
        result[name] = _build_dataset(name, source_root / (name + '.tsv'), output_dir, refs, history_root)
        _progress(f'{name}: complete {result[name]["num_items"]}; '
                  f'no image {result[name]["num_text_only_items"]}; '
                  f'exact fit/val overlap {result[name]["training_overlap_count"]}; '
                  f'near suspects {result[name]["near_duplicate_suspected_count"]}')
    sources = {str(Path(__file__).resolve()): sha256(Path(__file__)),
               str(Path(transfer_data.__file__).resolve()): sha256(Path(transfer_data.__file__)),
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
                        pandas_version=PANDAS_VERSION,
                        pandas_default_na_values=sorted(STR_NA_VALUES),
                        public_metadata_policy=PUBLIC_METADATA_POLICY,
                        policy='All original TSV rows, hints, answers and non-image metadata retained; public options omit empty source cells as VLMEvalKit default NaN prompts do. Two MMStar empty correct A options are preserved in metadata and flagged, never guessed or removed. ChartQA parseable lists are alternative answers; unparseable bracketed text remains one literal answer, matching VLMEvalKit. No prediction, answer-value or teacher-success selection. _local aliases excluded.',
                        image_materialization='VLMEvalKit: PIL open; RGBA/P/LA to RGB; default PIL save in original image_path extension, otherwise .jpg. References resolve to original source anchor.',
                        text_only='178 original HallusionBench image-free controls retain None image fields; no synthetic image. Same text-only inference for every arm; excluded only from visual timing.',
                        pixel_hash='SHA256(big-endian uint32 width,height + PIL RGB pixels)',
                        exact_overlap='Current student fit/val source or materialized bytes/RGB raster; metadata flags only.',
                        near_duplicate='64-bit grayscale dHash 9x8, Hamming <=3, aspect ratio within 3% versus fit+val; flags only.',
                        subsets='Main denominator retains every row. decontaminated excludes exact fit/val overlap; conservative excludes near suspects too; novel exists only with configured historical identity evidence.',
                        historical_audit='Configured prior full EADP prediction-table identities only; absent evidence is unknown, never proof of project-unseen data.',
                        limitations='Exact hashes miss recompression/crops; dHash has false positives/negatives. Project exposure differs from student training; base pretraining exposure unknown. MMBench EN/CN share images, so they are not independent visual domains.'))
    atomic_new(path, manifest)
    return manifest


def normalize_unfrozen_public_metadata(path: Path) -> dict:
    """One deterministic pre-freeze migration; preserve the old manifest.

    This handles the already-running CPU materialization whose source module
    was loaded before pandas NA parity was audited. It must never migrate a
    protocol-frozen or inference-bearing run.
    """
    path = Path(path)
    if any((path.parent / name).exists() for name in
           ('freeze.json', 'protocol.json', 'evaluation/results.sqlite', 'evaluation/results.sqlite3')):
        raise ValueError('Cannot normalize public metadata after protocol freeze/inference')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    changes = {}
    for name, dataset in manifest['datasets'].items():
        counts = Counter()
        for item in dataset['items']:
            fresh = _row_metadata(name, item['metadata'], item['source_row'], set())
            for key in ('options', 'hint', 'annotation_warnings'):
                if item.get(key) != fresh[key]:
                    counts[key] += 1
                item[key] = fresh[key]
        dataset['annotation_warning_count'] = sum(bool(r['annotation_warnings']) for r in dataset['items'])
        changes[name] = dict(counts)
    manifest['provenance'].update(pandas_version=PANDAS_VERSION,
        pandas_default_na_values=sorted(STR_NA_VALUES), public_metadata_policy=PUBLIC_METADATA_POLICY)
    manifest['provenance']['source_files'][str(Path(__file__).resolve())] = sha256(Path(__file__))
    backup = path.with_name('manifest_pre_pandas_na_policy.json')
    if backup.exists():
        raise ValueError('Pre-freeze metadata normalization has already been recorded')
    # Hard-link the old inode, then replace the main manifest atomically.
    os.link(path, backup)
    manifest['provenance']['pre_freeze_public_metadata_normalization'] = dict(
        backup_path=str(backup.resolve()), backup_sha256=sha256(backup), changes=changes,
        reason='Default pandas NA parity discovered during input-only audit; no target inference or metric-dependent selection.')
    pending = path.with_name('manifest_normalized_pending.json')
    atomic_new(pending, manifest)
    os.replace(pending, path)
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
    keys = ('num_items', 'num_unique_images', 'num_unique_parents', 'num_text_only_items',
            'training_overlap_count', 'near_duplicate_suspected_count', 'history_exposed_count')
    print(json.dumps({name: {k: value[k] for k in keys}
                     for name, value in manifest['datasets'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
