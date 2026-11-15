"""Freeze input-selected, disjoint original TRAIN pages for detail diagnostics.

Only one released parquet shard per dataset defines this diagnostic pool. It
is not a uniform sample of the complete TRAIN corpus or an official benchmark.
No predictions, teacher outputs, or answer values determine image/question
selection. Answers remain solely for the offline diagnostic evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from PIL import Image

from scripts.student_transfer import data as transfer_data
from scripts.student_transfer.data import atomic_new, dhash64, materialize_image, pixel_sha256, sha256

QWEN = Path(__file__).resolve().parents[2]
INFO_REVISION = '539088ef8a8ada01ac8e2e6d4e372586748a265e'
TEXT_REVISION = '9c0699cd19768ac5ab97568f6b3cbac4c0062884'
DEFAULT_INFO_SOURCE = QWEN / f'outputs/native_detail_sources/infographicvqa/{INFO_REVISION}/train-00000-of-00024.parquet'
DEFAULT_TEXT_SOURCE = Path(f'/data/models/huggingface/hub/datasets--lmms-lab--TextVQA/snapshots/{TEXT_REVISION}/data/train-00000-of-00020.parquet')
EXCLUSION_MANIFESTS = (
    QWEN / 'outputs/paired_pruning/run_20261111/data/manifest.json',
    QWEN / 'outputs/student_transfer/run_20261112_v2/manifest.json',
    QWEN / 'outputs/student_broad/run_20261113_v2/manifest.json',
)
DATASETS = ('InfoVQA_TRAIN', 'TextVQA_TRAIN')


def _progress(message):
    print('[detail-allocation data] ' + message, file=sys.stderr, flush=True)


def stable_order(seed, *parts):
    return hashlib.sha256(json.dumps([int(seed), *map(str, parts)], ensure_ascii=False).encode()).hexdigest()


def _all_rows(manifest):
    if isinstance(manifest.get('datasets'), dict):
        return [r for d in manifest['datasets'].values() for r in d['items']]
    if isinstance(manifest.get('splits'), dict):
        return [r for rows in manifest['splits'].values() for r in rows]
    return manifest.get('items', manifest.get('records', []))


def _exclusions(paths):
    """Use frozen input identities only, never any historical predictions."""
    sources, byte_hashes, pixels, near_refs = {}, set(), set(), {}
    manifests = []
    for path in paths:
        path = Path(path).resolve()
        sources[str(path)] = sha256(path)
        raw = json.loads(path.read_text(encoding='utf-8'))
        rows = _all_rows(raw)
        count = 0
        for row in rows:
            if not row.get('has_image', True):
                continue
            count += 1
            for key in ('image_sha256', 'source_image_sha256'):
                if row.get(key):
                    byte_hashes.add(row[key])
            for key in ('pixel_sha256', 'source_pixel_sha256'):
                if row.get(key):
                    pixels.add(row[key])
            if row.get('dhash64') and row.get('image_size'):
                ref = dict(identity=str(row.get('item_id', row.get('page_id'))),
                           dataset=row.get('dataset', 'DocVQA_TRAIN'),
                           dhash64=row['dhash64'], image_size=row['image_size'])
                near_refs[(ref['dhash64'], tuple(ref['image_size']))] = ref
            else:
                # The current student's TRAIN manifest stores original input
                # bytes and RGB hashes but no perceptual hash. Compute it from
                # the verified source file, covering dev/fit/val/test alike.
                image_path = Path(row['image_path']).resolve()
                digest = sha256(image_path)
                if digest != row['image_sha256']:
                    raise ValueError(f'Historical original image changed: {image_path}')
                sources[str(image_path)] = digest
                with Image.open(image_path) as image:
                    raster = pixel_sha256(image)
                    if row.get('pixel_sha256') and raster != row['pixel_sha256']:
                        raise ValueError(f'Historical RGB raster changed: {image_path}')
                    pixels.add(raster)
                    ref = dict(identity=str(row.get('page_id', row.get('item_id'))),
                               dataset='DocVQA_TRAIN', dhash64=f'{dhash64(image):016x}',
                               image_size=list(image.size))
                    near_refs[(ref['dhash64'], tuple(ref['image_size']))] = ref
        manifests.append(dict(path=str(path), sha256=sources[str(path)],
                              original_rows=len(rows), image_rows=count))
    return byte_hashes, pixels, list(near_refs.values()), sources, manifests


def _source_rows(path, dataset, seed):
    path = Path(path).absolute()
    if not path.name.startswith('train-') or path.suffix != '.parquet':
        raise ValueError('Diagnostic sources must be explicit original TRAIN parquet shards')
    parquet = pq.ParquetFile(path)
    if dataset == 'InfoVQA_TRAIN':
        columns = ['questionId', 'question', 'answers', 'image.path', 'image_url', 'data_split']
        id_key, split_key = 'questionId', 'data_split'
    else:
        columns = ['image_id', 'question_id', 'question', 'answers', 'image.path',
                   'image_width', 'image_height', 'set_name']
        id_key, split_key = 'question_id', 'set_name'
    rows = parquet.read(columns=columns).to_pylist()
    groups, qids = defaultdict(list), set()
    for position, row in enumerate(rows):
        if row[split_key] != 'train':
            raise ValueError(f'Non-TRAIN annotation in {path}:{position}')
        question_id = str(row[id_key])
        if question_id in qids:
            raise ValueError(f'Duplicate original question ID in shard {question_id}')
        qids.add(question_id)
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError(f'Invalid original question at {path}:{position}')
        answers = row['answers']
        if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
            raise ValueError(f'Invalid original answers at {path}:{position}')
        if dataset == 'TextVQA_TRAIN' and len(answers) != 10:
            raise ValueError('Every TextVQA question must retain its ten human answer votes')
        source_image_path = row['image']['path']
        image_id = str(row.get('image_id', source_image_path))
        if not image_id or not source_image_path:
            raise ValueError('Missing original TRAIN image identity')
        groups[image_id].append(dict(source_row=position, question_id=question_id,
            question=row['question'], answers=list(answers), source_image_id=image_id,
            source_image_path=source_image_path,
            source_metadata={k: v for k, v in row.items() if k not in ('image', 'question', 'answers')}))
    selected = []
    for image_id in sorted(groups, key=lambda i: (stable_order(seed, dataset, 'image', i), i)):
        row = min(groups[image_id], key=lambda r: (stable_order(seed, dataset, 'question', r['question_id']),
                                                 r['question_id']))
        selected.append(dict(row, source_question_count=len(groups[image_id])))
    return selected, dict(num_original_questions=len(rows), num_image_id_groups=len(groups),
                         num_row_groups=parquet.num_row_groups)


def _image_payloads(path, rows):
    wanted = {row['source_row'] for row in rows}
    found, offset = {}, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=['image']):
        for position in sorted(i for i in wanted if offset <= i < offset + len(batch)):
            payload = batch.column(0)[position - offset].as_py()
            content = payload.get('bytes') if isinstance(payload, dict) else payload
            if not isinstance(content, bytes) or not content:
                raise ValueError(f'Missing original embedded image at {path}:{position}')
            found[position] = content
        offset += len(batch)
        if len(found) == len(wanted):
            break
    if len(found) != len(wanted):
        raise ValueError('Requested TRAIN image source rows were not found')
    return found


def _near_matches(metadata, refs):
    value = int(metadata['dhash64'], 16)
    width, height = metadata['image_size']
    matches = []
    for ref in refs:
        distance = (value ^ int(ref['dhash64'], 16)).bit_count()
        if distance > 3:
            continue
        rw, rh = ref['image_size']
        if abs((width / height) / (rw / rh) - 1) <= .03:
            matches.append(dict(identity=ref['identity'], dataset=ref['dataset'], dhash_hamming=distance))
    return matches


def _build_dataset(path, dataset, run_dir, n_main, n_dev, seed, bytes_seen, pixels_seen, near_refs):
    candidates, inventory = _source_rows(path, dataset, seed)
    chosen, rejected = [], []
    for offset in range(0, len(candidates), 128):
        batch = candidates[offset:offset + 128]
        payloads = _image_payloads(path, batch)
        for candidate in batch:
            raw = payloads[candidate['source_row']]
            suffix = Path(candidate['source_image_path']).suffix.lower()
            metadata = materialize_image(raw, suffix, run_dir)
            identities = dict(image_id=candidate['source_image_id'],
                              question_id=candidate['question_id'], source_row=candidate['source_row'])
            reason = None
            if {metadata['image_sha256'], metadata['source_image_sha256']} & bytes_seen:
                reason = 'exact_source_or_materialized_image_bytes'
            elif {metadata['pixel_sha256'], metadata['source_pixel_sha256']} & pixels_seen:
                reason = 'exact_source_or_materialized_rgb_raster'
            near = _near_matches(metadata, near_refs)
            if reason or near:
                rejected.append(dict(identities, reason=reason or 'near_duplicate_suspected', near_matches=near))
                continue
            item = dict(dataset=dataset, item_id=f"{dataset}_{candidate['question_id']}",
                parent_id=f"{dataset}:image:{candidate['source_image_id']}",
                question=candidate['question'], answers=candidate['answers'],
                question_id=candidate['question_id'], source_row=candidate['source_row'],
                source_image_id=candidate['source_image_id'], source_image_path=candidate['source_image_path'],
                source_question_count=candidate['source_question_count'],
                source_metadata=candidate['source_metadata'], source_parquet=str(Path(path).absolute()),
                original_split='train', training_overlap=False, near_duplicate_suspected=False,
                **metadata)
            chosen.append(item)
            bytes_seen.update((metadata['image_sha256'], metadata['source_image_sha256']))
            pixels_seen.update((metadata['pixel_sha256'], metadata['source_pixel_sha256']))
            near_refs.append(dict(identity=item['item_id'], dataset=dataset,
                                  dhash64=metadata['dhash64'], image_size=metadata['image_size']))
            if len(chosen) == n_main + n_dev:
                break
        del payloads
        if len(chosen) == n_main + n_dev:
            break
    if len(chosen) != n_main + n_dev:
        raise ValueError(f'{dataset}: only {len(chosen)} independent clean images; need {n_main+n_dev}')
    dev, main = chosen[:n_dev], chosen[n_dev:]
    for role, rows in [('dev', dev), ('main', main)]:
        for rank, item in enumerate(rows):
            item.update(role=role, role_rank=rank)
    return dict(items=main, dev=dev, rejected_candidates=rejected,
        source_scope=dict(path=str(Path(path).absolute()), original_split='train',
            repository='lmms-lab/DocVQA' if dataset == 'InfoVQA_TRAIN' else 'lmms-lab/TextVQA',
            revision=INFO_REVISION if dataset == 'InfoVQA_TRAIN' else TEXT_REVISION,
            limitation='Input-selected subset of this single released TRAIN shard, not a uniform complete-TRAIN sample or official held-out benchmark; pretraining exposure unknown.',
            **inventory),
        counts=dict(main=len(main), dev=len(dev), rejected=len(rejected),
                    rejected_by_reason=dict(Counter(r['reason'] for r in rejected))))


def verify_manifest(manifest):
    if manifest.get('schema') != 1 or not manifest.get('complete'):
        raise ValueError('Incomplete or unsupported detail diagnostic manifest')
    for path, expected in manifest['provenance']['source_files'].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f'Frozen diagnostic source changed: {path}')
    seen_pixels, seen_bytes, seen_parents = set(), set(), set()
    for name in DATASETS:
        dataset = manifest['datasets'][name]
        if len(dataset['items']) != manifest['selection']['n_main'] or len(dataset['dev']) != manifest['selection']['n_dev']:
            raise ValueError('Frozen diagnostic split counts changed')
        for item in dataset['dev'] + dataset['items']:
            if item['dataset'] != name or item['parent_id'] in seen_parents:
                raise ValueError('Duplicate diagnostic parent or dataset mismatch')
            seen_parents.add(item['parent_id'])
            raw = Path(item['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['image_sha256']:
                raise ValueError('Frozen diagnostic input image changed')
            with Image.open(io.BytesIO(raw)) as image:
                if pixel_sha256(image) != item['pixel_sha256']:
                    raise ValueError('Frozen diagnostic RGB raster changed')
            if item['pixel_sha256'] in seen_pixels or item['image_sha256'] in seen_bytes:
                raise ValueError('Duplicate diagnostic raster across datasets/splits')
            seen_pixels.add(item['pixel_sha256'])
            seen_bytes.add(item['image_sha256'])
    return manifest


def build_manifest(run_dir, info_source=DEFAULT_INFO_SOURCE, text_source=DEFAULT_TEXT_SOURCE,
                   n_main=192, n_dev=2, seed=20261114, *, exclusion_manifests=EXCLUSION_MANIFESTS):
    run_dir = Path(run_dir).resolve()
    if n_main < 1 or n_dev < 1:
        raise ValueError('Both diagnostic split sizes must be positive')
    # Snapshot parquet names are symlinks to extension-free HF blob names.
    # Preserve their pinned TRAIN path in the source identity and schema guard.
    paths = tuple(Path(p).absolute() for p in (info_source, text_source))
    exclusions = tuple(Path(p).resolve() for p in exclusion_manifests)
    config = dict(n_main=int(n_main), n_dev=int(n_dev), seed=int(seed),
                  source_parquets=list(map(str, paths)), exclusion_manifests=list(map(str, exclusions)),
                  image_order='SHA256(JSON[seed,dataset,image,original_image_id]); question order uses questionId only.',
                  role_order=['dev', 'main'], uses_answers_or_predictions_for_selection=False,
                  duplicate_policy='Exclude exact source/materialized bytes or RGB against current student all four splits and old thirteen datasets; also exclude 64-bit dHash Hamming<=3 and aspect ratio within3%; apply same tests to already selected pages across datasets/dev/main. Flags are input-only heuristics, not proof of no possible near duplicates.')
    target = run_dir / 'manifest.json'
    if target.exists():
        result = json.loads(target.read_text(encoding='utf-8'))
        if result['selection'] != config:
            raise ValueError('Existing diagnostic manifest has a different selection configuration')
        return verify_manifest(result)
    run_dir.mkdir(parents=True, exist_ok=True)
    byte_hashes, pixels, refs, sources, exclusion_inventory = _exclusions(exclusions)
    sources.update({str(path): sha256(path) for path in paths})
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__))
    sources[str(Path(transfer_data.__file__).resolve())] = sha256(Path(transfer_data.__file__))
    result = {}
    for name, path in zip(DATASETS, paths):
        result[name] = _build_dataset(path, name, run_dir, n_main, n_dev, seed, byte_hashes, pixels, refs)
        _progress(f'{name}: main {n_main}, dev {n_dev}, rejected {result[name]["counts"]["rejected"]}')
    # Detect source modifications during materialization before freezing.
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Diagnostic source changed during preparation: {path}')
    manifest = dict(schema=1, complete=True, selection=config, datasets=result,
        provenance=dict(source_files=sources, exclusion_inventory=exclusion_inventory,
            image_materialization='Pinned student_transfer PIL decode/default-save convention; original source byte and pixel identities recorded.',
            exclusions='Frozen input identities from current student dev/fit/val/test and previous thirteen datasets, never historical prediction scores.',
            online_inputs='Only image and original question may enter online allocation; source OCR, answer aliases/votes and other metadata remain offline.',
            limitations='A small diagnostic panel from first available TRAIN shards; not an untouched whole-project blind test, official benchmark, uniform full-TRAIN sample, or guarantee of base-model pretraining independence. dHash has false positives and false negatives.'))
    atomic_new(target, manifest)
    return verify_manifest(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--info-source', type=Path, default=DEFAULT_INFO_SOURCE)
    parser.add_argument('--text-source', type=Path, default=DEFAULT_TEXT_SOURCE)
    parser.add_argument('--n-main', type=int, default=192)
    parser.add_argument('--n-dev', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261114)
    args = parser.parse_args()
    manifest = build_manifest(args.run_dir, args.info_source, args.text_source, args.n_main, args.n_dev, args.seed)
    print(json.dumps({name: d['counts'] for name, d in manifest['datasets'].items()}, indent=2))


if __name__ == '__main__':
    main()
