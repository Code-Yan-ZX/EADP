"""Freeze original multi-question DocVQA TRAIN pages without running a model.

Run from Qwen_vl with ``python -m scripts.paired_pruning.data --out RUN/data``.
The first run hashes all source shards and excluded images (CPU and disk work).
A completed manifest is immutable: later runs verify it, never resample it.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import io
from itertools import islice
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unicodedata

QWEN = Path(__file__).resolve().parents[2]
TRAIN = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/m3_train_data')
VAL_TSV = Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv')
BASE = QWEN / 'scripts/multistream_oracle/data.py'
SPLITS = ('dev', 'fit', 'val', 'test')
NUMERIC = re.compile(r'^\s*[$€£]?\s*[+−-]?\d[\d,.]*\s*%?\s*$')


def _progress(message):
    print(f'[paired-pruning data] {message}', file=sys.stderr, flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(content):
    """Hash the decoded RGB raster, including dimensions, not file metadata."""
    from PIL import Image
    with Image.open(io.BytesIO(content)) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(struct.pack('>II', *rgb.size) + rgb.tobytes()).hexdigest()


def stable_order(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *map(str, parts)], ensure_ascii=False).encode()).hexdigest()


def question_key(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def numeric_answers(answers):
    return any(NUMERIC.fullmatch(answer) is not None for answer in answers)


def _config(args):
    counts = {name: int(getattr(args, name, {'fit': 128, 'val': 24, 'test': 64, 'dev': 1}[name]))
              for name in SPLITS}
    questions = int(getattr(args, 'questions_per_page', 3))
    if any(value < 1 for value in counts.values()) or questions < 2:
        raise ValueError('All split sizes must be positive; questions_per_page must be >= 2')
    return dict(train_root=str(Path(args.train_root).resolve()),
                val_tsv=str(Path(args.val_tsv).resolve()),
                history_root=str(Path(getattr(args, 'history_root', QWEN / 'outputs')).resolve()),
                extra_manifests=sorted(str(Path(p).resolve()) for p in getattr(args, 'extra_manifest', [])),
                seed=int(getattr(args, 'seed', 2026111101)), split_pages=counts,
                questions_per_page=questions, minimum_nonnumeric_questions_per_page=1,
                selection_uses_model_outputs=False, original_split='train',
                selection='Seeded parent, then page ordering; one page per parent; one seeded nonnumeric question plus seeded distinct original questions; reject byte/RGB duplicates before assigning splits; replenish without reducing counts.',
                split_assignment_order=list(SPLITS),
                scope='Project-disjoint TRAIN pages; not evidence of pretraining or unseen-layout independence.')


def _manifest_paths(config, out):
    root = Path(config['history_root'])
    paths = {p.resolve() for p in root.rglob('*manifest*.json')
             if not p.resolve().is_relative_to(out)}
    paths.update(Path(p) for p in config['extra_manifests'])
    return sorted(paths)


def _rows(raw):
    if isinstance(raw.get('splits'), dict):
        return [row for rows in raw['splits'].values() for row in rows]
    for key in ('records', 'items'):
        if isinstance(raw.get(key), list):
            return raw[key]
    return []


def _annotations(shards):
    import pyarrow.parquet as pq
    pages = {}
    ids = {}
    duplicate_questions = 0
    for shard_number, path in enumerate(shards, 1):
        rows = pq.read_table(path, columns=['question_id', 'question', 'answers',
                                           'other_metadata', 'data_split']).to_pylist()
        for index, row in enumerate(rows):
            if str(row['data_split']).lower() != 'train':
                raise ValueError(f'Non-TRAIN source in {path}:{index}')
            question, answers = row['question'], row['answers']
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f'Empty original question in {path}:{index}')
            if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                raise ValueError(f'Invalid original answer aliases in {path}:{index}')
            md = row['other_metadata']
            image_id, parent = str(md['image']), str(md['ucsf_document_id'])
            qid = str(row['question_id'])
            signature = (image_id, parent, question, tuple(answers))
            if qid in ids:
                if ids[qid] != signature:
                    raise ValueError(f'Conflicting duplicate question_id {qid}')
                duplicate_questions += 1
                continue
            ids[qid] = signature
            page = pages.setdefault(image_id, dict(page_id=image_id, parent_id=parent,
                source=dict(parquet=str(path), row=index, image_id=image_id,
                            page_number=str(md.get('ucsf_document_page_no', '')),
                            document_id=str(md.get('doc_id', ''))), questions=[]))
            if page['parent_id'] != parent:
                raise ValueError(f'Page {image_id} has multiple source parents')
            if any(question_key(q['question']) == question_key(question) for q in page['questions']):
                duplicate_questions += 1
                continue
            page['questions'].append(dict(item_id=f'docvqa_train_q{qid}', question_id=qid,
                question=question, answers=list(answers), numeric=numeric_answers(answers),
                source=dict(parquet=str(path), row=index)))
        _progress(f'annotations shard {shard_number}/{len(shards)}: '
                  f'{len(ids)} question IDs, {len(pages)} unique pages')
    return pages, dict(unique_question_ids=len(ids), unique_pages=len(pages),
                       duplicate_questions_skipped=duplicate_questions)


def _image_payloads(rows):
    """Stream small Arrow batches; retain only requested original image bytes."""
    import pyarrow.parquet as pq
    by_shard = defaultdict(list)
    for row in rows:
        by_shard[row['source']['parquet']].append(row)
    for path, group in sorted(by_shard.items()):
        wanted = defaultdict(list)
        for row in group:
            wanted[row['source']['row']].append(row)
        last_row, offset = max(wanted), 0
        parquet = pq.ParquetFile(path)
        if min(wanted) < 0 or last_row >= parquet.metadata.num_rows:
            raise ValueError(f'Source row outside parquet: {path}')
        # Reading a whole image column can retain gigabytes of repeated image
        # buffers in Arrow's allocator. Streaming bounds this preparation cost.
        for batch in parquet.iter_batches(batch_size=32, columns=['image']):
            for index in sorted(i for i in wanted if offset <= i < offset + len(batch)):
                payload = batch.column(0)[index - offset].as_py()
                content = payload.get('bytes') if isinstance(payload, dict) else payload
                if not isinstance(content, bytes) or not content:
                    raise ValueError(f'Missing embedded original image bytes: {path}:{index}')
                for row in wanted[index]:
                    yield row, content
            offset += len(batch)
            if offset > last_row:
                break


def _historical_image_hashes(rows):
    """Bounded CPU decoding; never retain the full historical image corpus.

    Pillow decoding and hashlib release the GIL. Four independent workers
    handle at most sixteen payloads at a time, and map preserves input order.
    """
    def hash_one(item):
        page, content = item
        return page, hashlib.sha256(content).hexdigest(), pixel_sha256(content)

    iterator = iter(_image_payloads(rows))
    with ThreadPoolExecutor(max_workers=4) as workers:
        while batch := list(islice(iterator, 16)):
            yield from workers.map(hash_one, batch)


def _legacy_exclusions(config):
    """Reuse the previous project's conservative exclusion policy verbatim."""
    spec = importlib.util.spec_from_file_location('paired_pruning_legacy_data', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(config['history_root'])
    module.PRIOR_MANIFESTS = {label: root / path.relative_to(QWEN / 'outputs')
                              for label, path in module.PRIOR_MANIFESTS.items()}
    # Missing historical inputs are errors, not an invitation to weaken exclusions.
    return module.exclusions(Path(config['train_root']), Path(config['val_tsv']))


def _exclusions(config, pages, paths):
    _progress('reading historical exclusions and VAL image references')
    pools, legacy_sources, old_hashes = _legacy_exclusions(config)
    sources = {str(Path(v['path']).resolve()): v['sha256'] for v in legacy_sources.values()}
    sources[str(BASE)] = sha256(BASE)
    byte_hashes, pixels = set(old_hashes), set()
    known_images = {name: page['parent_id'] for name, page in pages.items()}
    val_payloads, val_parent, val_indices = {}, {}, {}
    # VLMEvalKit TSV may reference the index of an earlier embedded image.
    csv.field_size_limit(1024 ** 3)
    with Path(config['val_tsv']).open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = str(row['image_path'])
            val_parent[name] = str(row['ucsf_document_id'])
            value = row.get('image', '')
            if row.get('index'):
                val_indices[row['index']] = value
            if name not in val_payloads:
                val_payloads[name] = value
    for image_number, (name, value) in enumerate(val_payloads.items(), 1):
        chain = set()
        while value.isdigit():
            if value in chain or value not in val_indices:
                raise ValueError(f'Unresolvable VAL image reference {value}')
            chain.add(value)
            value = val_indices[value]
        if value:
            content = base64.b64decode(value, validate=True)
        else:
            image_path = Path(config['val_tsv']).parent / 'images/DocVQA_VAL' / name
            content = image_path.read_bytes()
        byte_hashes.add(hashlib.sha256(content).hexdigest())
        pixels.add(pixel_sha256(content))
        if image_number % 500 == 0 or image_number == len(val_payloads):
            _progress(f'VAL RGB hashes {image_number}/{len(val_payloads)} unique pages')
    del val_payloads, val_indices
    known_images.update(val_parent)
    image_paths = set()
    for path in paths:
        raw = json.loads(path.read_text(encoding='utf-8'))
        rows = _rows(raw)
        # Run manifests are included in the frozen inventory but do not carry panels.
        sources[str(path)] = sha256(path)
        parents = set()
        for row in rows:
            parent = row.get('parent_id', row.get('source_document_id'))
            if parent is None:
                parent = known_images.get(str(row.get('image_id', row.get('page_id', row.get('image_path', '')))))
            if parent is None:
                raise ValueError(f'Cannot resolve historical source parent in {path}')
            parents.add(str(parent))
            original = row.get('source_image', row.get('image_path', row.get('image')))
            if original:
                image_path = Path(original)
                if not image_path.is_absolute():
                    choices = [path.parent / image_path,
                               Path(config['val_tsv']).parent / 'images/DocVQA_VAL' / image_path]
                    image_path = next((p for p in choices if p.is_file()), image_path)
                if not image_path.is_file():
                    raise ValueError(f'Missing historical original image: {original} ({path})')
                image_paths.add(image_path.resolve())
        provenance = raw.get('provenance', {})
        parents.update(map(str, provenance.get('excluded_parent_documents', [])))
        if parents:
            pools[str(path)] = parents
    # The old roster's validation entries are a reserved candidate pool, not
    # proof of model exposure. Keep the legacy fit exclusion; actual old
    # validation/test experiments are excluded through their source manifests.
    for image_number, path in enumerate(sorted(image_paths), 1):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        sources[str(path)] = digest
        byte_hashes.add(digest)
        pixels.add(pixel_sha256(content))
        if image_number % 500 == 0 or image_number == len(image_paths):
            _progress(f'historical manifest RGB hashes {image_number}/{len(image_paths)} original paths')
    excluded = set().union(*pools.values())
    return excluded, byte_hashes, pixels, sources, pools


def _validate(manifest, config):
    if manifest.get('schema_version') != 1 or manifest.get('protocol') != config:
        raise ValueError('Frozen manifest protocol differs; use a new output directory')
    parents, images, pixels, questions = set(), set(), set(), set()
    excluded = manifest.get('exclusions', {})
    forbidden_parents = set(excluded.get('parent_ids', []))
    forbidden_bytes = set(excluded.get('prior_image_sha256', []))
    forbidden_pixels = set(excluded.get('prior_pixel_sha256', []))
    for split in SPLITS:
        rows = manifest['splits'][split]
        if len(rows) != config['split_pages'][split]:
            raise ValueError(f'Frozen {split} size differs')
        for page in rows:
            if (page['parent_id'] in forbidden_parents or page['image_sha256'] in forbidden_bytes
                    or page['pixel_sha256'] in forbidden_pixels):
                raise ValueError('Frozen page overlaps a historical parent or image')
            content = Path(page['image_path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != page['image_sha256'] or pixel_sha256(content) != page['pixel_sha256']:
                raise ValueError(f'Frozen image changed: {page["image_path"]}')
            for value, seen, label in ((page['parent_id'], parents, 'parent'),
                                       (page['image_sha256'], images, 'image bytes'),
                                       (page['pixel_sha256'], pixels, 'RGB pixels')):
                if value in seen:
                    raise ValueError(f'Frozen split overlap in {label}: {value}')
                seen.add(value)
            qa = page['questions']
            if len(qa) != config['questions_per_page'] or all(q['numeric'] for q in qa):
                raise ValueError('Frozen page lacks the required distinct multi-question mix')
            if len({question_key(q['question']) for q in qa}) != len(qa):
                raise ValueError('Duplicate questions in frozen page')
            for q in qa:
                if q['item_id'] in questions:
                    raise ValueError('Duplicate frozen question item_id')
                questions.add(q['item_id'])


def build(args):
    """Build or verify the immutable manifest; return its absolute Path."""
    config = _config(args)
    out = Path(args.out).resolve()
    target = out / 'manifest.json'
    paths = _manifest_paths(config, out)
    shards = sorted((Path(config['train_root']) / 'train_fetch_001/shards').glob('*.parquet'))
    if not shards:
        raise ValueError('No local TRAIN parquet shards')
    if target.exists():
        _progress(f'verifying existing immutable manifest: {target}')
        manifest = json.loads(target.read_text(encoding='utf-8'))
        if manifest.get('historical_manifest_inventory') != list(map(str, paths)):
            raise ValueError('Historical manifest inventory changed; refusing silent resampling')
        if manifest.get('source_shards') != list(map(str, shards)):
            raise ValueError('TRAIN shard inventory changed')
        _validate(manifest, config)
        source_items = manifest['source_sha256'].items()
        for source_number, (path, expected) in enumerate(source_items, 1):
            if sha256(path) != expected:
                raise ValueError(f'Frozen source changed: {path}')
            if source_number % 100 == 0 or source_number == len(source_items):
                _progress(f'resume source hashes {source_number}/{len(source_items)}')
        _progress('resume verification complete; frozen manifest unchanged')
        return target
    _progress(f'building {sum(config["split_pages"].values())} independent pages, '
              f'{config["questions_per_page"]} original questions per page; CPU only')
    pages, inventory = _annotations(shards)
    excluded, previous_bytes, previous_pixels, sources, pools = _exclusions(config, pages, paths)
    for shard_number, path in enumerate(shards, 1):
        sources[str(path)] = sha256(path)
        if shard_number % 6 == 0 or shard_number == len(shards):
            _progress(f'original TRAIN shard hashes {shard_number}/{len(shards)}')
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    per_parent = defaultdict(list)
    for page in pages.values():
        if (page['parent_id'] not in excluded and len(page['questions']) >= config['questions_per_page']
                and any(not q['numeric'] for q in page['questions'])):
            per_parent[page['parent_id']].append(page)
    seed = config['seed']
    parents = sorted(per_parent, key=lambda p: (stable_order(seed, 'parent', p), p))
    # Candidate pages are grouped by parent so duplicate first pages can be
    # skipped while still admitting another eligible page from the same parent.
    candidates = [page for parent in parents for page in sorted(per_parent[parent],
                  key=lambda p: (stable_order(seed, 'page', p['page_id']), p['page_id']))]
    need = sum(config['split_pages'].values())
    if len(parents) < need:
        raise ValueError(f'Only {len(parents)} eligible independent parents, need exactly {need}; no scale fallback')
    _progress(f'annotation gate: {len(parents)} eligible parents, need {need}; '
              f'{len(candidates)} candidate pages before byte/RGB checks')
    # Check the cheap annotation feasibility gate before reading historical
    # TRAIN image payloads. This catches re-encoded duplicates under new parents.
    excluded_train = [page for page in pages.values() if page['parent_id'] in excluded]
    _progress(f'historical TRAIN RGB scan starts: {len(excluded_train)} unique pages')
    for image_number, (_, image_hash, pixel_hash) in enumerate(_historical_image_hashes(excluded_train), 1):
        previous_bytes.add(image_hash)
        previous_pixels.add(pixel_hash)
        if image_number % 500 == 0 or image_number == len(excluded_train):
            _progress(f'historical TRAIN RGB hashes {image_number}/{len(excluded_train)} unique pages')
    selected, selected_parents, rejected = [], set(), []
    seen_bytes, seen_pixels = set(previous_bytes), set(previous_pixels)
    cursor = 0
    while len(selected) < need and cursor < len(candidates):
        batch = candidates[cursor:cursor + max(need - len(selected), 16)]
        cursor += len(batch)
        payloads = {page['page_id']: content for page, content in _image_payloads(batch)
                    if page['parent_id'] not in selected_parents}
        for page in batch:
            if page['parent_id'] in selected_parents:
                continue
            content = payloads[page['page_id']]
            image_hash, pixel_hash = hashlib.sha256(content).hexdigest(), pixel_sha256(content)
            if image_hash in seen_bytes or pixel_hash in seen_pixels:
                rejected.append(dict(page_id=page['page_id'], parent_id=page['parent_id'],
                    reason='duplicate_image_bytes' if image_hash in seen_bytes else 'duplicate_RGB_pixels'))
                continue
            questions = sorted(page['questions'], key=lambda q: (stable_order(seed, 'question', q['question_id']), q['question_id']))
            nonnumeric = next(q for q in questions if not q['numeric'])
            qa = [nonnumeric] + [q for q in questions if q is not nonnumeric][:config['questions_per_page'] - 1]
            qa.sort(key=lambda q: (stable_order(seed, 'question_order', q['question_id']), q['question_id']))
            # A digest-based filename avoids traversal and collisions in source IDs.
            image_path = out / 'images' / (image_hash + '.img')
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.exists() and image_path.read_bytes() != content:
                raise ValueError(f'Existing image differs: {image_path}')
            if not image_path.exists():
                with image_path.open('xb') as stream:
                    stream.write(content)
            selected.append(dict(page_id=page['page_id'], parent_id=page['parent_id'],
                image_path=str(image_path), image_sha256=image_hash, pixel_sha256=pixel_hash,
                questions=qa, source=page['source'], source_question_count=len(page['questions'])))
            selected_parents.add(page['parent_id'])
            seen_bytes.add(image_hash)
            seen_pixels.add(pixel_hash)
            if len(selected) == need:
                break
        _progress(f'new candidates: selected {len(selected)}/{need} independent parents; '
                  f'{len(rejected)} byte/RGB duplicates skipped')
    if len(selected) != need:
        raise ValueError(f'Only {len(selected)} byte/RGB-disjoint parents after exclusions; need exactly {need}; no manifest written')
    splits, offset = {}, 0
    for split in SPLITS:
        count = config['split_pages'][split]
        splits[split] = selected[offset:offset + count]
        offset += count
    manifest = dict(schema_version=1, protocol=config, source_sha256=sources,
        source_shards=list(map(str, shards)), historical_manifest_inventory=list(map(str, paths)),
        splits=splits, counts=dict(pages=need, parents=need, questions=need * config['questions_per_page'],
            splits={split: dict(pages=len(rows), questions=sum(len(p['questions']) for p in rows),
                numeric_questions=sum(q['numeric'] for p in rows for q in p['questions'])) for split, rows in splits.items()}),
        exclusions=dict(parent_ids=sorted(excluded), parent_counts={k: len(v) for k, v in pools.items()},
                        prior_image_sha256=sorted(previous_bytes), prior_pixel_sha256=sorted(previous_pixels),
                        rejected_candidates=rejected),
        candidate_inventory={**inventory, 'eligible_parents': len(parents), 'candidate_pages': len(candidates)})
    _validate(manifest, config)
    out.mkdir(parents=True, exist_ok=True)
    # Atomic publication plus exclusive hard link: crashes cannot leave a
    # partially written manifest and parallel builders cannot overwrite one.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=out,
                                         prefix='.manifest-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _progress(f'completed {need} pages / {need * config["questions_per_page"]} questions: {target}')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=TRAIN)
    parser.add_argument('--val-tsv', type=Path, default=VAL_TSV)
    parser.add_argument('--history-root', type=Path, default=QWEN / 'outputs')
    parser.add_argument('--extra-manifest', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026111101)
    for split, default in [('fit', 128), ('val', 24), ('test', 64), ('dev', 1)]:
        parser.add_argument('--' + split, type=int, default=default)
    parser.add_argument('--questions-per-page', type=int, default=3)
    path = build(parser.parse_args())
    print(json.dumps(dict(manifest=str(path), sha256=sha256(path)), ensure_ascii=False))


if __name__ == '__main__':
    main()
