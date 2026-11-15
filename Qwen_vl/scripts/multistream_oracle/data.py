"""Freeze a second fresh DocVQA TRAIN numeric panel for a multistream oracle.

No model imports, predictions, teacher screening, OCR matching or replacement
after evaluation. The development page and primary pages have distinct parents.
Original image bytes and every original human answer alias are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

QWEN = Path(__file__).resolve().parents[2]
TRAIN = Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression/m3_train_data')
REVISION = '33136ef456fa5a3fe68568d6e31dda4eeff95b9b'
OUT = QWEN / 'outputs/multistream_oracle/data'
NUMERIC = re.compile(r'^\s*[$€£]?\s*[+−-]?\d[\d,.]*\s*%?\s*$')
PRIOR_MANIFESTS = {
    'binding': QWEN / 'outputs/binding_oracle/data/combined_manifest.json',
    'cf': QWEN / 'outputs/cf_mining/manifest.json',
    'local_reward': QWEN / 'outputs/local_reward_diagnostic/data/manifest.json',
    'catchup': QWEN / 'outputs/catchup_rerank/data/manifest.json',
    'attention_calibration_primary': QWEN / 'outputs/attention_calibration/data/manifest.json',
    'attention_calibration_development': QWEN / 'outputs/attention_calibration/data/development_manifest.json',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frozen_json(path, value):
    """Write once; a rerun may verify identical frozen contents."""
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != encoded:
            raise ValueError(f'Refusing to change frozen artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(encoded, encoding='utf-8')
    temporary.replace(path)


def numeric_key(value):
    value = re.sub(r'[\s,$£€]', '', value).replace('−', '-')
    suffix = '%' if value.endswith('%') else ''
    try:
        number = Decimal(value[:-1] if suffix else value)
        return str(number.normalize()) + suffix if number.is_finite() else None
    except InvalidOperation:
        return None


def is_numeric_qa(answers):
    if not answers or not all(isinstance(x, str) for x in answers):
        raise ValueError('Original answer aliases must be nonempty strings')
    numeric = [a for a in answers if NUMERIC.fullmatch(a)
               and sum(c.isdigit() for c in a) >= 2]
    keys = {numeric_key(a) for a in numeric}
    return bool(numeric) and None not in keys and len(keys) == 1


def stable_order(seed, *parts):
    return hashlib.sha256((':'.join([str(seed), *map(str, parts)])).encode()).hexdigest()


def exclusions(train_root, val_tsv):
    val = pd.read_csv(val_tsv, sep='\t', usecols=['image_path', 'ucsf_document_id'])
    image_to_parent = dict(zip(val.image_path.astype(str), val.ucsf_document_id.astype(str)))
    pools = {'VAL': set(val.ucsf_document_id.astype(str))}
    image_hashes = set()
    sources = {'VAL': {'path': str(val_tsv), 'sha256': sha256(val_tsv)}}
    for label, path in PRIOR_MANIFESTS.items():
        raw = json.loads(path.read_text())
        rows = raw.get('records', raw.get('items'))
        parents = set()
        for row in rows:
            parent = row.get('parent_id', row.get('source_document_id'))
            if parent is None:
                parent = image_to_parent.get(row.get('image_id'))
            if parent is None:
                raise ValueError(f'Cannot resolve original source parent in {path}')
            parents.add(str(parent))
            # Read the actual original source image, including old CF rows
            # without recorded source hashes. This is data only, not a model
            # prediction or correctness-dependent selection input.
            original = row.get('source_image', row.get('image_path'))
            if original is None:
                raise ValueError(f'Missing original source image in {path}')
            digest = sha256(original)
            for key in ('source_sha256', 'image_sha256'):
                if row.get(key) and row[key] != digest:
                    raise ValueError(f'Prior source image hash changed: {original}')
            image_hashes.add(digest)
        pools[label] = parents
        sources[label] = dict(path=str(path), sha256=sha256(path), records=len(rows))
    roster = train_root / 'train_audit_001/rows.jsonl'
    historical_fit = set()
    with roster.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] == 'fit':
                historical_fit.add(str(row['document_id']))
    pools['historical_fit_roster'] = historical_fit
    sources['historical_fit_roster'] = dict(path=str(roster), sha256=sha256(roster))
    # Preserve the earlier conservative exclusion list as an additional audit.
    older = json.loads(PRIOR_MANIFESTS['local_reward'].read_text())
    pools['prior_conservative_union'] = set(older['provenance']['excluded_parent_documents'])
    return pools, sources, image_hashes


def build(args):
    train_root = Path(args.train_root).resolve()
    out = Path(args.out).resolve()
    pools, source_files, previous_hashes = exclusions(train_root, Path(args.val_tsv).resolve())
    excluded = set().union(*pools.values())
    shards = sorted((train_root / 'train_fetch_001/shards').glob('*.parquet'))
    if not shards:
        raise ValueError('No local TRAIN parquet shards')
    annotations = []
    for shard_index, path in enumerate(shards):
        rows = pq.read_table(path, columns=['question_id', 'question', 'answers',
                                           'other_metadata', 'data_split']).to_pylist()
        for row_index, row in enumerate(rows):
            if row['data_split'] != 'train':
                raise ValueError(f'Non-TRAIN source in {path}')
            md = row['other_metadata']
            annotations.append(dict(
                shard_index=shard_index, source_parquet=str(path),
                source_parquet_row=row_index, question_id=int(row['question_id']),
                question=row['question'], answers=row['answers'],
                image_id=md['image'], parent_id=str(md['ucsf_document_id']),
                source_page_number=str(md['ucsf_document_page_no']),
                source_doc_id=md['doc_id']))
    eligible = [r for r in annotations if r['parent_id'] not in excluded
                and is_numeric_qa(r['answers'])]
    per_parent = {}
    page_qa_counts = Counter(row['image_id'] for row in annotations)
    for row in eligible:
        per_parent.setdefault(row['parent_id'], []).append(row)
    parents = sorted(per_parent, key=lambda p: (stable_order(args.seed, 'parent', p), p))
    need = args.primary + args.development
    if len(parents) < need:
        raise ValueError(f'Only {len(parents)} eligible independent parents, need {need}')
    chosen = []
    for rank, parent in enumerate(parents[:need]):
        # One seeded page per parent, then the first numeric question by ID on
        # that page. Neither prediction files nor OCR content enter this rule.
        page = min({r['image_id'] for r in per_parent[parent]},
                   key=lambda p: (stable_order(args.seed, 'page', p), p))
        row = min((r for r in per_parent[parent] if r['image_id'] == page),
                  key=lambda r: (r['question_id'], r['shard_index'], r['source_parquet_row']))
        chosen.append({**row, 'selection_rank': rank,
                       'role': 'development' if rank < args.development else 'primary'})
    # Load each needed image column once; no source shard is mutated.
    shard_hashes = {}
    for shard_index in sorted({r['shard_index'] for r in chosen}):
        path = shards[shard_index]
        shard_hashes[str(path)] = sha256(path)
        images = pq.read_table(path, columns=['image'])['image']
        for row in (r for r in chosen if r['shard_index'] == shard_index):
            payload = images[row['source_parquet_row']].as_py()
            content = payload['bytes']
            digest = hashlib.sha256(content).hexdigest()
            if digest in previous_hashes:
                raise ValueError('Prior-panel source image hash overlap; fail without replacing sample')
            target = out / 'images' / row['image_id']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing source image differs: {target}')
            if not target.exists():
                target.write_bytes(content)
            with Image.open(io.BytesIO(content)) as image:
                dimensions = list(image.size)
            row.update(item_id=f"docvqa_train_{Path(row['image_id']).stem}_q{row['question_id']}",
                       doc_id=f"docvqa_train_{Path(row['image_id']).stem}",
                       dataset='DocVQA_TRAIN', source_document_id=row['parent_id'],
                       image_path=str(target), image=str(target), source_image=str(target),
                       image_sha256=digest, source_sha256=digest,
                       source_image_size=dimensions,
                       source_parquet_sha256=shard_hashes[str(path)],
                       answers_original=list(row['answers']),
                       annotation_repository_revision=REVISION,
                       source_qa_count=page_qa_counts[row['image_id']])
        del images
    chosen_parents = {r['parent_id'] for r in chosen}
    if len(chosen_parents) != need or len({r['image_sha256'] for r in chosen}) != need:
        raise ValueError('Source-parent/image independence failed')
    overlaps = {label: sorted(chosen_parents & pool) for label, pool in pools.items()}
    if any(overlaps.values()):
        raise ValueError('Prior source-parent overlap')
    provenance = dict(
        dataset='pixparse/docvqa-single-page-questions',
        repository='https://huggingface.co/datasets/pixparse/docvqa-single-page-questions',
        repository_revision=REVISION, original_split='TRAIN',
        scope='Exploratory within-project numeric QA pilot; not official held-out evaluation or proof of unseen layouts',
        seed=args.seed, selection_uses_model_outputs=False,
        selection='Exclude previous parents first; seeded SHA256 order of eligible parents; seeded page order within parent; first eligible numeric QA by question_id; one original page/QA per parent. Development prefix is separate from primary.',
        eligibility='At least one original pure numeric/currency/percentage alias with >=2 digits; eligible numeric aliases must canonicalize to one value. Keep ALL original aliases. No OCR, geometry, teacher-correctness or compression-failure filtering.',
        evidence_ground_truth='Original human DocVQA answer aliases scored against actual generated QA answers',
        exclusion_policy='All VAL parents, every parent in prior binding/CF/local-reward/CatchUp manifests (including unused reserves), attention-calibration primary32/development1, and conservative historical fit roster/earlier union. Actual old original-image bytes are also hashed.',
        exclusion_sources=source_files,
        excluded_parent_counts={**{k: len(v) for k, v in pools.items()}, 'union': len(excluded)},
        excluded_parent_documents=sorted(excluded),
        prior_panel_source_image_hashes=sorted(previous_hashes),
        source_parquet_sha256=shard_hashes,
        data_script=str(Path(__file__).resolve()), data_script_sha256=sha256(__file__),
        adapted_data_builder=str(QWEN / 'scripts/attention_calibration/data.py'),
        adapted_data_builder_sha256=sha256(QWEN / 'scripts/attention_calibration/data.py'),
    )
    common = dict(schema_version=1, complete=True, provenance=provenance,
                  candidate_inventory=dict(annotation_rows=len(annotations),
                                           eligible_numeric_qa_rows=len(eligible),
                                           eligible_independent_parents=len(parents)),
                  data_validation=dict(all_original_image_bytes_verified=True,
                                       all_original_answer_aliases_preserved=True,
                                       unique_source_parents=True, unique_source_images=True,
                                       overlap_by_exclusion_pool=overlaps,
                                       prior_panel_image_hash_overlap=0,
                                       development_primary_parent_overlap=0),
                  protocol=dict(keep_all_samples=True, teacher_correct_filter=False,
                                errors_and_truncations='Keep every frozen question in primary analysis; no replacement',
                                planned_input_tokens=1024, planned_visual_budget=256,
                                runner_freeze_required='Freeze arms, formula, layer, prompt, cap and code/model hashes before GPU QA; development output never selects primary documents'))
    for role, name in [('primary', 'manifest.json'), ('development', 'development_manifest.json')]:
        items = [r for r in chosen if r['role'] == role]
        manifest = dict(**common, role=role,
                        counts=dict(pages=len(items), independent_source_documents=len(items),
                                    questions=len(items), original_answer_aliases=sum(len(r['answers']) for r in items)),
                        items=items, records=items)
        frozen_json(out / name, manifest)
    audit = dict(complete=True, requested_primary=args.primary, requested_development=args.development,
                 candidate_inventory=common['candidate_inventory'], data_validation=common['data_validation'],
                 excluded_parent_counts=provenance['excluded_parent_counts'],
                 primary_manifest=str(out / 'manifest.json'), primary_manifest_sha256=sha256(out / 'manifest.json'),
                 development_manifest=str(out / 'development_manifest.json'),
                 development_manifest_sha256=sha256(out / 'development_manifest.json'),
                 selected_parent_ids=[r['parent_id'] for r in chosen],
                 source_files=source_files, data_script_sha256=sha256(__file__))
    frozen_json(out / 'data_audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, default=TRAIN)
    parser.add_argument('--val-tsv', type=Path, default=Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--primary', type=int, default=32)
    parser.add_argument('--development', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026110913)
    args = parser.parse_args()
    if args.primary < 1 or args.development < 1:
        parser.error('Need positive primary and development document counts')
    audit = build(args)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('selected_parent_ids', 'source_files')}, indent=2))


if __name__ == '__main__':
    main()
