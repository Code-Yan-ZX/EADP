"""Freeze disjoint DocVQA TRAIN fit/validation/test/dev student-distillation panels.

CPU data only. Uses original annotations and image bytes; never reads predictions,
model outputs or QA correctness. The historical conservative excluded parent and
source-image sets include both 33-page attention/multistream oracle panels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
QWEN = HERE.parents[1]
BASE = QWEN / 'scripts/multistream_oracle/data.py'
spec = importlib.util.spec_from_file_location('multistream_data_only', BASE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
OUT = QWEN / 'outputs/student_distill/data'
SEED = 2026110914


def build(out):
    out = Path(out).resolve()
    m.PRIOR_MANIFESTS = dict(m.PRIOR_MANIFESTS, **{
        'multistream_primary': m.OUT / 'manifest.json',
        'multistream_development': m.OUT / 'development_manifest.json',
    })
    pools, sources, previous_hashes = m.exclusions(
        m.TRAIN, Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv'))
    excluded = set().union(*pools.values())
    shards = sorted((m.TRAIN / 'train_fetch_001/shards').glob('*.parquet'))
    if len(shards) != 36:
        raise ValueError(f'Expected 36 local original TRAIN shards; found {len(shards)}')
    annotations = []
    for si, path in enumerate(shards):
        rows = pq.read_table(path, columns=['question_id', 'question', 'answers',
                                           'other_metadata', 'data_split']).to_pylist()
        for ri, row in enumerate(rows):
            if row['data_split'] != 'train':
                raise ValueError(f'Non-TRAIN source in {path}')
            md = row['other_metadata']
            annotations.append(dict(shard_index=si, source_parquet=str(path),
                source_parquet_row=ri, question_id=int(row['question_id']),
                question=row['question'], answers=list(row['answers']),
                image_id=md['image'], parent_id=str(md['ucsf_document_id']),
                source_page_number=str(md['ucsf_document_page_no']),
                source_doc_id=md['doc_id']))
    counts = Counter(r['image_id'] for r in annotations)
    eligible = [r for r in annotations if r['parent_id'] not in excluded
                and m.is_numeric_qa(r['answers'])]
    per_parent = {}
    for row in eligible:
        per_parent.setdefault(row['parent_id'], []).append(row)
    parents = sorted(per_parent, key=lambda p: (m.stable_order(SEED, 'parent', p), p))
    candidates = []
    for rank, parent in enumerate(parents):
        entries = per_parent[parent]
        page = min({r['image_id'] for r in entries},
                   key=lambda p: (m.stable_order(SEED, 'page', p), p))
        row = min((r for r in entries if r['image_id'] == page),
                  key=lambda r: (r['question_id'], r['shard_index'], r['source_parquet_row']))
        candidates.append({**row, 'candidate_rank': rank})
    # Fetch selected candidate image bytes from the original shard row; compute
    # digests before assigning roles, and skip only exact byte duplicates.
    payloads, shard_hashes = {}, {}
    for si in sorted({r['shard_index'] for r in candidates}):
        path = shards[si]
        shard_hashes[str(path)] = m.sha256(path)
        images = pq.read_table(path, columns=['image'])['image']
        for row in (r for r in candidates if r['shard_index'] == si):
            content = images[row['source_parquet_row']].as_py()['bytes']
            if not isinstance(content, bytes) or not content:
                raise ValueError('Missing original image bytes')
            payloads[row['candidate_rank']] = content
            row['image_sha256'] = hashlib.sha256(content).hexdigest()
        del images
    unique = []
    rejected = []
    seen = set(previous_hashes)
    for row in candidates:
        if row['image_sha256'] in seen:
            rejected.append({'candidate_rank': row['candidate_rank'], 'parent_id': row['parent_id'],
                             'image_sha256': row['image_sha256'], 'reason': 'duplicate_exact_source_image_bytes'})
            continue
        seen.add(row['image_sha256'])
        unique.append(row)
    planned = [('dev', 1), ('fit', 120), ('val', 16), ('test', 32)]
    if len(unique) < sum(n for _, n in planned):
        planned = [('dev', 1), ('fit', 96), ('val', 16), ('test', 32)]
    if len(unique) < sum(n for _, n in planned):
        raise ValueError(f'Only {len(unique)} duplicate-free independent eligible parents; need 145')
    selected = unique[:sum(n for _, n in planned)]
    offset = 0
    groups = {}
    for role, count in planned:
        groups[role] = selected[offset:offset+count]
        offset += count
    # Fail rather than mutate any previous frozen manifest or image.
    if any((out / f'{role}_manifest.json').exists() for role, _ in planned) or (out/'data_audit.json').exists():
        raise ValueError('Frozen student manifest already exists; refusing replacement')
    for role, rows in groups.items():
        for rank_in_role, row in enumerate(rows):
            content = payloads[row['candidate_rank']]
            target = out / 'images' / row['image_id']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing source image differs: {target}')
            if not target.exists():
                target.write_bytes(content)
            if m.sha256(target) != row['image_sha256']:
                raise ValueError('Image write/readback mismatch')
            with Image.open(io.BytesIO(content)) as image:
                dimensions = list(image.size)
            row.update(item_id=f"docvqa_train_{Path(row['image_id']).stem}_q{row['question_id']}",
                doc_id=f"docvqa_train_{Path(row['image_id']).stem}", dataset='DocVQA_TRAIN',
                source_document_id=row['parent_id'], image_path=str(target), image=str(target),
                source_image=str(target), source_sha256=row['image_sha256'],
                source_image_size=dimensions, source_parquet_sha256=shard_hashes[row['source_parquet']],
                answers_original=list(row['answers']), annotation_repository_revision=m.REVISION,
                source_qa_count=counts[row['image_id']], role=role, role_rank=rank_in_role)
    chosen_parents = {r['parent_id'] for r in selected}
    chosen_hashes = {r['image_sha256'] for r in selected}
    if len(chosen_parents) != len(selected) or len(chosen_hashes) != len(selected):
        raise ValueError('Selected parent/hash independence violation')
    if chosen_parents & excluded or chosen_hashes & previous_hashes:
        raise ValueError('Historical parent/hash independence violation')
    overlaps = {label: sorted(chosen_parents & pool) for label, pool in pools.items()}
    if any(overlaps.values()):
        raise ValueError('Historical parent overlap')
    provenance = dict(dataset='pixparse/docvqa-single-page-questions',
        repository='https://huggingface.co/datasets/pixparse/docvqa-single-page-questions',
        repository_revision=m.REVISION, original_split='TRAIN', seed=SEED,
        selection_uses_model_outputs=False,
        selection='Parent SHA256-seeded ordering; seeded page order per parent; first eligible numeric QA by question_id. Drop only duplicate original-image byte hashes against previous panels/earlier seeded candidates; assign first 1 dev, next 120 fit, next 16 val, next 32 test (fallback 96 fit only if fewer than 169 unique candidates).',
        eligibility='Original numeric/currency/percentage answer alias with >=2 digits; all numeric aliases canonicalize to the same value. Retain all human aliases; no OCR, model scores, gradient, correctness or failure filtering.',
        scope='Project-disjoint TRAIN source parents, not an official held-out benchmark or a guarantee of pretraining/unseen layout independence.',
        exclusion_sources=sources,
        excluded_parent_counts={**{k:len(v) for k,v in pools.items()},'union':len(excluded)},
        excluded_parent_documents=sorted(excluded),
        prior_panel_source_image_hashes=sorted(previous_hashes),
        source_parquet_sha256=shard_hashes,
        data_script=str(Path(__file__).resolve()), data_script_sha256=m.sha256(__file__),
        adapted_data_builder=str(BASE), adapted_data_builder_sha256=m.sha256(BASE))
    inventory = dict(annotation_rows=len(annotations), eligible_numeric_qa_rows=len(eligible),
                     eligible_independent_parents=len(parents),
                     eligible_unique_images_after_hash_check=len(unique),
                     hash_duplicate_candidates_rejected=len(rejected),
                     fixed_split_counts=dict(planned))
    data_validation = dict(unique_source_parents=True, unique_source_images=True,
        selected_parent_overlap_by_exclusion_pool=overlaps, prior_panel_image_hash_overlap=0,
        disjoint_split_parent_and_image_hashes=True, all_original_image_bytes_readback_verified=True,
        all_original_answer_aliases_preserved=True, keep_all_samples=True)
    manifests = {}
    for role, rows in groups.items():
        manifest = dict(schema_version=1, complete=True, role=role, provenance=provenance,
            candidate_inventory=inventory, data_validation=data_validation,
            counts=dict(pages=len(rows), independent_source_documents=len(rows), questions=len(rows)),
            items=rows, records=rows)
        path = out/f'{role}_manifest.json'
        m.frozen_json(path, manifest)
        manifests[role] = dict(path=str(path), sha256=m.sha256(path), count=len(rows))
    audit = dict(complete=True, candidate_inventory=inventory, data_validation=data_validation,
                 excluded_parent_counts=provenance['excluded_parent_counts'],
                 rejected_candidates=rejected, manifests=manifests,
                 data_script_sha256=m.sha256(__file__), adapted_data_builder_sha256=m.sha256(BASE))
    m.frozen_json(out/'data_audit.json', audit)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(build(args.out),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
