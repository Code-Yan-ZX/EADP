"""CPU-only data integrity checks, using tiny original-image parquet fixtures."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, PngImagePlugin
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from . import data


def image_bytes(value, note=None):
    image = Image.new('RGB', (7, 5), (value, value // 2, 255 - value))
    target = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    if note:
        metadata.add_text('note', note)
    image.save(target, format='PNG', pnginfo=metadata)
    return target.getvalue()


def row(parent, page, question_id, question, answers, content):
    return dict(question_id=question_id, question=question, answers=answers,
        image={'bytes': content, 'path': None}, data_split='train',
        other_metadata=dict(image=page, ucsf_document_id=parent,
                            ucsf_document_page_no=1, doc_id=question_id))


def fixture(tmp_path, *, duplicate_pixels=False, historical_parent=None, count=10):
    train, history, out = tmp_path / 'train', tmp_path / 'history', tmp_path / 'data'
    shard = train / 'train_fetch_001/shards/part.parquet'
    shard.parent.mkdir(parents=True)
    roster = train / 'train_audit_001/rows.jsonl'
    roster.parent.mkdir(parents=True)
    roster.write_text('')
    # The previous builder's exact required manifests, with no old samples.
    manifests = ['binding_oracle/data/combined_manifest.json', 'cf_mining/manifest.json',
        'local_reward_diagnostic/data/manifest.json', 'catchup_rerank/data/manifest.json',
        'attention_calibration/data/manifest.json',
        'attention_calibration/data/development_manifest.json']
    for name in manifests:
        path = history / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(records=[], provenance={'excluded_parent_documents': []})))
    val = tmp_path / 'val.tsv'
    with val.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'ucsf_document_id', 'image', 'index'], delimiter='\t')
        writer.writeheader()
        writer.writerow(dict(image_path='validation.png', ucsf_document_id='VAL_PARENT',
            image=base64.b64encode(image_bytes(250)).decode(), index='999'))
    rows = []
    for number in range(count):
        content = image_bytes(20 + number)
        if duplicate_pixels and number == 1:
            content = image_bytes(20, 'different PNG bytes, identical RGB pixels')
        parent, page = f'parent_{number}', f'page_{number}.png'
        rows.extend([
            row(parent, page, number * 10, 'What is the quantity?', ['12', '12.0'], content),
            row(parent, page, number * 10 + 1, 'Which organization issued this?', ['Example Ltd', 'Example'], content),
            row(parent, page, number * 10 + 2, 'What is the department?', ['Research'], content),
            # Distinct IDs do not make a repeated normalized question independent.
            row(parent, page, number * 10 + 3, '  WHICH organization issued this? ', ['Example Ltd'], content),
        ])
    pq.write_table(pa.Table.from_pylist(rows), shard)
    if historical_parent is not None:
        path = history / 'student_distill/data/test_manifest.json'
        path.parent.mkdir(parents=True)
        image_path = tmp_path / 'historical.png'
        image_path.write_bytes(image_bytes(20 + historical_parent))
        path.write_text(json.dumps({'records': [dict(parent_id=f'parent_{historical_parent}',
            image_path=str(image_path), image_sha256=data.sha256(image_path))]}))
    args = argparse.Namespace(train_root=train, val_tsv=val, history_root=history,
        extra_manifest=[], out=out, seed=17, fit=2, val=1, test=2, dev=1,
        questions_per_page=3)
    return args, shard


def read_manifest(args):
    return json.loads(data.build(args).read_text())


def all_pages(manifest):
    return [page for pages in manifest['splits'].values() for page in pages]


def test_parent_splits_original_aliases_and_distinct_questions(tmp_path):
    args, _ = fixture(tmp_path, historical_parent=2)
    manifest = read_manifest(args)
    pages = all_pages(manifest)
    assert len(pages) == len({page['parent_id'] for page in pages}) == 6
    assert 'parent_2' not in {page['parent_id'] for page in pages}
    assert manifest['counts']['questions'] == 18
    assert manifest['candidate_inventory']['duplicate_questions_skipped'] == 10
    for page in pages:
        questions = page['questions']
        assert len(questions) == len({data.question_key(q['question']) for q in questions}) == 3
        assert any(not q['numeric'] for q in questions)
        assert next(q for q in questions if q['numeric'])['answers'] == ['12', '12.0']
        assert Path(page['image_path']).read_bytes() == image_bytes(20 + int(page['parent_id'].split('_')[-1]))
    assert not manifest['protocol']['selection_uses_model_outputs']


def test_pixels_are_distinct_from_encoding_and_dimensions():
    a, b = image_bytes(7), image_bytes(7, 'metadata')
    assert hashlib.sha256(a).digest() != hashlib.sha256(b).digest()
    assert data.pixel_sha256(a) == data.pixel_sha256(b)
    image = Image.new('RGB', (5, 7), (7, 3, 248))
    output = io.BytesIO()
    image.save(output, format='PNG')
    assert data.pixel_sha256(a) != data.pixel_sha256(output.getvalue())


def test_bounded_parallel_history_hashes_match_serial_and_preserve_order(tmp_path):
    args, shard = fixture(tmp_path, count=20)
    pages, _ = data._annotations([shard])
    rows = list(pages.values())
    expected = [(page['page_id'], hashlib.sha256(content).hexdigest(), data.pixel_sha256(content))
                for page, content in data._image_payloads(rows)]
    actual = [(page['page_id'], image_hash, pixel_hash)
              for page, image_hash, pixel_hash in data._historical_image_hashes(rows)]
    assert actual == expected


def test_duplicate_pixels_are_skipped_and_requested_counts_replenished(tmp_path):
    args, _ = fixture(tmp_path, duplicate_pixels=True, count=7)
    args.fit, args.test = 3, 1  # six unique pages, so every candidate is considered
    manifest = read_manifest(args)
    pages = all_pages(manifest)
    assert len(pages) == 6
    assert len({p['pixel_sha256'] for p in pages}) == 6
    assert any(r['reason'] == 'duplicate_RGB_pixels' for r in manifest['exclusions']['rejected_candidates'])
    assert not {'parent_0', 'parent_1'}.issubset({p['parent_id'] for p in pages})


def test_resume_verifies_identical_freeze_and_rejects_config_change(tmp_path):
    args, _ = fixture(tmp_path)
    path = data.build(args)
    original, stat = path.read_bytes(), path.stat().st_mtime_ns
    assert data.build(args) == path
    assert path.read_bytes() == original and path.stat().st_mtime_ns == stat
    args.seed += 1
    with pytest.raises(ValueError, match='protocol differs'):
        data.build(args)
    assert path.read_bytes() == original


@pytest.mark.parametrize('mutation', ['image', 'source', 'inventory'])
def test_resume_rejects_mutated_inputs(tmp_path, mutation):
    args, shard = fixture(tmp_path)
    manifest = read_manifest(args)
    if mutation == 'image':
        Path(all_pages(manifest)[0]['image_path']).write_bytes(image_bytes(233))
        pattern = 'Frozen image changed'
    elif mutation == 'source':
        with shard.open('ab') as stream:
            stream.write(b'changed source')
        pattern = 'Frozen source changed'
    else:
        path = args.history_root / 'new_manifest.json'
        path.write_text('{"records": []}')
        pattern = 'inventory changed'
    with pytest.raises(ValueError, match=pattern):
        data.build(args)


def test_no_silent_scale_reduction(tmp_path):
    args, _ = fixture(tmp_path, duplicate_pixels=True, count=6)
    with pytest.raises(ValueError, match='need exactly 6'):
        data.build(args)
    assert not (args.out / 'manifest.json').exists()


def test_missing_history_is_not_silently_ignored(tmp_path):
    args, _ = fixture(tmp_path)
    (args.history_root / 'catchup_rerank/data/manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        data.build(args)


def test_excludes_old_fit_roster_and_cross_parent_pixel_duplicates(tmp_path):
    args, _ = fixture(tmp_path, duplicate_pixels=True)
    roster = args.train_root / 'train_audit_001/rows.jsonl'
    roster.write_text(json.dumps({'split': 'fit', 'document_id': 'parent_0'}) + '\n')
    manifest = read_manifest(args)
    selected = {p['parent_id'] for p in all_pages(manifest)}
    assert 'parent_0' not in selected
    assert 'parent_1' not in selected  # same RGB image under a different parent


def test_validation_pixels_excluded_even_under_different_parent_and_encoding(tmp_path):
    args, _ = fixture(tmp_path)
    with args.val_tsv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'ucsf_document_id', 'image', 'index'], delimiter='\t')
        writer.writeheader()
        writer.writerow(dict(image_path='val.png', ucsf_document_id='VAL_PARENT',
            image=base64.b64encode(image_bytes(20, 'reencoded')).decode(), index='999'))
        writer.writerow(dict(image_path='val_copy.png', ucsf_document_id='VAL_PARENT', image='999', index='1000'))
    manifest = read_manifest(args)
    assert 'parent_0' not in {p['parent_id'] for p in all_pages(manifest)}


def test_import_does_not_load_model_or_dataframe_stack():
    result = subprocess.run([sys.executable, '-c',
        'import sys; from scripts.paired_pruning import data; '
        'assert not any(n in sys.modules for n in ["torch", "transformers", "pandas", "pyarrow"])'],
        cwd=data.QWEN, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_cli_progress_is_on_stderr_and_stdout_remains_json(tmp_path):
    args, _ = fixture(tmp_path)
    command = [sys.executable, '-m', 'scripts.paired_pruning.data',
        '--out', str(args.out), '--train-root', str(args.train_root),
        '--val-tsv', str(args.val_tsv), '--history-root', str(args.history_root),
        '--fit', '2', '--val', '1', '--test', '2', '--dev', '1']
    result = subprocess.run(command, cwd=data.QWEN, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload['manifest'] == str(args.out / 'manifest.json')
    assert 'annotations shard 1/1' in result.stderr
    assert 'selected 6/6' in result.stderr
    assert 'completed 6 pages / 18 questions' in result.stderr
