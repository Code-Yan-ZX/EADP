#!/usr/bin/env python3
"""CPU audit of the exact source-linked POPE release and preserved predictions."""
import collections
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile
import sys

sys.dont_write_bytecode = True

OUT = Path(__file__).resolve().parent
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
OLD = Path('/media/disk2/YZX/research/EADP_amp')
EVAL = OLD / 'LLaVA/playground/data/eval'
LABELS = OUT / 'pinned_pope_labels'
INPUTS = OUT / 'inputs'
INPUTS.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def describe(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def norm(text):
    if '.' in text:
        text = text.split('.')[0]
    words = text.replace(',', '').split(' ')
    return 0 if any(w in words for w in ['No', 'not', 'no']) else 1


def main():
    with zipfile.ZipFile(EVAL / 'eval.zip') as z:
        matches = [x for x in z.namelist() if x.endswith('pope/llava_pope_test.jsonl')]
        assert len(matches) == 1
        member = matches[0]
        data = z.read(member)
    qpath = INPUTS / 'llava_pope_test.jsonl'
    qpath.write_bytes(data)
    assert data == (EVAL / 'pope/llava_pope_test.jsonl').read_bytes()
    questions = [json.loads(x) for x in data.splitlines()]
    q_by_id = {q['question_id']: q for q in questions}
    assert len(q_by_id) == len(questions) == 8910
    report = {'official_commit': 'e1a08801461c181871b9ff1cb0803b8e9966f083',
              'question_input': describe(qpath), 'eval_zip': {'path': str(EVAL / 'eval.zip'), 'member': member,
              'member_identical_to_old_input': True}, 'pinned_pope_commit': 'e3e39262c85a6a83f26cf5094022a782cb0df58d',
              'categories': {}, 'historical_rescores': {}}
    for category in ['random', 'popular', 'adversarial']:
        gtpath = LABELS / f'coco_pope_{category}.json'
        gt = [json.loads(x) for x in gtpath.read_text().splitlines()]
        q = [x for x in questions if x['category'] == category]
        assert len(q) == len(gt)
        mismatches = []
        mapping = []
        for i, (question, label) in enumerate(zip(q, gt)):
            exact = question['image'] == label['image'] and question['text'].replace('\nAnswer the question using a single word or phrase.', '') == label['text']
            if not exact:
                mismatches.append(i)
            mapping.append({'position': i, 'question_id': question['question_id'], 'label_question_id': label['question_id'],
                            'image': question['image'], 'question': label['text'], 'label': label['label'], 'exact_match': exact})
        assert not mismatches
        mpath = OUT / f'{category}_question_label_mapping.json'
        mpath.write_text(json.dumps(mapping, indent=2) + '\n')
        report['categories'][category] = {'labels': describe(gtpath), 'question_count': len(q), 'label_count': len(gt),
            'label_counts': dict(collections.Counter(x['label'] for x in gt)), 'position_identity_mismatches': mismatches,
            'download_url': f'https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_{category}.json',
            'question_label_mapping': describe(mpath), 'newer_local_gt_sha256': sha(EVAL / 'pope' / gtpath.name)}
    images = sorted({q['image'] for q in questions})
    image_root = EVAL / 'pope/val2014'
    image_manifest = [{'filename': name, **describe(image_root / name)} for name in images]
    ipath = OUT / 'canonical_image_manifest.json'
    ipath.write_text(json.dumps(image_manifest, indent=2) + '\n')
    report['images'] = {'root': str(image_root), 'unique_images': len(images), 'all_present': True, 'manifest': describe(ipath)}
    source = AUTHOR / 'LLaVA/llava/eval/eval_pope.py'
    spec = importlib.util.spec_from_file_location('author_eval_pope', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report['official_evaluator'] = describe(source)
    cases = {
        'EADP32': (OLD / 'Qwen_vl/outputs/audit_followup_20261008/rerun_fast/pope_v15_EADP32_streamwait.jsonl', 84.07359966443313),
        'EADP128': (OLD / 'Qwen_vl/outputs/audit_followup_20261008/streamwait_continuation_20261008/v15_pope_EGATHER_streamwait_attempt2.jsonl', 87.51959583686018),
    }
    for name, (path, old_score) in cases.items():
        answers = [json.loads(x) for x in path.read_text().splitlines()]
        ids = [x['question_id'] for x in answers]
        assert len(ids) == len(set(ids)) == len(questions) and set(ids) == set(q_by_id)
        case = {'predictions': describe(path), 'n_predictions': len(answers), 'previous_identity_join_macro_f1_percent': old_score, 'categories': {}}
        for category in ['random', 'popular', 'adversarial']:
            selected = [x for x in answers if q_by_id[x['question_id']]['category'] == category]
            expected_ids = [q['question_id'] for q in questions if q['category'] == category]
            assert [x['question_id'] for x in selected] == expected_ids
            gtpath = LABELS / f'coco_pope_{category}.json'
            gt = [json.loads(x) for x in gtpath.read_text().splitlines()]
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                result = module.eval_pope(copy.deepcopy(selected), str(gtpath))
            logpath = OUT / f'{name}_{category}_official_stdout.txt'
            logpath.write_text(capture.getvalue())
            counts = collections.Counter((norm(a['text']), int(g['label'] != 'no')) for a, g in zip(selected, gt))
            tp, fp, tn, fn = [counts[k] for k in [(1, 1), (1, 0), (0, 0), (0, 1)]]
            independent_f1 = 2 * tp / (2 * tp + fp + fn)
            assert independent_f1 == result['f1'] or abs(independent_f1 - result['f1']) < 1e-15
            case['categories'][category] = {**result, 'n_predictions': len(selected), 'n_labels': len(gt),
                'order_matches_canonical': True, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'stdout': describe(logpath)}
        case['official_pinned_macro_f1_percent'] = sum(x['f1'] for x in case['categories'].values()) / 3 * 100
        case['pinned_minus_previous_percentage_points'] = case['official_pinned_macro_f1_percent'] - old_score
        report['historical_rescores'][name] = case
    report['conclusion'] = 'The 8910 question input is byte-identical to author-linked LLaVA eval.zip and positionally identical to author-pinned POPE labels (2910/3000/3000). Newer local labels have a different revision; identity matching already preserved the correct historical scores. Re-scoring existing predictions with untouched author evaluator and exact pinned labels does not explain the POPE K32 gap. No question fill or GPU generation was performed by this CPU audit.'
    (OUT / 'input_and_score_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'canonical_counts': {k: v['question_count'] for k, v in report['categories'].items()},
                      'images': len(images), 'rescores': {k: {'pinned_f1': v['official_pinned_macro_f1_percent'], 'delta_vs_previous': v['pinned_minus_previous_percentage_points']} for k, v in report['historical_rescores'].items()}}))


if __name__ == '__main__':
    main()
