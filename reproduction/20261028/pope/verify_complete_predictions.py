#!/usr/bin/env python3
"""Require all canonical POPE rows; untouched author CLI plus independent counts."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
QPATH = ROOT / 'inputs/llava_pope_test.jsonl'
LABELS = ROOT / 'pinned_pope_labels'
SCORER = AUTHOR / 'LLaVA/llava/eval/eval_pope.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yes(text):
    words = text.split('.')[0].replace(',', '').split(' ')
    return int(not any(word in words for word in ['No', 'not', 'no']))


def verify(prediction_path, output_dir):
    questions = [json.loads(x) for x in QPATH.read_text().splitlines()]
    raw = prediction_path.read_bytes()
    assert raw.endswith(b'\n'), 'incomplete last record'
    answers = [json.loads(x) for x in raw.splitlines()]
    assert len(answers) == len(questions) == 8910, 'full8910 denominator required'
    assert len({a['question_id'] for a in answers}) == 8910
    assert [a['question_id'] for a in answers] == [q['question_id'] for q in questions]
    assert all(a['prompt'] == q['text'] for a, q in zip(answers, questions))
    assert all(isinstance(a['text'], str) and not a['text'].startswith('FAILED') for a in answers)
    categories = {}
    for category, n in [('random', 2910), ('popular', 3000), ('adversarial', 3000)]:
        selected = [(a, q) for a, q in zip(answers, questions) if q['category'] == category]
        labels = [json.loads(x) for x in (LABELS / f'coco_pope_{category}.json').read_text().splitlines()]
        assert len(selected) == len(labels) == n
        for (_, q), label in zip(selected, labels):
            assert q['image'] == label['image']
            assert q['text'].replace('\nAnswer the question using a single word or phrase.', '') == label['text']
        counts = collections.Counter((yes(a['text']), int(label['label'] != 'no')) for (a, _), label in zip(selected, labels))
        tp, fp, tn, fn = (counts[key] for key in [(1, 1), (1, 0), (0, 0), (0, 1)])
        categories[category] = {'n': n, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'f1_percent': 200 * tp / (2 * tp + fp + fn),
            'pinned_label_sha256': sha(LABELS / f'coco_pope_{category}.json')}
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, str(SCORER), '--annotation-dir', str(LABELS), '--question-file', str(QPATH), '--result-file', str(prediction_path)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    scored = subprocess.run(command, cwd=AUTHOR / 'LLaVA', env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output_dir / 'official_score.log').write_text(scored.stdout)
    assert scored.returncode == 0, scored.stdout
    official = float(re.search(r'Average F1 score:\s*([0-9.eE+-]+)', scored.stdout).group(1)) * 100
    independent = sum(row['f1_percent'] for row in categories.values()) / 3
    assert abs(official - independent) < 1e-12
    repeated = collections.defaultdict(list)
    for a, q in zip(answers, questions):
        repeated[(q['image'], q['text'])].append(yes(a['text']))
    groups = [values for values in repeated.values() if len(values) > 1]
    result = {'complete': True, 'n_predictions': 8910, 'all_question_ids_order_verified': True,
        'all_raw_prompts_verified': True, 'all_pinned_image_question_positions_verified': True,
        'prediction_path': str(prediction_path), 'predictions_sha256': sha(prediction_path),
        'predictions_bytes': len(raw), 'source_scorer_sha256': sha(SCORER),
        'canonical_questions_sha256': sha(QPATH), 'official_score_command': command,
        'official_scorer_exit_code': scored.returncode,
        'official_macro_f1_percent': official, 'independent_macro_f1_percent': independent,
        'categories': categories, 'same_input_repeated_groups': len(groups),
        'same_input_yes_no_conflict_groups': sum(len(set(values)) > 1 for values in groups),
        'empty_answers_preserved_in_official_denominator': sum(a['text'] == '' for a in answers),
        'model_ids': dict(collections.Counter(a.get('model_id') for a in answers)),
        'score_log_sha256': sha(output_dir / 'official_score.log')}
    (output_dir / 'full_prediction_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.predictions, args.output_dir)
    print(json.dumps({key: result[key] for key in ['complete', 'n_predictions', 'official_macro_f1_percent',
        'independent_macro_f1_percent', 'predictions_sha256', 'same_input_yes_no_conflict_groups']}))


if __name__ == '__main__':
    main()
