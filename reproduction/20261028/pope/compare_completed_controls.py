#!/usr/bin/env python3
"""Compare two completed full POPE source recipes, retaining every outcome."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def yes(text):
    return not any(word in text.split('.')[0].replace(',', '').split(' ') for word in ['No', 'not', 'no'])


def describe(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read_complete(name, questions, labels, images):
    run = ROOT / name
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'complete' and state['score_exit_code'] == state['generation_exit_code'] == 0
    prediction_path = run / 'predictions.jsonl'
    answers = [json.loads(x) for x in prediction_path.read_text().splitlines()]
    assert len(answers) == 8910 and [x['question_id'] for x in answers] == [x['question_id'] for x in questions]
    groups = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    predictions = {}
    for q, a in zip(questions, answers):
        assert q['text'] == a['prompt']
        value = yes(a['text'])
        counts[q['category']][(value, labels[q['question_id']])] += 1
        groups[(images[q['image']], q['text'])].append({'question_id': q['question_id'], 'category': q['category'], 'parsed_yes': value})
        predictions[q['question_id']] = value
    categories = {}
    for category, counter in counts.items():
        tp, fp, tn, fn = [counter[k] for k in [(True, True), (True, False), (False, False), (False, True)]]
        categories[category] = {'n': sum(counter.values()), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'f1_percent': 100 * 2 * tp / (2 * tp + fp + fn)}
    independent = sum(x['f1_percent'] for x in categories.values()) / 3
    score_path = run / 'official_score.log'
    official = float(re.search(r'Average F1 score: ([\d.eE+-]+)', score_path.read_text()).group(1)) * 100
    assert abs(official - independent) < 1e-12
    repeated = [v for v in groups.values() if len(v) > 1]
    conflicts = [v for v in repeated if len({a['parsed_yes'] for a in v}) > 1]
    result = {'run_id': name, 'parameters': json.loads((run / 'run_plan.json').read_text())['parameters'],
        'n_predictions': len(answers), 'official_macro_f1_percent': official, 'categories': categories,
        'paper_K32_f1_percent': 86.7, 'local_minus_paper_percentage_points': official - 86.7,
        'same_input_repeated_groups': len(repeated), 'same_input_conflict_groups': len(conflicts),
        'same_input_conflicts': conflicts, 'stable_on_observed_repeated_inputs': not conflicts,
        'predictions': describe(prediction_path), 'official_scorer_stdout': describe(score_path),
        'runtime': state, 'source_plan': describe(run / 'run_plan.json')}
    (run / 'independently_verified_full_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result, predictions


def main():
    questions = [json.loads(x) for x in (ROOT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    labels = {}
    for category in ['random', 'popular', 'adversarial']:
        gt = [json.loads(x) for x in (ROOT / f'pinned_pope_labels/coco_pope_{category}.json').read_text().splitlines()]
        selected = [q for q in questions if q['category'] == category]
        assert len(gt) == len(selected)
        for q, g in zip(selected, gt):
            assert q['image'] == g['image'] and q['text'].split('\nAnswer the question using a single word or phrase.')[0] == g['text']
            labels[q['question_id']] = g['label'] == 'yes'
    images = {r['filename']: r['sha256'] for r in json.loads((ROOT / 'canonical_image_manifest.json').read_text())}
    source, source_predictions = read_complete('run01_source_as_is_k32', questions, labels, images)
    waited, waited_predictions = read_complete('run02_source_wait_only_k32', questions, labels, images)
    transitions = collections.defaultdict(collections.Counter)
    for q in questions:
        key = q['question_id']
        previous, current = source_predictions[key], waited_predictions[key]
        if previous != current:
            transitions[q['category']]['rescue' if current == labels[key] else 'break'] += 1
        else:
            transitions[q['category']]['unchanged'] += 1
    report = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_commit': 'e1a08801461c181871b9ff1cb0803b8e9966f083', 'model': 'LLaVA-1.5-7B', 'budget': 32,
        'canonical_counts': {'random': 2910, 'popular': 3000, 'adversarial': 3000},
        'source_as_is': {k: v for k, v in source.items() if k != 'same_input_conflicts'},
        'wait_only': {k: v for k, v in waited.items() if k != 'same_input_conflicts'},
        'wait_minus_source_percentage_points': waited['official_macro_f1_percent'] - source['official_macro_f1_percent'],
        'paired_answer_transitions_wait_vs_source': {k: dict(v) for k, v in transitions.items()},
        'historical_alpha05_beta2_waited_f1': 84.07359966443313,
        'same_recipe_control_changes': ['caller_stream capture', 'image_stream wait_stream(caller_stream)', 'text_stream wait_stream(caller_stream)'],
        'unmodified': ['model weights', 'alpha1/beta1/K32', 'canonical input8910/images500', 'source arithmetic', 'original trailing synchronize', 'official evaluator/pinned labels'],
        'paper_recipe_caveat': 'The current released task shell sets alpha1/beta1; generic loader/README example setsalpha.5/beta1. The paper does not publish complete main-table argv. Both fresh outcomes are retained; a close macro alone is insufficient if deterministic same-input consistency fails.',
        'proofs': {n: describe(ROOT / n) for n in ['input_and_score_audit.json', 'recipe_and_source_audit.json',
            'model_identity/model_identity_audit.json', 'preprocess_identity_comparison.json', 'sentencepiece_identity_comparison.json',
            'run02_source_wait_only_k32/runtime_patch_proof.json']}}
    (ROOT / 'full_control_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    rows = [('Paper Table1', 86.7, 'reported'), ('Historical alpha.5/beta2, wait', 84.07359966443313, 'preserved'),
            ('Fresh alpha1/beta1, source as-is', source['official_macro_f1_percent'], f"{source['same_input_conflict_groups']}/{source['same_input_repeated_groups']} same-input conflicts"),
            ('Fresh alpha1/beta1, wait only', waited['official_macro_f1_percent'], f"{waited['same_input_conflict_groups']}/{waited['same_input_repeated_groups']} same-input conflicts")]
    lines = ['All8910canonicalquestions were generated and scored with the unchanged author evaluator and exact source-pinned labels. No partial-run scores or higher-result selection were used.', '',
        '| Run | POPE mean F1 (%) | Evidence boundary |', '|---|---:|---|']
    lines += [f'| {name} | {value:.6f} | {note} |' for name, value, note in rows]
    lines += ['', report['paper_recipe_caveat'], '', 'Model/tokenizer/preprocessor payloads match17officialHFfiles. All500image RGB/FP32/FP16 transforms and8910full LLM/CLIP prompt IDs match across tested current/author package pins. Full raw predictions, stdout, source hashes, exact runtime patch proof and per-category TP/FP/TN/FN remain beside this report.']
    (ROOT / 'pope_reproduction_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'source_f1': source['official_macro_f1_percent'], 'wait_f1': waited['official_macro_f1_percent'],
                      'wait_minus_paper': waited['local_minus_paper_percentage_points'],
                      'source_conflicts': source['same_input_conflict_groups'], 'wait_conflicts': waited['same_input_conflict_groups']}))


if __name__ == '__main__':
    main()
