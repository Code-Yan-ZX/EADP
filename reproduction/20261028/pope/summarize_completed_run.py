#!/usr/bin/env python3
"""Summarize only a completed full released-source POPE run; never prefix-score."""
import collections
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
RUN = OUT / 'run01_source_as_is_k32'


def yes(text):
    text = text.split('.')[0].replace(',', '')
    words = text.split(' ')
    return not any(word in words for word in ['No', 'not', 'no'])


def main():
    state = json.loads((RUN / 'state.json').read_text())
    assert state['status'] == 'complete', 'Refusing to score an incomplete/prefix result'
    questions = [json.loads(x) for x in (OUT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    answers_path = RUN / 'predictions.jsonl'
    data = answers_path.read_bytes()
    answers = [json.loads(x) for x in data.splitlines()]
    assert len(answers) == len(questions) == 8910
    assert [a['question_id'] for a in answers] == [q['question_id'] for q in questions]
    question_index = {q['question_id']: q for q in questions}
    old_path = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008/rerun_fast/pope_v15_EADP32_streamwait.jsonl')
    old = {r['question_id']: r for r in [json.loads(x) for x in old_path.read_text().splitlines()]}
    report = {'run_id': RUN.name, 'complete': True, 'prediction_count': len(answers),
              'predictions_sha256': hashlib.sha256(data).hexdigest(), 'state': state, 'categories': {}}
    for category in ['random', 'popular', 'adversarial']:
        selected = [a for a in answers if question_index[a['question_id']]['category'] == category]
        labels = [json.loads(x) for x in (OUT / f'pinned_pope_labels/coco_pope_{category}.json').read_text().splitlines()]
        assert len(selected) == len(labels)
        counts = collections.Counter((yes(a['text']), g['label'] == 'yes') for a, g in zip(selected, labels))
        tp, fp, tn, fn = [counts[k] for k in [(True, True), (True, False), (False, False), (False, True)]]
        flips = collections.Counter()
        for a, g in zip(selected, labels):
            previous, current, correct = yes(old[a['question_id']]['text']), yes(a['text']), g['label'] == 'yes'
            if previous != current:
                flips['rescue' if current == correct else 'break'] += 1
        report['categories'][category] = {'n': len(selected), 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'f1': 2 * tp / (2 * tp + fp + fn),
            'accuracy': (tp + tn) / len(selected), 'yes_ratio': (tp + fp) / len(selected), 'vs_historical_K32': dict(flips)}
    report['independent_macro_f1_percent'] = sum(x['f1'] for x in report['categories'].values()) / 3 * 100
    stdout = (RUN / 'official_score.log').read_text()
    macro = float(re.search(r'Average F1 score: ([\d.eE+-]+)', stdout).group(1)) * 100
    assert abs(macro - report['independent_macro_f1_percent']) < 1e-12
    report['official_macro_f1_percent'] = macro
    report['paper_table1_K32'] = 86.7
    report['local_minus_paper_percentage_points'] = macro - 86.7
    report['historical_K32_macro_f1_percent'] = 84.07359966443313
    report['new_minus_historical_percentage_points'] = macro - 84.07359966443313
    report['scope'] = 'Fresh source-as-is released POPE shell-default recipe alpha1/beta1, compared with paper and prior alpha.5/beta2 wait-corrected run. These differ in both parameters and synchronization; no single-factor causal claim. Canonical 8910 source-linked questions/labels, no prefix scores.'
    (RUN / 'full_result_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
