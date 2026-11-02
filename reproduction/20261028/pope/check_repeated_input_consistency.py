#!/usr/bin/env python3
"""Diagnose same-input output consistency without computing any accuracy score."""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT / 'run01_source_as_is_k32'


def as_yes(text):
    text = text.split('.')[0].replace(',', '')
    words = text.split(' ')
    return not any(w in words for w in ['No', 'not', 'no'])


def main():
    questions = {q['question_id']: q for q in [json.loads(x) for x in (OUT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]}
    images = {r['filename']: r['sha256'] for r in json.loads((OUT / 'canonical_image_manifest.json').read_text())}
    data = (RUN / 'predictions.jsonl').read_bytes()
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    answers = [json.loads(x) for x in lines]
    groups = collections.defaultdict(list)
    for answer in answers:
        q = questions[answer['question_id']]
        assert answer['prompt'] == q['text']
        # Same model/conv template and exact raw user input give identical full
        # multimodal prompt tokens; the source adds the same image prefix.
        key = (images[q['image']], q['text'])
        groups[key].append({'question_id': answer['question_id'], 'category': q['category'],
                            'image_filename': q['image'], 'text': answer['text'], 'parsed_yes': as_yes(answer['text'])})
    repeated = {key: values for key, values in groups.items() if len(values) > 1}
    conflicts = [(key, values) for key, values in repeated.items() if len({x['parsed_yes'] for x in values}) > 1]
    report = {'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'run_id': RUN.name,
        'n_complete_prediction_lines': len(answers), 'n_distinct_image_prompt_groups': len(groups),
        'n_repeated_groups': len(repeated), 'n_repeated_rows': sum(len(v) for v in repeated.values()),
        'n_yes_no_conflict_groups': len(conflicts), 'per_category_observed_counts': dict(collections.Counter(questions[a['question_id']]['category'] for a in answers)),
        'diagnostic_only_no_accuracy_or_ground_truth_used': True,
        'identity_rule': 'Exact source image SHA256 and exact full raw user prompt; same source model and vicuna_v1 template give identical full LLM prompt tokens and identical CLIP guidance. No labels are read.',
        'conflicts': [{'image_sha256': key[0], 'raw_prompt': key[1],
                       'raw_prompt_sha256': hashlib.sha256(key[1].encode()).hexdigest(), 'rows': rows}
                      for key, rows in conflicts],
        'interpretation': 'A parsed Yes/No disagreement on these identical deterministic inputs invalidates stable-generation attribution. It is not by itself an isolated proof of which stream operation caused the difference. Keep source-as-is baseline, then evaluate a separately declared wait-only control with the same source recipe.'}
    path = RUN / 'same_input_consistency.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'conflicts'}))


if __name__ == '__main__':
    main()
