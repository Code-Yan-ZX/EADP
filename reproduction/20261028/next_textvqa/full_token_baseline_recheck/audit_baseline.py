#!/usr/bin/env python3
"""Re-score the complete preserved NeXT FULL baseline and pair existing runs.

CPU only. This never constructs a model, regenerates answers, or changes a
prediction/protocol. Shared author answer normalization is explicit; the second
consensus-score implementation uses the closed-form count formula.
"""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
ROOT = NEXT.parents[2]
AUTHOR = ROOT / '_reproduction/author_eadp/LLaVA'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def describe(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    evidence_path = ROOT / 'docs/versions/v2/evidence.json'
    evidence = read(evidence_path)['records']
    local = [r for r in evidence if r['id'] == 'local:table:next:0:FULL (reproduced):TextVQA']
    reported = [r for r in evidence if r['id'] == 'paper:T2:0:FULL:TextVQA:as_reported']
    assert len(local) == len(reported) == 1
    local, reported = local[0], reported[0]
    raw_path = Path(local['raw_prediction']['path'])
    stem = raw_path.with_suffix('')
    score_path = Path(str(stem) + '.official.score.json')
    control_path = Path(str(stem) + '.control.protocol.json')
    protocol_path = Path(str(raw_path) + '.protocol.json')
    runtime_path = Path(str(raw_path) + '.runtime.jsonl')
    finished_path = Path(str(stem) + '.finished.json')
    score, control, protocol, finished = map(read, [score_path, control_path, protocol_path, finished_path])
    assert sha(raw_path) == local['raw_prediction']['sha256'] == score['prediction_sha256'] == finished['prediction_sha256']
    assert sha(score_path) == local['source_sha256'] == finished['score_sha256']
    assert sha(control_path) == score['control_protocol_sha256']
    assert sha(runtime_path) == score['runtime_sha256']
    assert score['success'] and finished['success'] and finished['complete'] and finished['generation_exit_code'] == 0
    assert score['n'] == control['n'] == finished['n'] == 5000
    assert score['method'] == control['method'] == 'FULL'
    assert protocol['anchorzip'] is False and protocol['visual_token_num'] == score['visual_token_num'] == 0
    for name, value in [('visual_token_num', 0), ('alpha', .5), ('beta', 2.), ('temperature', 0.), ('max_new_tokens', 128), ('num_beams', 1), ('conv_mode', 'vicuna_v1')]:
        assert protocol[name] == control['parameters'][name] == value
    for relative, digest in protocol['source_sha256'].items():
        assert sha(AUTHOR / relative) == digest
    model = Path(protocol['model_path'])
    assert sha(model / 'config.json') == protocol['config.json_sha256']
    assert sha(model / 'generation_config.json') == protocol['generation_config.json_sha256']
    plan_path = NEXT / 'full_default_K128.plan.json'
    plan = read(plan_path)
    questions_path = Path(plan['questions']['path'])
    gt_path = Path(plan['annotation']['path'])
    assert sha(questions_path) == protocol['question_file_sha256'] == plan['questions']['sha256']
    assert sha(gt_path) == plan['annotation']['sha256']
    assert control['AST_validation']['inserted_nodes'] == plan['waits']
    image_manifest_path = Path(control['image_manifest'])
    assert sha(image_manifest_path) == score['image_manifest_sha256'] == control['image_manifest_sha256']
    images = read(image_manifest_path)['records']
    assert len(images) == 3166 and all(sha(r['path']) == r['sha256'] for r in images)
    paired_dir = NEXT / 'beta1_vs_historical_beta2_full'
    paired_report_path = paired_dir / 'comparison_report.json'
    paired_report = read(paired_report_path)
    assert paired_report['success'] and paired_report['n_paired'] == 5000
    paired_path = paired_dir / 'per_question_pairs.jsonl'
    assert sha(paired_path) == paired_report['paired_records']['sha256']
    predictions, trace, questions, paired = map(rows, [raw_path, runtime_path, questions_path, paired_path])
    assert len(predictions) == len(trace) == len(questions) == len(paired) == 5000
    canonical = [(str(q['question_id']), q['text']) for q in questions]
    assert len(set(canonical)) == 5000
    assert [(str(p['question_id']), p['prompt']) for p in predictions] == canonical
    assert all(isinstance(p['text'], str) and p['text'].strip() and not p['text'].startswith('FAILED') for p in predictions)
    assert all(p['model_id'] == 'llava-v1.6-vicuna-7b' for p in predictions)
    for i, (q, t, pair) in enumerate(zip(questions, trace, paired)):
        assert i == t['question_position'] == pair['position']
        assert str(q['question_id']) == str(t['question_id']) == str(pair['question_id'])
        assert q['image'] == t['image'] == pair['image']
        assert hashlib.sha256(q['text'].encode()).hexdigest() == t['prompt_sha256'] == pair['prompt_sha256']
        assert t['actual_visual_tokens_retained'] == 2880 and t['visual_token_budget_parameter'] == 0
        assert t['image_tensor_shape'] == [1, 5, 3, 336, 336]
        assert t['source_wrapper_sha256'] == protocol['wrapper_sha256']
        assert t['input_sha256'] == protocol['question_file_sha256']
    assert score['actual_count_distribution'] == {'2880': 5000}
    paths = [evidence_path, raw_path, score_path, control_path, protocol_path, runtime_path, finished_path,
             plan_path, questions_path, gt_path, image_manifest_path, paired_path, paired_report_path,
             AUTHOR / 'llava/eval/eval_textvqa.py', AUTHOR / 'llava/eval/m4c_evaluator.py', Path(__file__).resolve()]
    registry = [describe(p) for p in paths]
    sys.path.insert(0, str(AUTHOR))
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    assert inspect.getfile(TextVQAAccuracyEvaluator) == str(AUTHOR / 'llava/eval/m4c_evaluator.py')
    evaluator = TextVQAAccuracyEvaluator()
    gt = {(str(g['image_id']), g['question'].lower()): g for g in read(gt_path)['data']}
    keys = [(str(p['question_id']), prompt_processor(p['prompt'])) for p in predictions]
    assert len(set(keys)) == len(gt) == 5000 and set(keys) == set(gt)
    full_value = 100 * evaluator.eval_pred_list([dict(pred_answer=p['text'], gt_answers=gt[k]['answers']) for p, k in zip(predictions, keys)])
    assert abs(full_value - score['accuracy_percent']) < 1e-10 and abs(full_value - local['value']) < 1e-10
    command = [PYTHON, '-m', 'llava.eval.eval_textvqa', '--annotation-file', str(gt_path), '--result-file', str(raw_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(AUTHOR), PYTHONDONTWRITEBYTECODE='1')
    cli = subprocess.run(command, cwd=AUTHOR, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cli_log = OUT / 'fresh_author_cli.txt'
    cli_log.write_text(cli.stdout)
    assert cli.returncode == 0 and 'Samples: 5000' in cli.stdout
    assert Decimal(re.search(r'Accuracy: ([0-9.]+)%', cli.stdout).group(1)) == Decimal(str(full_value)).quantize(Decimal('.01'))
    totals = Counter()
    count = {name: Counter() for name in ['beta1', 'beta2']}
    gains = Counter()
    losses = Counter()
    max_independent_delta = 0.
    records_path = OUT / 'per_question_full_vs_pruned.jsonl'
    with records_path.open('w') as stream:
        for i, (p, k, pair) in enumerate(zip(predictions, keys, paired)):
            assert list(k) == pair['GT_key']
            answers = [evaluator.answer_processor(a) for a in gt[k]['answers']]
            assert len(answers) == 10
            pred = evaluator.answer_processor(p['text'])
            n = Counter(answers)[pred]
            independent = (n * min(1., max(n - 1, 0) / 3.) + (10 - n) * min(1., n / 3.)) / 10.
            official = evaluator._compute_answer_scores(gt[k]['answers']).get(pred, 0.)
            max_independent_delta = max(max_independent_delta, abs(independent - official))
            assert abs(independent - official) < 1e-12
            full_units = round(independent * 10)
            totals['full'] += full_units
            record = dict(position=i, question_id=p['question_id'], prompt_sha256=pair['prompt_sha256'], full_answer=p['text'], full_score_tenths=full_units)
            for name, field in [('beta1', 'new_official_score'), ('beta2', 'old_official_score')]:
                units = round(pair[field] * 10)
                delta = units - full_units
                totals[name] += units
                count[name]['improved' if delta > 0 else 'declined' if delta < 0 else 'unchanged'] += 1
                gains[name] += max(delta, 0)
                losses[name] += min(delta, 0)
                record[name + '_score_tenths'] = units
                record[name + '_minus_full_tenths'] = delta
            stream.write(json.dumps(record) + '\n')
    exact = {k: Decimal(v) / Decimal(500) for k, v in totals.items()}
    assert abs(float(exact['full']) - full_value) < 1e-10
    for name in ['beta1', 'beta2']:
        assert exact[name] == Decimal(paired_report['exact_consensus_accuracy_percent'][name])
        assert sum(count[name].values()) == 5000
        assert gains[name] + losses[name] == totals[name] - totals['full']
    paper_full = Decimal(str(reported['value']))
    paper_pruned = Decimal('59.2')
    import torch
    assert not torch.cuda.is_initialized()
    assert all(sha(r['path']) == r['sha256'] for r in registry)
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), success=True, n=5000,
        purpose='Full-token baseline and full paired-score diagnosis; no new generation or paper-recipe recovery claim.',
        actual_full_retained_count_distribution={'2880': 5000}, source_and_runtime_provenance_verified=True,
        ordered_IDs_full_prompts_all_GT_and3166images_verified=True, author_full_accuracy_percent=full_value,
        independent_full_accuracy_percent=str(exact['full']), max_per_question_author_independent_delta=max_independent_delta,
        paper_full_percent=str(paper_full), full_minus_paper_percent=str(exact['full']-paper_full),
        full_matches_paper_one_decimal_rounding=exact['full'].quantize(Decimal('.1')) == paper_full,
        paper_nominal640_pruning_loss_pp=str(paper_full-paper_pruned),
        paired_complete_runs={name:dict(accuracy_percent=str(exact[name]), local_pruning_loss_pp=str(exact['full']-exact[name]),
            excess_loss_over_paper_pp=str((exact['full']-exact[name])-(paper_full-paper_pruned)),
            compared_to_full_counts=dict(count[name]), score_gain_tenths=gains[name], score_loss_tenths_signed=losses[name]) for name in ['beta1','beta2']},
        author_cli=dict(command=command, exit_code=cli.returncode, **describe(cli_log)),
        records=describe(records_path), sources_before_after_unchanged=registry, cuda_initialized=False,
        limits=['The full baseline is a preserved historical run, freshly re-scored with the unmodified author evaluator; no new FULL generation was performed.',
                'A small FULL baseline gap supports investigating the pruning path but does not prove every model/runtime/prompt factor irrelevant.',
                'Pairing complete runs quantifies answer-score changes; historical/new wrappers and low-level scheduling were not replayed.',
                'The 60.370 FULL baseline differs by +0.070 from published60.3 and rounds to60.4, so exact published rounding is not claimed.',
                'The original paper main-table argv remains unverified and the reproduction goal is not achieved.'])
    (OUT / 'baseline_recheck.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ['success','n','independent_full_accuracy_percent','full_minus_paper_percent','full_matches_paper_one_decimal_rounding','paired_complete_runs']}))


if __name__ == '__main__':
    main()
