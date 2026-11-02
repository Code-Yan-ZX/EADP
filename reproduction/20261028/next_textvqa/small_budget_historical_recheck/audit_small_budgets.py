#!/usr/bin/env python3
"""Complete only missing CPU pointwise gates for three preserved full runs."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
NEXT = OUT.parent
VERSION = NEXT.parents[2]
AUTHOR = VERSION / '_reproduction/author_eadp/LLaVA'
OLD = Path('/media/disk2/YZX/research/EADP_amp/Qwen_vl/outputs/audit_followup_20261008')
RUNS = [
    ('K64_beta2', 64, 2., OLD / 'streamwait_continuation_20261008/next_text_E64_streamwait'),
    ('K32_beta2', 32, 2., OLD / 'next_gap_diagnosis_20261008/stability/next_text_K32_EADP_beta2_streamwait'),
    ('K32_beta1', 32, 1., OLD / 'next_gap_diagnosis_20261008/stability/next_text_K32_EADP_beta1_streamwait')]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def describe(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def consensus(n):
    assert type(n) is int and 0 <= n <= 10
    return (n * min(max(n - 1, 0) / 3, 1) + (10 - n) * min(n / 3, 1)) / 10


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan_path = NEXT / 'full_default_K128.plan.json'; plan = read(plan_path)
    questions_path, gt_path = Path(plan['questions']['path']), Path(plan['annotation']['path'])
    assert sha(questions_path) == plan['questions']['sha256']
    assert sha(gt_path) == plan['annotation']['sha256']
    questions = rows(questions_path)
    canonical = [(str(q['question_id']), q['text']) for q in questions]
    assert len(questions) == len(set(canonical)) == 5000
    canonical_manifest_path = NEXT / 'full_default_K128.images.json'
    canonical_images = {Path(row['path']).name: row['sha256'] for row in read(canonical_manifest_path)}
    assert len(canonical_images) == 3166
    sys.path.insert(0, str(AUTHOR))
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    evaluator = TextVQAAccuracyEvaluator()
    assert inspect.getfile(TextVQAAccuracyEvaluator) == str(AUTHOR / 'llava/eval/m4c_evaluator.py')
    annotations = read(gt_path)['data']
    by_key = {(str(row['image_id']), row['question'].lower()): row for row in annotations}
    gt_keys = [(str(q['question_id']), prompt_processor(q['text'])) for q in questions]
    assert len(by_key) == len(set(gt_keys)) == 5000 and set(gt_keys) == set(by_key)
    reports = []
    for name, K, beta, stem in RUNS:
        output = OUT / name; output.mkdir(exist_ok=True)
        paths = {'predictions': Path(str(stem) + '.jsonl'), 'runtime': Path(str(stem) + '.jsonl.runtime.jsonl'),
                 'score': Path(str(stem) + '.official.score.json'), 'control': Path(str(stem) + '.control.protocol.json'),
                 'protocol': Path(str(stem) + '.jsonl.protocol.json'), 'finished': Path(str(stem) + '.finished.json')}
        score, control, protocol, finished = (read(paths[k]) for k in ['score', 'control', 'protocol', 'finished'])
        assert score['success'] and score['n'] == finished['n'] == 5000
        assert finished['complete'] and finished['success'] and finished['generation_exit_code'] == 0
        assert score['prediction_sha256'] == finished['prediction_sha256'] == sha(paths['predictions'])
        assert score['runtime_sha256'] == sha(paths['runtime']) and score['control_protocol_sha256'] == sha(paths['control'])
        assert finished['score_sha256'] == sha(paths['score'])
        assert protocol['question_file_sha256'] == plan['questions']['sha256'] and protocol['question_rows'] == 5000
        assert score.get('GT_sha256', plan['annotation']['sha256']) == plan['annotation']['sha256']
        assert score.get('official_evaluator_sha256', score.get('evaluator_sha256')) == sha(AUTHOR / 'llava/eval/m4c_evaluator.py')
        for key, expected in [('visual_token_num', K), ('alpha', .5), ('beta', beta), ('temperature', 0.),
                              ('max_new_tokens', 128), ('num_beams', 1), ('conv_mode', 'vicuna_v1')]:
            assert protocol[key] == control['parameters'][key] == expected
        assert protocol['anchorzip'] is False and protocol['model_path'] == plan['model_path']
        assert protocol['torch_version'] == '2.1.2+cu118' and protocol['transformers_version'] == '4.37.2'
        assert sha(Path(plan['model_path']) / 'config.json') == protocol['config.json_sha256']
        for relative, expected in protocol['source_sha256'].items():
            assert sha(AUTHOR / relative) == expected
        assert control['AST_validation']['inserted_nodes'] == plan['waits']
        manifest_path = Path(control['image_manifest'])
        assert sha(manifest_path) == score['image_manifest_sha256'] == control['image_manifest_sha256']
        old_images = {row['image']: row['sha256'] for row in read(manifest_path)['records']}
        assert old_images == canonical_images
        predictions, runtime = rows(paths['predictions']), rows(paths['runtime'])
        assert len(predictions) == len(runtime) == 5000
        assert [(str(p['question_id']), p['prompt']) for p in predictions] == canonical
        assert all(isinstance(p['text'], str) and p['text'].strip() and not p['text'].startswith('FAILED') for p in predictions)
        assert all(p['model_id'] == 'llava-v1.6-vicuna-7b' for p in predictions)
        histogram = Counter()
        for position, (question, trace) in enumerate(zip(questions, runtime)):
            assert trace['question_position'] == position and str(trace['question_id']) == str(question['question_id'])
            assert trace['image'] == question['image'] and trace['prompt_sha256'] == hashlib.sha256(question['text'].encode()).hexdigest()
            assert trace['input_sha256'] == plan['questions']['sha256'] and trace['source_wrapper_sha256'] == protocol['wrapper_sha256']
            assert trace['visual_token_budget_parameter'] == K and trace['image_tensor_shape'] == [1, 5, 3, 336, 336]
            assert type(trace['actual_visual_tokens_retained']) is int and 0 < trace['actual_visual_tokens_retained'] <= K * 5
            histogram[trace['actual_visual_tokens_retained']] += 1
        assert {str(k): v for k, v in histogram.items()} == score['actual_count_distribution']
        registry = [describe(p) for p in paths.values()] + [describe(manifest_path)]
        official_value = 100 * evaluator.eval_pred_list([
            {'pred_answer': pred['text'], 'gt_answers': by_key[key]['answers']} for pred, key in zip(predictions, gt_keys)])
        max_difference = 0.; units = 0
        per_question = output / 'per_question_scores.jsonl'
        with per_question.open('w') as stream:
            for position, (question, pred, key) in enumerate(zip(questions, predictions, gt_keys)):
                gt = by_key[key]['answers']; assert len(gt) == 10
                normalized = evaluator.answer_processor(pred['text'])
                n = sum(evaluator.answer_processor(answer) == normalized for answer in gt)
                independent = consensus(n)
                official = evaluator._compute_answer_scores(gt).get(normalized, 0.)
                error = abs(independent - official); max_difference = max(max_difference, error)
                assert error < 1e-12
                tenths = round(independent * 10); units += tenths
                stream.write(json.dumps({'position': position, 'question_id': question['question_id'], 'image': question['image'],
                    'prompt_sha256': hashlib.sha256(question['text'].encode()).hexdigest(), 'GT_key': list(key),
                    'official_score': official, 'independent_score': independent, 'consensus_units_tenths': tenths,
                    'actual_retained_tokens': runtime[position]['actual_visual_tokens_retained']}) + '\n')
        exact = Decimal(units) / Decimal(500)
        assert abs(float(exact) - official_value) < 1e-10 and abs(official_value - score['accuracy_percent']) < 1e-10
        command = [sys.executable, '-m', 'llava.eval.eval_textvqa', '--annotation-file', str(gt_path), '--result-file', str(paths['predictions'])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(AUTHOR), PYTHONDONTWRITEBYTECODE='1')
        cli = subprocess.run(command, cwd=AUTHOR, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = output / 'fresh_author_cli.txt'; log.write_text(cli.stdout)
        assert cli.returncode == 0 and 'Samples: 5000' in cli.stdout
        assert all(sha(item['path']) == item['sha256'] for item in registry)
        report = {'run': name, 'success': True, 'n': 5000, 'alpha': .5, 'beta': beta, 'budget_cli': K,
            'budget_nominal': K * 5, 'official_value_percent': official_value, 'independent_exact_percent': str(exact),
            'historical_recorded_score': score['accuracy_percent'], 'per_question_scores_verified': 5000,
            'max_per_question_official_independent_difference': max_difference,
            'ordered_full_composite_IDs_prompts_runtime_and_all_GT_verified': True,
            'recorded3166_image_manifest_equal_to_canonical': True, 'individual_images_rehashed': False,
            'current_model_small_config_and_core_source_identity_verified': True,
            'historical_recorded_torch': protocol['torch_version'], 'historical_recorded_transformers': protocol['transformers_version'],
            'runtime_count_histogram': dict(sorted(histogram.items())), 'per_question_artifact': describe(per_question),
            'fresh_author_cli': {'command': command, 'exit_code': cli.returncode, **describe(log)},
            'unchanged_input_registry': registry, 'new_GPU_generation': False,
            'already_closed_before_this_audit': 'Historical complete5000 official scorer/raw/runtime/protocol/GT identities existed; this pass adds explicit fresh-CLI and independent closed-form pointwise evidence rather than regenerating predictions.',
            'limitations': 'Historical full runs remain historical. No output-token decoding proof exists in old traces. No current same-environment or paper-argv identity is inferred; image identity is reused from qualified manifests, not newly rehashed3166payloads.'}
        (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n'); reports.append(report)
    import torch
    assert not torch.cuda.is_initialized()
    report = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(), 'n_runs': 3,
        'n_pointwise_rows': 15000, 'runs': reports,
        'source_registry': [describe(p) for p in [plan_path, questions_path, gt_path, canonical_manifest_path,
             AUTHOR / 'llava/eval/eval_textvqa.py', AUTHOR / 'llava/eval/m4c_evaluator.py', Path(__file__)]],
        'independent_score_method': 'Shared author answer normalization; separate normalized-GT match count and closed-form ten-annotator leave-one-out consensus for every question.',
        'GPU_started': False, 'CUDA_initialized': False, 'old_artifacts_modified': False}
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'success': True, 'n_runs': 3, 'runs': [{k: r[k] for k in ['run', 'n', 'official_value_percent', 'independent_exact_percent', 'max_per_question_official_independent_difference']} for r in reports]}))


if __name__ == '__main__':
    main()
