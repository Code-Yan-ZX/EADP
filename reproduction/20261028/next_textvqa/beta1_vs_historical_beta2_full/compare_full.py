#!/usr/bin/env python3
"""CPU-only paired comparison of two preserved, complete TextVQA runs."""
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
COMPARISON = VERSION / 'docs/versions/v2/eadp_comparison_20261028/comparison.json'
NEW = NEXT / 'full_default_K128_alpha05_beta1_wait'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def describe(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(exist_ok=True)
    reference = [row for row in read(COMPARISON)['rows'] if row['model_key'] == 'next'
        and row['budget_nominal'] == 640 and row['task'] == 'TextVQA']
    assert len(reference) == 1 and reference[0]['status'] == 'qualified_local'
    reference = reference[0]
    old_path = Path(reference['raw_prediction']['path'])
    assert sha(old_path) == reference['raw_prediction']['sha256']
    old_stem = old_path.with_suffix('')
    old_score_path = Path(str(old_stem) + '.official.score.json')
    old_control_path = Path(str(old_stem) + '.control.protocol.json')
    old_protocol_path = Path(str(old_path) + '.protocol.json')
    old_runtime_path = Path(str(old_path) + '.runtime.jsonl')
    old_score, old_control, old_protocol = map(read, [old_score_path, old_control_path, old_protocol_path])
    new_score_path = NEW / 'official.score.json'
    new_score = read(new_score_path)
    plan_path = NEXT / 'full_default_K128.plan.json'
    plan = read(plan_path)
    questions_path, annotation_path = Path(plan['questions']['path']), Path(plan['annotation']['path'])
    assert sha(questions_path) == plan['questions']['sha256'] == old_protocol['question_file_sha256']
    assert sha(annotation_path) == plan['annotation']['sha256']
    assert sha(new_score['plan']['path']) == new_score['plan']['sha256']
    assert old_score['success'] and old_score['n'] == new_score['n'] == new_score['token_decoding_checks'] == 5000
    assert old_score['prediction_sha256'] == sha(old_path)
    assert old_score['control_protocol_sha256'] == sha(old_control_path)
    assert old_score['runtime_sha256'] == sha(old_runtime_path)
    assert old_control['n'] == 5000 and old_control['method'] == 'EADP'
    for key, expected in [('visual_token_num', 128), ('alpha', .5), ('beta', 2.), ('temperature', 0.), ('max_new_tokens', 128), ('num_beams', 1), ('conv_mode', 'vicuna_v1')]:
        assert old_protocol[key] == old_control['parameters'][key] == expected
    for key in ['visual_token_num', 'alpha', 'temperature', 'top_p', 'num_beams', 'max_new_tokens', 'conv_mode']:
        assert old_protocol[key] == new_score['parameters'][key]
    assert new_score['parameters']['beta'] == 1.
    assert old_protocol['model_path'] == plan['model_path']
    assert old_protocol['config.json_sha256'] == sha(Path(plan['model_path']) / 'config.json')
    assert old_protocol['generation_config.json_sha256'] == sha(Path(plan['model_path']) / 'generation_config.json')
    for relative, digest in old_protocol['source_sha256'].items():
        assert sha(AUTHOR / relative) == digest
    assert old_control['AST_validation']['inserted_nodes'] == new_score['stream_wait_repair'] == plan['waits']
    old_images_path = Path(old_control['image_manifest'])
    assert sha(old_images_path) == old_control['image_manifest_sha256'] == old_score['image_manifest_sha256']
    old_images = {row['image']: row['sha256'] for row in read(old_images_path)['records']}
    new_images_path = NEXT / 'full_default_K128.images.json'
    new_images = {Path(row['path']).name: row['sha256'] for row in read(new_images_path)}
    assert old_images == new_images and len(new_images) == 3166
    old_answers, new_answers, questions = rows(old_path), rows(NEW / 'answers.jsonl'), rows(questions_path)
    assert len(old_answers) == len(new_answers) == len(questions) == 5000
    canonical = [(str(row['question_id']), row['text']) for row in questions]
    assert len(set(canonical)) == 5000
    for answers in [old_answers, new_answers]:
        assert [(str(row['question_id']), row['prompt']) for row in answers] == canonical
        assert all(isinstance(row['text'], str) and not row['text'].startswith('FAILED') for row in answers)
        assert all(row['model_id'] == 'llava-v1.6-vicuna-7b' for row in answers)
    for name, artifact in new_score['artifacts'].items():
        assert sha(artifact['path']) == artifact['sha256']
    old_runtime, new_runtime = rows(old_runtime_path), rows(NEW / 'runtime.jsonl')
    assert len(old_runtime) == len(new_runtime) == 5000
    for position, (question, old_trace, new_trace) in enumerate(zip(questions, old_runtime, new_runtime)):
        expected_prompt = hashlib.sha256(question['text'].encode()).hexdigest()
        assert old_trace['question_position'] == new_trace['position'] == position
        assert str(old_trace['question_id']) == str(new_trace['question_id']) == str(question['question_id'])
        assert old_trace['image'] == new_trace['image'] == question['image']
        assert old_trace['prompt_sha256'] == new_trace['prompt_sha256'] == expected_prompt
        assert old_trace['visual_token_budget_parameter'] == 128
        assert old_trace['input_sha256'] == plan['questions']['sha256']
        assert old_trace['source_wrapper_sha256'] == old_protocol['wrapper_sha256']
        assert old_trace['image_tensor_shape'] == new_trace['image_tensor_shape'] == [1, 5, 3, 336, 336]
        assert all(type(trace['actual_visual_tokens_retained']) is int and 0 < trace['actual_visual_tokens_retained'] <= 640 for trace in [old_trace, new_trace])
    old_counts = Counter(row['actual_visual_tokens_retained'] for row in old_runtime)
    new_counts = Counter(row['actual_visual_tokens_retained'] for row in new_runtime)
    assert {str(key): value for key, value in old_counts.items()} == old_score['actual_count_distribution']
    assert {str(key): value for key, value in new_counts.items()} == new_score['retained_count_histogram']
    registry_paths = [COMPARISON, old_path, old_runtime_path, old_score_path, old_control_path, old_protocol_path,
        old_images_path, new_images_path, NEW / 'answers.jsonl', NEW / 'runtime.jsonl', NEW / 'loaded_runtime.json',
        new_score_path, NEW / 'per_question_scores.jsonl', NEW / 'release.json', plan_path, questions_path, annotation_path,
        AUTHOR / 'llava/eval/eval_textvqa.py', AUTHOR / 'llava/eval/m4c_evaluator.py', Path(__file__).resolve()]
    registry = [describe(path) for path in registry_paths]
    sys.path.insert(0, str(AUTHOR))
    from llava.eval.eval_textvqa import prompt_processor
    from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
    assert inspect.getfile(TextVQAAccuracyEvaluator) == str(AUTHOR / 'llava/eval/m4c_evaluator.py')
    evaluator = TextVQAAccuracyEvaluator()
    annotations = read(annotation_path)['data']
    by_key = {(str(row['image_id']), row['question'].lower()): row for row in annotations}
    gt_keys = [(str(question['question_id']), prompt_processor(question['text'])) for question in questions]
    assert len(by_key) == len(annotations) == len(set(gt_keys)) == 5000
    assert set(gt_keys) == set(by_key)
    all_pairs = {name: [{'pred_answer': answer['text'], 'gt_answers': by_key[key]['answers']}
        for answer, key in zip(answers, gt_keys)] for name, answers in [('beta1', new_answers), ('beta2', old_answers)]}
    full_values = {name: 100 * evaluator.eval_pred_list(pairs) for name, pairs in all_pairs.items()}
    assert abs(full_values['beta1'] - new_score['value']) < 1e-10
    assert abs(full_values['beta2'] - old_score['accuracy_percent']) < 1e-10
    assert abs(full_values['beta2'] - reference['local_value']) < 1e-10
    cli_evidence = {}
    for name, prediction in [('beta1', NEW / 'answers.jsonl'), ('beta2', old_path)]:
        command = [PYTHON, '-m', 'llava.eval.eval_textvqa', '--annotation-file', str(annotation_path), '--result-file', str(prediction)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(AUTHOR), PYTHONDONTWRITEBYTECODE='1')
        scored = subprocess.run(command, cwd=AUTHOR, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = OUT / f'{name}_fresh_author_stdout.txt'
        log.write_text(scored.stdout)
        assert scored.returncode == 0 and 'Samples: 5000' in scored.stdout
        cli_evidence[name] = {'command': command, 'exit_code': 0, **describe(log)}
    counters = Counter()
    transitions, count_deltas = Counter(), Counter()
    old_units_total = new_units_total = positive_units = negative_units = 0
    new_official_rows = rows(NEW / 'per_question_scores.jsonl')
    paired_path = OUT / 'per_question_pairs.jsonl'
    with paired_path.open('w') as stream:
        for position, (question, old, new, gt_key, old_trace, new_trace, new_official) in enumerate(zip(
                questions, old_answers, new_answers, gt_keys, old_runtime, new_runtime, new_official_rows)):
            gt = by_key[gt_key]['answers']
            assert len(gt) == 10
            answer_scores = evaluator._compute_answer_scores(gt)
            old_normalized, new_normalized = evaluator.answer_processor(old['text']), evaluator.answer_processor(new['text'])
            old_score_value, new_score_value = answer_scores.get(old_normalized, 0.), answer_scores.get(new_normalized, 0.)
            old_units, new_units = round(old_score_value * 10), round(new_score_value * 10)
            assert abs(old_score_value - old_units / 10) < 1e-12 and abs(new_score_value - new_units / 10) < 1e-12
            assert new_official['position'] == position and abs(new_official['score'] - new_score_value) < 1e-12
            difference = new_units - old_units
            change = 'improved' if difference > 0 else 'declined' if difference < 0 else 'unchanged'
            raw_changed, normalized_changed = old['text'] != new['text'], old_normalized != new_normalized
            counters['score_' + change] += 1
            counters['raw_answer_changed' if raw_changed else 'raw_answer_unchanged'] += 1
            counters['normalized_answer_changed' if normalized_changed else 'normalized_answer_unchanged'] += 1
            counters[f'raw_{"changed" if raw_changed else "same"}__score_{change}'] += 1
            old_units_total += old_units
            new_units_total += new_units
            positive_units += max(0, difference)
            negative_units += min(0, difference)
            transitions[(old_units, new_units)] += 1
            old_count, new_count = old_trace['actual_visual_tokens_retained'], new_trace['actual_visual_tokens_retained']
            count_deltas[new_count - old_count] += 1
            counters['retained_count_equal' if old_count == new_count else 'retained_count_changed'] += 1
            stream.write(json.dumps({'position': position, 'question_id': question['question_id'], 'image': question['image'],
                'prompt_sha256': hashlib.sha256(question['text'].encode()).hexdigest(), 'GT_key': list(gt_key),
                'old_beta2_answer': old['text'], 'new_beta1_answer': new['text'],
                'old_normalized_answer': old_normalized, 'new_normalized_answer': new_normalized,
                'old_official_score': old_score_value, 'new_official_score': new_score_value,
                'score_delta_tenths': difference, 'score_change': change,
                'old_actual_retained_tokens': old_count, 'new_actual_retained_tokens': new_count,
                'retained_count_delta': new_count - old_count}) + '\n')
    old_exact = Decimal(old_units_total) / Decimal(500)
    new_exact = Decimal(new_units_total) / Decimal(500)
    net = new_exact - old_exact
    assert abs(float(old_exact) - full_values['beta2']) < 1e-10
    assert abs(float(new_exact) - full_values['beta1']) < 1e-10
    assert positive_units + negative_units == new_units_total - old_units_total
    assert sum(value for key, value in counters.items() if key.startswith('score_')) == 5000
    assert all(sha(row['path']) == row['sha256'] for row in registry)
    import torch
    assert not torch.cuda.is_initialized()
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'success': True, 'n_paired': 5000,
        'direction': 'new beta1 minus preserved historical beta2', 'same_ordered_composite_IDs_full_prompts_GT': True,
        'same3166_recorded_image_SHA256': True, 'old_reference_from_comparison': reference,
        'fresh_author_full_score': full_values, 'exact_consensus_accuracy_percent': {'beta1': str(new_exact), 'beta2': str(old_exact), 'delta': str(net)},
        'official_CLI': cli_evidence, 'paired_counts': dict(counters),
        'total_score_gain_tenths': positive_units, 'total_score_loss_tenths_signed': negative_units,
        'net_score_change_tenths': positive_units + negative_units,
        'score_transition_counts': [{'old_score_tenths': key[0], 'new_score_tenths': key[1], 'n': value} for key, value in sorted(transitions.items())],
        'retained_token_trace_comparison': {'available_and_identity_qualified_both_runs': True,
            'old_runtime_sha256_verified_against_historical_official_score': old_score['runtime_sha256'],
            'new_runtime_sha256_verified_against_current_official_score': new_score['artifacts']['runtime.jsonl']['sha256'],
            'old_count_histogram': dict(sorted(old_counts.items())), 'new_count_histogram': dict(sorted(new_counts.items())),
            'paired_new_minus_old_count_delta_histogram': dict(sorted(count_deltas.items())),
            'same_count_rows': counters['retained_count_equal'], 'different_count_rows': counters['retained_count_changed'],
            'old_mean_count': sum(k*v for k,v in old_counts.items()) / 5000,
            'new_mean_count': sum(k*v for k,v in new_counts.items()) / 5000,
            'selection_token_indices_or_per_crop_quotas_comparable': False,
            'limit': 'Both traces record complete real returned total counts and five-crop shape. Equal counts do not establish equal selected visual content; old trace lacks selected-token indices/output-token IDs.'},
        'protocol_comparison': {'same_core_source_SHA': old_protocol['source_sha256'],
            'old_recorded_torch': old_protocol['torch_version'], 'old_recorded_transformers': old_protocol['transformers_version'],
            'same_stream_wait_statements': plan['waits'], 'same_other_recorded_generation_parameters': True,
            'changed_beta': {'old': 2., 'new': 1.}, 'old_entry': old_control['native_runtime_argv'][0],
            'new_entry': str(AUTHOR / 'llava/eval/model_vqa_loader.py')},
        'limits': ['This quantifies these two complete preserved runs only. Historical and current entry wrappers, observations, process execution and low-level scheduling cannot be replayed or equated by this CPU pairing.',
            'A paired numerical difference is not proof that beta alone causes every answer or score change.',
            'The unpublished paper main-table argv remains unknown. The new lower result does not reproduce the paper59.2 or establish completion of the overall reproduction goal.',
            'Published paper Avg is not a local average; no result, fixed plan, worker, prediction or active GPU process was modified.'],
        'paired_records': describe(paired_path), 'input_registry_before_and_after_unchanged': registry,
        'cuda_initialized': False, 'GPU_or_new_parameter_run_started': False}
    (OUT / 'comparison_report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'success': True, 'n': 5000, 'exact_accuracy_percent': result['exact_consensus_accuracy_percent'],
        'paired_counts': dict(counters), 'retained_counts': result['retained_token_trace_comparison']}))


if __name__ == '__main__':
    main()
