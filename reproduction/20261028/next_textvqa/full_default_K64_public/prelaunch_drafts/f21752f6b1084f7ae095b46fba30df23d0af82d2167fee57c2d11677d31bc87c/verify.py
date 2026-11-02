"""Complete-only CPU pointwise audit; never scores an incomplete prefix."""
from collections import Counter
from decimal import Decimal
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import worker

OUT, NEXT, RUN = worker.OUT, worker.NEXT, worker.RUN
read = lambda p: json.loads(Path(p).read_text())
rows = lambda p: [json.loads(line) for line in Path(p).read_text().splitlines()]
sha = worker.sha


def consensus(n):
    assert type(n) is int and 0 <= n <= 10
    return (n * min(max(n - 1, 0) / 3, 1) + (10 - n) * min(n / 3, 1)) / 10


def check_observation(record, reference, runtime, position):
    assert record['position'] == reference['position'] == position
    assert record['input_shape'] == [1, reference['n_tokens']] and record['input_dtype'] == 'torch.int64'
    assert record['input_ids_sha256'] == reference['token_ids_sha256']
    assert record['crop_count'] == 5 and record['initial_quotas'] == [64] * 5
    quotas = record['allocated_quotas']
    assert len(quotas) == 5 and all(type(q) is int and 1 <= q <= 320 for q in quotas)
    assert quotas == record['greedy_quotas'] == record['effective_mask_counts']
    assert record['effective_mask_shape'] == [5, 576]
    assert len(record['effective_mask_indices']) == len(record['greedy_effective_prefixes']) == 5
    for q, mask, prefix in zip(quotas, record['effective_mask_indices'], record['greedy_effective_prefixes']):
        assert len(mask) == len(prefix) == len(set(mask)) == len(set(prefix)) == q
        assert mask == sorted(prefix) and all(type(i) is int and 0 <= i < 576 for i in prefix)
    assert 0 < sum(quotas) <= 320
    assert sum(quotas) == record['returned_retained_tokens'] == runtime['actual_visual_tokens_retained']
    assert record['importance_shape'] == [5, 576] and record['importance_finite']
    assert record['projected_feature_shape'] == [5, 576, 4096] and record['projected_feature_dtype'] == 'torch.float16'
    assert record['post_generation_tensor_reads_only'] is True


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (OUT / 'failure.json').exists() and not (RUN / 'failure.json').exists()
    native = read(RUN / 'state.json')
    assert native['status'] == 'complete' and native['n'] == 5000
    controller = read(OUT / 'controller_state.json')
    assert controller['native_exit_code'] == 0 and controller['status'] == 'native_exited'
    from run import identity, gpu_pids, check
    assert identity(controller['native_worker']['pid']) != controller['native_worker']
    assert controller['native_worker']['pid'] not in gpu_pids()
    plan = read(worker.PLAN)
    check(plan)
    full, proof = worker.adapted_module()
    official = full.verify(write=False)
    stored = read(RUN / 'official.score.json')
    assert stored['n'] == stored['token_decoding_checks'] == 5000
    assert native['score_sha256'] == sha(RUN / 'official.score.json')
    assert stored['value'] == official['value']
    for entry in stored['artifacts'].values():
        assert sha(entry['path']) == entry['sha256']
    assert sha(RUN / 'per_question_scores.jsonl') == stored['per_question_scores_sha256']
    questions = rows(plan['questions']['path'])
    predictions, runtime = rows(RUN / 'answers.jsonl'), rows(RUN / 'runtime.jsonl')
    observations = rows(RUN / 'input_budget_observations.jsonl')
    refs = read(NEXT / 'llm_tokens_current.json')['rows']
    assert refs == read(NEXT / 'llm_tokens_author_pins.json')['rows']
    pointwise = rows(RUN / 'per_question_scores.jsonl')
    assert len(questions) == len(predictions) == len(runtime) == len(observations) == len(refs) == len(pointwise) == 5000
    prompt_processor, evaluator_type = full.author_imports()
    evaluator = evaluator_type()
    annotations = read(plan['annotation']['path'])['data']
    gt = {(str(a['image_id']), a['question'].lower()): a['answers'] for a in annotations}
    assert len(gt) == len(annotations) == 5000
    units, maximum_delta, counts = 0, 0., Counter()
    independent_rows = []
    for i, (q, pred, trace, observation, reference, saved) in enumerate(zip(questions, predictions, runtime, observations, refs, pointwise)):
        assert str(q['question_id']) == str(reference['question_id'])
        check_observation(observation, reference, trace, i)
        key = (str(q['question_id']), prompt_processor(q['text']))
        answers = gt[key]
        assert len(answers) == 10
        normalized = evaluator.answer_processor(pred['text'])
        matched = sum(evaluator.answer_processor(answer) == normalized for answer in answers)
        independent = consensus(matched)
        native_score = evaluator._compute_answer_scores(answers).get(normalized, 0.)
        assert saved['position'] == i and str(saved['question_id']) == str(q['question_id']) and saved['score'] == native_score
        delta = abs(native_score - independent)
        maximum_delta = max(maximum_delta, delta)
        assert delta < 1e-12
        units += round(independent * 10)
        counts[observation['returned_retained_tokens']] += 1
        independent_rows.append(dict(position=i, question_id=q['question_id'], prompt_sha256=trace['prompt_sha256'],
            official_score=native_score, independent_score=independent, native_quotas=observation['allocated_quotas']))
    exact = Decimal(units) / Decimal(500)
    assert abs(float(exact) - official['value']) < 1e-10
    output = RUN / 'independent_per_question_scores.jsonl'
    with output.open('x') as stream:
        stream.write(''.join(json.dumps(r) + '\n' for r in independent_rows))
    command = [sys.executable, '-m', 'llava.eval.eval_textvqa', '--annotation-file', plan['annotation']['path'], '--result-file', str(RUN / 'answers.jsonl')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(full.AUTHOR / 'LLaVA'), PYTHONDONTWRITEBYTECODE='1')
    cli = subprocess.run(command, env=env, cwd=full.AUTHOR / 'LLaVA', text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = RUN / 'fresh_author_cli.txt'
    log.write_text(cli.stdout)
    assert cli.returncode == 0 and 'Samples: 5000' in cli.stdout
    report = dict(success=True, n_rows=5000, token_decoding_checks=5000, input_budget_observation_checks=5000,
        per_question_score_checks=5000, official_accuracy_percent=official['value'], independent_accuracy_percent=float(exact),
        independent_exact_percent=str(exact), max_per_question_score_delta=maximum_delta,
        budget_cli=64, budget_nominal=320, actual_retained_histogram=dict(sorted(counts.items())),
        budget_semantics=plan['budget_semantics'], worker_adapter=proof,
        full_denominator_retained=True, empty_answers=sum(not p['text'].strip() for p in predictions),
        failed_placeholder_count=sum(p['text'].startswith(('FAILED', 'Failed to obtain answer')) for p in predictions),
        plan_sha256=sha(worker.PLAN), official_score_sha256=sha(RUN / 'official.score.json'),
        fresh_author_cli=dict(command=command, exit_code=cli.returncode, path=str(log), sha256=sha(log)),
        artifacts={n:dict(path=str(RUN / n), sha256=sha(RUN / n)) for n in ['answers.jsonl', 'runtime.jsonl', 'loaded_runtime.json',
            'input_budget_observations.jsonl', 'per_question_scores.jsonl', output.name]},
        CPU_only=True, model_loaded=False, paper_argv_confirmed=False, metric_or_parameter_selection=False,
        limitations='Public-default coverage, not proof of paper argv or a repair. Shared author normalization with separate closed-form consensus; all5000 remain in denominator. Native floor/clamp crop allocation and actual mask positions are preserved; not forced to320.')
    import torch
    assert not torch.cuda.is_initialized()
    with (RUN / 'independent_verification.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['success', 'n_rows', 'official_accuracy_percent', 'independent_accuracy_percent', 'actual_retained_histogram']}))


if __name__ == '__main__':
    main()
