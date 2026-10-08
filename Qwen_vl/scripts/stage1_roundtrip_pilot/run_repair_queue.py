"""Persist a single GPU lane and score each repaired protocol after generation.

Plans contain explicit generation/score commands. Existing predictions are
never overwritten. The plan can be edited atomically before a pending job is
started; completed jobs remain immutable in the state file.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def verify_sources(plan):
    verified = {}
    for filename, expected in plan['source_sha256'].items():
        actual = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source/input SHA256 drift: {filename}')
        verified[filename] = actual
    return verified


def check_textvqa_score(log_path, expected_count):
    content = Path(log_path).read_text()
    samples = re.findall(r'^Samples:\s*(\d+)\s*$', content, flags=re.MULTILINE)
    accuracies = re.findall(r'^Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%?\s*$', content, flags=re.MULTILINE)
    if ('Traceback' in content or len(samples) != 1 or int(samples[0]) != expected_count
            or len(accuracies) != 1 or not 0 <= float(accuracies[0]) <= 100):
        raise RuntimeError(f'Invalid/incomplete official TextVQA score log: {log_path}')
    return dict(n=int(samples[0]), accuracy=float(accuracies[0]), official_score_log=str(log_path))


def check_predictions(job):
    output = Path(job['output'])
    predictions = [json.loads(line) for line in output.open()]
    question_path = Path(job['question_file'])
    if job['task'] == 'sqa':
        import importlib.util
        helper_path = Path(__file__).with_name('run_sqa_shipped_control.py')
        spec = importlib.util.spec_from_file_location('shipped_sqa_checks', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.check_science_identity()
        helper.check_questions()
        questions = json.loads(question_path.read_text())
        expected = {str(q['id']) for q in questions}
    else:
        questions = [json.loads(line) for line in question_path.open()]
        # Official TextVQA question_id is an image ID shared by multiple
        # questions. The complete prompt disambiguates the 5000 records.
        expected = {(str(q['question_id']), q['text']) for q in questions}
    actual = [str(row['question_id']) for row in predictions]
    actual_keys = ([(str(row['question_id']), row['prompt']) for row in predictions]
                   if job['task'] == 'textvqa' else actual)
    if len(expected) != len(questions) or len(actual_keys) != len(expected) or len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected:
        raise RuntimeError(f'Prediction ID/count integrity failure: {output}')
    if any(str(row.get('text', '')).strip() in ('', 'FAILED') for row in predictions):
        raise RuntimeError(f'Empty/FAILED predictions: {output}')
    budget = int(job['generate_command'][job['generate_command'].index('--visual_token_num') + 1])
    if job['task'] == 'sqa':
        prompts = {}
        for question in questions:
            prompt = question['conversations'][0]['value'].replace('<image>', '').strip()
            if 'image' in question:
                prompt = '<image>\n' + prompt
            if '--single-pred-prompt' in job['generate_command']:
                prompt += "\nAnswer with the option's letter from the given choices directly."
            prompts[str(question['id'])] = prompt
        for row in predictions:
            metadata = row.get('metadata', {})
            if (row['prompt'] != prompts[str(row['question_id'])]
                    or metadata.get('visual_token_budget_parameter') != budget
                    or metadata.get('has_image') is not True
                    or not isinstance(metadata.get('actual_visual_tokens_retained'), int)
                    or metadata['actual_visual_tokens_retained'] <= 0):
                raise RuntimeError('SQA prompt/runtime budget metadata changed or absent')
    if job['task'] == 'textvqa':
        if actual_keys != [(str(q['question_id']), q['text']) for q in questions]:
            raise RuntimeError('TextVQA LLM prompt/input order changed')
        if job.get('requires_runtime_trace', True) is False:
            return dict(n_rows=len(predictions), unique_records=len(set(actual_keys)),
                        unique_question_ids=len(set(actual)), runtime_trace_available=False,
                        prediction_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        question_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest())
        trace_file = Path(str(output) + '.runtime.jsonl')
        trace = [json.loads(line) for line in trace_file.open()]
        if [str(row['question_id']) for row in trace] != actual:
            raise RuntimeError('Runtime trace ID/order does not match predictions')
        for position, (row, question) in enumerate(zip(trace, questions)):
            if (row.get('question_position') != position
                    or row.get('image') != question['image']
                    or row.get('prompt_sha256') != hashlib.sha256(question['text'].encode()).hexdigest()):
                raise RuntimeError('Runtime trace composite question identity mismatch')
        if not all(isinstance(row['actual_visual_tokens_retained'], int) and row['actual_visual_tokens_retained'] > 0 for row in trace):
            raise RuntimeError('Invalid runtime count')
        for row in trace:
            shape = row.get('image_tensor_shape', [])
            if row.get('visual_token_budget_parameter') != budget or len(shape) not in (4, 5):
                raise RuntimeError('TextVQA runtime budget/image shape mismatch')
            crops = shape[1] if len(shape) == 5 else 1
            if row['actual_visual_tokens_retained'] > crops * (576 + 32):
                raise RuntimeError('Retained tokens exceed image geometry')
    return dict(n_rows=len(predictions), unique_records=len(set(actual_keys)),
                unique_question_ids=len(set(actual)),
                prediction_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                question_sha256=hashlib.sha256(question_path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    lock = args.state.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.state.exists():
        raise FileExistsError(f'Preserving existing queue state {args.state}')
    state = dict(supervisor_pid=os.getpid(), created_utc=now(), completed=[], status='waiting')
    save(args.state, state)
    try:
        plan = json.loads(args.plan.read_text())
        wait_pid = plan.get('wait_pid')
        while wait_pid and Path(f'/proc/{wait_pid}').exists():
            state.update(status='waiting', wait_pid=wait_pid, updated_utc=now())
            save(args.state, state)
            time.sleep(5)
        marker = plan.get('wait_marker')
        while marker and not Path(marker).is_file():
            state.update(status='waiting', wait_marker=marker, updated_utc=now())
            save(args.state, state)
            time.sleep(5)
        verified_sources = verify_sources(plan)
        state.update(verified_source_sha256=verified_sources, sources_verified_utc=now())
        save(args.state, state)
        if plan.get('after_wait_command'):
            state.update(status='scoring_predecessor', updated_utc=now())
            save(args.state, state)
            with Path(plan['after_wait_log']).open('x') as log:
                subprocess.run(plan['after_wait_command'], cwd=plan['cwd'],
                               env=dict(os.environ, **plan.get('environment', {})),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            state['predecessor_scored_utc'] = now()
        for required_path in plan.get('wait_required_outputs', []):
            if not Path(required_path).is_file():
                raise RuntimeError(f'Initial queue did not complete: {required_path}')
        state['predecessors_validated'] = []
        for predecessor in plan.get('predecessor_jobs', []):
            integrity = check_predictions(predecessor)
            score = check_textvqa_score(predecessor['score_log'], integrity['n_rows'])
            state['predecessors_validated'].append(dict(key=predecessor['key'], integrity=integrity, score=score))
        save(args.state, state)
        controlled_marker = plan.get('wait_after_predecessor_marker')
        while controlled_marker and not Path(controlled_marker).is_file():
            state.update(status='waiting_after_predecessor', child_pid=None,
                         wait_after_predecessor_marker=controlled_marker, updated_utc=now())
            save(args.state, state)
            time.sleep(5)
        if controlled_marker:
            marker_result = json.loads(Path(controlled_marker).read_text())
            if marker_result.get('success') is not True:
                raise RuntimeError(f'Controlled predecessor was not validated successfully: {controlled_marker}')
            state['controlled_predecessor_validated_utc'] = now()
            save(args.state, state)
        while True:
            plan = json.loads(args.plan.read_text())
            done = {entry['key'] for entry in state['completed']}
            job = next((entry for entry in plan['jobs'] if entry['key'] not in done), None)
            if job is None:
                state.update(status='complete', updated_utc=now())
                save(args.state, state)
                return
            verified_sources = verify_sources(plan)
            state.update(verified_source_sha256=verified_sources, sources_verified_utc=now())
            save(args.state, state)
            output = Path(job['output'])
            output.parent.mkdir(parents=True, exist_ok=True)
            protected = [output, Path(str(output) + '.protocol.json'),
                         Path(str(output) + '.runtime.jsonl'), output.with_suffix('.output.json'),
                         output.with_suffix('.result.json'),
                         Path(job['generation_log']), Path(job['score_log'])]
            if any(path.exists() for path in protected):
                raise FileExistsError(f'Existing pending job artifact: {output}')
            if job['task'] == 'sqa':
                protocol = dict(job['protocol_metadata'])
                protocol['job'] = job
                protocol['queue_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                protocol['source_sha256'] = verified_sources
                protocol['sources_verified_utc'] = state['sources_verified_utc']
                protocol['fixed_subset_sha256'] = hashlib.sha256(Path(job['question_file']).read_bytes()).hexdigest()
                Path(str(output) + '.protocol.json').write_text(json.dumps(protocol, indent=2))
            state.update(status='generating', current_job=job, started_utc=now(), updated_utc=now())
            save(args.state, state)
            env = dict(os.environ, **plan.get('environment', {}))
            with Path(job['generation_log']).open('x') as log:
                process = subprocess.Popen(job['generate_command'], env=env, cwd=plan['cwd'],
                                           stdout=log, stderr=subprocess.STDOUT)
                state.update(child_pid=process.pid, updated_utc=now())
                save(args.state, state)
                if process.wait() != 0:
                    raise RuntimeError(f'Generation failed: {job["key"]}')
            integrity = check_predictions(job)
            state.update(status='scoring', integrity=integrity, updated_utc=now())
            save(args.state, state)
            with Path(job['score_log']).open('x') as log:
                subprocess.run(job['score_command'], env=env, cwd=plan['cwd'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            if job['task'] == 'sqa':
                result = json.loads(output.with_suffix('.result.json').read_text())
                if result['count'] != len(json.loads(Path(job['question_file']).read_text())):
                    raise RuntimeError('SQA scorer denominator changed')
                score = dict(n=result['count'], correct=result['correct'], accuracy=result['acc'])
            else:
                score = check_textvqa_score(job['score_log'], integrity['n_rows'])
            state['completed'].append(dict(key=job['key'], output=str(output),
                                           integrity=integrity, score=score, completed_utc=now()))
            state.update(status='between_jobs', child_pid=None, updated_utc=now())
            save(args.state, state)
    except Exception as exc:
        state.update(status='failed', error=str(exc), updated_utc=now())
        save(args.state, state)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
