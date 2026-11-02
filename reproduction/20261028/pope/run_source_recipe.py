#!/usr/bin/env python3
"""Run the fresh author POPE shell-default recipe without source patches."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
AUTHOR = Path('/media/disk2/YZX/research/EADP_v2/_reproduction/author_eadp')
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
MODEL = Path('/media/disk2/YZX/doct/FastV/llava-v1.5-7b')
IMAGES = Path('/media/disk2/YZX/research/EADP_amp/LLaVA/playground/data/eval/pope/val2014')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def prepare():
    run = OUT / 'run01_source_as_is_k32'
    run.mkdir(exist_ok=True)
    paths = {'predictions': run / 'predictions.jsonl', 'generation_log': run / 'generation.log',
             'score_log': run / 'official_score.log', 'plan': run / 'run_plan.json', 'state': run / 'state.json'}
    command = [PYTHON, '-m', 'llava.eval.model_vqa_loader', '--model-path', str(MODEL),
        '--question-file', str(OUT / 'inputs/llava_pope_test.jsonl'), '--image-folder', str(IMAGES),
        '--answers-file', str(paths['predictions']), '--visual_token_num', '32', '--beta', '1.0',
        '--alpha', '1.0', '--temperature', '0', '--conv-mode', 'vicuna_v1']
    score = [PYTHON, str(AUTHOR / 'LLaVA/llava/eval/eval_pope.py'),
        '--annotation-dir', str(OUT / 'pinned_pope_labels'), '--question-file', str(OUT / 'inputs/llava_pope_test.jsonl'),
        '--result-file', str(paths['predictions'])]
    sources = ['LLaVA/scripts/v1_5/eval/pope.sh', 'LLaVA/llava/eval/model_vqa_loader.py',
        'LLaVA/llava/eval/eval_pope.py', 'LLaVA/llava/model/llava_arch.py',
        'LLaVA/llava/model/multimodal_encoder/clip_encoder.py', 'LLaVA/llava/model/builder.py',
        'LLaVA/llava/model/language_model/llava_llama.py', 'LLaVA/llava/conversation.py']
    plan = {'run_id': run.name, 'recipe_origin': 'author LLaVA/scripts/v1_5/eval/pope.sh default positional arguments',
        'official_commit': 'e1a08801461c181871b9ff1cb0803b8e9966f083', 'source_root': str(AUTHOR),
        'source_files': {name: sha(AUTHOR / name) for name in sources}, 'source_runtime_patches': [],
        'generation_command': command, 'generation_cwd': str(AUTHOR / 'LLaVA'), 'score_command': score,
        'environment_overrides': {'PYTHONPATH': str(AUTHOR / 'LLaVA'), 'CUDA_VISIBLE_DEVICES': '0',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false'},
        'cleared_environment_prefixes': ['USE_LLAVA_ARCH_', 'ANCHORZIP_', 'ENTROPY_', 'EADP_'],
        'parameters': {'visual_token_num': 32, 'alpha': 1.0, 'beta': 1.0, 'q': .2,
            'entropy_T': 100., 'local_aggregation': 'negative_entropy', 'aggregation_temperature': .01,
            'smoothing_kernel': 3, 'smoothing_sigma': 1., 'temperature': 0, 'num_beams': 1,
            'max_new_tokens': 128, 'conv_mode': 'vicuna_v1', 'dtype': 'float16',
            'source_caller_stream_wait': False},
        'input_audit': {'path': str(OUT / 'input_and_score_audit.json'), 'sha256': sha(OUT / 'input_and_score_audit.json')},
        'model_small_files': {str(p): sha(p) for p in sorted(MODEL.iterdir()) if p.is_file() and p.suffix in {'.json', '.model'} and p.stat().st_size < 10_000_000},
        'paths': {k: str(v) for k, v in paths.items()},
        'claim_boundary': 'This tests the current released POPE shell-default recipe, not an asserted reconstruction of the undocumented paper main-table argv. Source as-is has no caller-stream dependency fix; any later corrected run must receive a separate run id.'}
    save(paths['plan'], plan)
    return plan, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan, paths = prepare()
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    if paths['predictions'].exists() and paths['predictions'].stat().st_size:
        raise RuntimeError('Refusing to overwrite preserved predictions')
    env = os.environ.copy()
    for key in list(env):
        if any(key.startswith(prefix) for prefix in plan['cleared_environment_prefixes']):
            del env[key]
    env.update(plan['environment_overrides'])
    started = time.time()
    state = {'run_id': plan['run_id'], 'controller_pid': os.getpid(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'status': 'starting', 'source_runtime_patches': []}
    save(paths['state'], state)
    with paths['generation_log'].open('w') as log:
        process = subprocess.Popen(plan['generation_command'], cwd=plan['generation_cwd'], env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        state.update(status='generating', worker_pid=process.pid)
        save(paths['state'], state)
        code = process.wait()
    state.update(generation_exit_code=code, generation_elapsed_seconds=time.time() - started)
    if code:
        state['status'] = 'generation_failed'
        save(paths['state'], state)
        raise SystemExit(code)
    rows = [json.loads(x) for x in paths['predictions'].read_text().splitlines()]
    questions = [json.loads(x) for x in (OUT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    if len(rows) != 8910 or [x['question_id'] for x in rows] != [x['question_id'] for x in questions]:
        state['status'] = 'completion_gate_failed'
        save(paths['state'], state)
        raise RuntimeError('Incomplete or reordered question ids')
    state.update(status='scoring', n_predictions=len(rows), predictions_sha256=sha(paths['predictions']))
    save(paths['state'], state)
    with paths['score_log'].open('w') as log:
        scored = subprocess.run(plan['score_command'], cwd=plan['generation_cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(score_exit_code=scored.returncode, status='complete' if scored.returncode == 0 else 'score_failed',
                 completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), total_elapsed_seconds=time.time() - started)
    save(paths['state'], state)
    print(json.dumps(state, indent=2))
    raise SystemExit(scored.returncode)


if __name__ == '__main__':
    main()
