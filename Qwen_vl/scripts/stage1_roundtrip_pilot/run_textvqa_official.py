"""Generate separate artifacts with the official TextVQA loader protocol.

Default is a CPU plan. Explicit --generate runs selected arms sequentially,
then invokes the official evaluator. Existing files are never overwritten.
This is a protocol repair (fixed alpha/beta/lambda/budgets), not a parameter sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
LLAVA = REPO / 'LLaVA'
EVAL = LLAVA / 'playground/data/eval'
WRAPPER = Path(__file__).with_name('llava_eval_arm_model_vqa.py')
QUESTION = EVAL / 'textvqa/llava_textvqa_val_v051_ocr.jsonl'
ARMS = {'FULL': (0, False), 'EGATHER': (128, False),
        'LRMAIN025': (128, True), 'LRMAIN0125': (64, True),
        'LRMAIN00625': (32, True)}
MODELS = {'v15': '/media/disk2/YZX/doct/FastV/llava-v1.5-7b',
          'next': '/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['v15', 'next', 'both'], default='both')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        parser.error('Choose unique arm names from ' + ','.join(ARMS))
    model_names = list(MODELS) if args.model == 'both' else [args.model]
    jobs = []
    for model_name in model_names:
        task_dir = 'textvqa_official_20261008' if model_name == 'v15' else 'next_textvqa_official_20261008'
        for arm in arms:
            budget, az = ARMS[arm]
            answer = EVAL / 'anchorzip_p3' / task_dir / f'{arm}.jsonl'
            cmd = [sys.executable, str(WRAPPER), '--model-path', MODELS[model_name],
                   '--question-file', str(QUESTION), '--image-folder', str(EVAL/'textvqa/train_images'),
                   '--answers-file', str(answer), '--temperature', '0', '--conv-mode', 'vicuna_v1',
                   '--visual_token_num', str(budget), '--beta', '2.0', '--alpha', '0.5',
                   '--official-textvqa']
            if az:
                cmd.append('--anchorzip')
            scorer = [sys.executable, '-m', 'llava.eval.eval_textvqa', '--annotation-file',
                      str(EVAL/'textvqa/TextVQA_0.5.1_val.json'), '--result-file', str(answer)]
            jobs.append(dict(model=model_name, arm=arm, answer=str(answer),
                             generate_command=cmd, score_command=scorer,
                             generation_log=str(answer.with_suffix('.log')),
                             score_log=str(answer.with_suffix('.score.txt'))))
    plan = dict(protocol='official_textvqa_loader', n_jobs=len(jobs),
                source_sha256={str(WRAPPER):sha(WRAPPER), str(QUESTION):sha(QUESTION)},
                frozen=dict(alpha=0.5, beta=2.0, lambda_completion=0.25,
                            conv_mode='vicuna_v1', temperature=0, max_new_tokens=128), jobs=jobs)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.generate:
        return
    for variable in ('USE_LLAVA_ARCH_CDPRUNER', 'USE_LLAVA_ARCH_DIVPRUNE',
                     'USE_LLAVA_ARCH_HIPRUNE', 'USE_LLAVA_ARCH_ABLATION'):
        if os.environ.get(variable) == '1':
            raise RuntimeError(f'Official EADP comparison has unexpected architecture: {variable}')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(LLAVA)
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    for job in jobs:
        answer = Path(job['answer'])
        sidecar = Path(str(answer)+'.protocol.json')
        if answer.exists() or sidecar.exists():
            raise FileExistsError(f'Preserving existing artifact: {answer}')
        answer.parent.mkdir(parents=True, exist_ok=True)
        with open(job['generation_log'], 'x') as log:
            subprocess.run(job['generate_command'], env=env, cwd=LLAVA,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        n = sum(1 for _ in answer.open())
        if n != 5000:
            raise RuntimeError(f'Incomplete TextVQA artifact {answer}: {n}/5000')
        with open(job['score_log'], 'x') as log:
            subprocess.run(job['score_command'], env=env, cwd=LLAVA,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Complete {job["model"]}/{job["arm"]}: {job["score_log"]}', flush=True)


if __name__ == '__main__':
    main()
