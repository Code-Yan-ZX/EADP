"""SQA control using the unchanged question text shipped in eval.zip.

This is the complete 2017-image test subset. It does not claim that the
unavailable author CQM-I file is identical to shipped CQM-A. Default is a
CPU plan; --generate runs new artifacts with the unchanged official scorer.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
LLAVA = REPO / 'LLaVA'
SCRIPT = Path(__file__).with_name('llava_eval_arm_science.py')
OUT = REPO / 'Qwen_vl/outputs/audit_followup_20261008'
QUESTION = OUT / 'sqa_CQMA_image2017.json'
SOURCE = LLAVA / 'playground/data/eval/scienceqa/llava_test_CQM-A.json'
ORIGINAL_SCIENCE = Path('/media/disk2/YZX/research/audit_base_gap_20261003/eadp_official/LLaVA/llava/eval/model_vqa_science.py')
BASE = OUT / 'sqa_image_only_base'
ARMS = {'FULL': (576, False), 'E_GATHER_K128': (128, False), 'AZ_K128': (128, True),
        'E_GATHER_K64': (64, False), 'AZ_K64': (64, True),
        'E_GATHER_K32': (32, False), 'AZ_K32': (32, True)}
MODELS = {'v15': '/media/disk2/YZX/doct/FastV/llava-v1.5-7b',
          'next': '/media/disk2/YZX/doct/FastV/llava-v1.6-vicuna-7b'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_ast(path):
    return next(x for x in ast.parse(Path(path).read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == 'eval_model')


def check_science_identity():
    official, current = function_ast(ORIGINAL_SCIENCE), function_ast(SCRIPT)
    current = copy.deepcopy(current)
    current.body = [x for x in current.body if not (
        isinstance(x, ast.If) and isinstance(x.test, ast.Call)
        and isinstance(x.test.func, ast.Name) and x.test.func.id == 'getattr')]
    class NormalizeLogging(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'runtime_retained_tokens'
                   for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Dict(self, node):
            self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'metadata':
                    node.values[i] = ast.Dict(keys=[], values=[])
            return node
    current = NormalizeLogging().visit(current)
    if ast.dump(official) != ast.dump(current):
        raise RuntimeError('Science wrapper input/generation diverged from official source beyond optional method installation and logging')
    return dict(input_and_generation_ast_identical=True,
                permitted_differences=['optional AnchorZip installation', 'runtime retained-token metadata'],
                official_source_sha256=sha(ORIGINAL_SCIENCE), wrapper_sha256=sha(SCRIPT))


def check_questions():
    questions = json.loads(QUESTION.read_text())
    source = json.loads(SOURCE.read_text())
    source = {str(x['id']):x for x in source}
    ids = [str(x['id']) for x in questions]
    if len(ids) != 2017 or len(set(ids)) != 2017 or not all('image' in x for x in questions):
        raise RuntimeError('Expected all 2017 unique image questions')
    if not all(source[str(x['id'])] == x for x in questions):
        raise RuntimeError('Subset question content differs from original shipped CQM-A')
    splits = json.loads((BASE/'pid_splits.json').read_text())['test']
    if {str(x) for x in splits} != set(ids) or len(splits) != 2017:
        raise RuntimeError('Official evaluator base does not contain precisely the fixed 2017 IDs')
    return questions


def check_predictions(path, questions):
    rows = [json.loads(line) for line in Path(path).open()]
    ids = [str(x['question_id']) for x in rows]
    if len(rows) != 2017 or len(set(ids)) != 2017 or set(ids) != {str(x['id']) for x in questions}:
        raise RuntimeError(f'Incomplete/duplicate SQA predictions: {path}')
    if any(str(x.get('text','')).strip() in ('','FAILED') for x in rows):
        raise RuntimeError(f'Empty/failed SQA predictions: {path}')
    return len(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', choices=['v15','next','both'], default='v15')
    parser.add_argument('--arms', default='E_GATHER_K128,AZ_K128')
    parser.add_argument('--generate', action='store_true')
    args=parser.parse_args()
    arms=args.arms.split(',')
    if len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        parser.error('Choose unique arms from '+','.join(ARMS))
    input_identity=check_science_identity()
    questions=check_questions()
    models=list(MODELS) if args.model=='both' else [args.model]
    jobs=[]
    for model_name in models:
        for arm in arms:
            budget, az=ARMS[arm]
            if arm=='FULL' and model_name=='next':
                budget=0  # keep-all AnyRes control; no crop allocation
            output=OUT/'sqa_shipped_control'/model_name/f'{arm}.jsonl'
            cmd=[sys.executable,str(SCRIPT),'--model-path',MODELS[model_name],
                 '--question-file',str(QUESTION),'--image-folder',str(LLAVA/'playground/data/eval/scienceqa/test'),
                 '--answers-file',str(output),'--temperature','0','--conv-mode','vicuna_v1',
                 '--visual_token_num',str(budget),'--beta','2.0','--alpha','0.5','--single-pred-prompt']
            if az:
                cmd.append('--anchorzip')
            scorer=[sys.executable,str(LLAVA/'llava/eval/eval_science_qa.py'),
                    '--base-dir',str(BASE),'--result-file',str(output),
                    '--output-file',str(output.with_suffix('.output.json')),
                    '--output-result',str(output.with_suffix('.result.json'))]
            jobs.append(dict(model=model_name,arm=arm,output=str(output),
                             generate_command=cmd,score_command=scorer,
                             generation_log=str(output.with_suffix('.log')),
                             score_log=str(output.with_suffix('.score.txt'))))
    plan=dict(protocol='shipped_CQM_A_image2017_control', n_questions=2017,
              source_file=str(SOURCE), source_sha256=sha(SOURCE),
              fixed_subset=str(QUESTION), fixed_subset_sha256=sha(QUESTION),
              input_identity=input_identity,
              frozen=dict(alpha=.5,beta=2.,lambda_completion=.25,conv_mode='vicuna_v1',
                          temperature=0,max_new_tokens=1024,single_pred_prompt=True),
              official_evaluator_sha256=sha(LLAVA/'llava/eval/eval_science_qa.py'),
              reference_full=str(OUT/'sqa_FULL_CQMA_image2017_result.json'),
              caveat='Entire SQA image subset; not 4241 overall; author CQM-I file unavailable', jobs=jobs)
    print(json.dumps(plan,indent=2),flush=True)
    if not args.generate:
        return
    for variable in ('USE_LLAVA_ARCH_CDPRUNER','USE_LLAVA_ARCH_DIVPRUNE','USE_LLAVA_ARCH_HIPRUNE','USE_LLAVA_ARCH_ABLATION'):
        if os.environ.get(variable)=='1':
            raise RuntimeError(f'Unexpected architecture: {variable}')
    env=dict(os.environ, PYTHONPATH=str(LLAVA), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for job in jobs:
        output=Path(job['output']);sidecar=Path(str(output)+'.protocol.json')
        if any(p.exists() for p in (output,sidecar,Path(job['generation_log']),Path(job['score_log']))):
            raise FileExistsError(f'Preserving previous artifact: {output}')
        output.parent.mkdir(parents=True,exist_ok=True)
        metadata={k:v for k,v in plan.items() if k!='jobs'}
        metadata['job']=job
        sidecar.write_text(json.dumps(metadata,indent=2))
        with open(job['generation_log'],'x') as log:
            subprocess.run(job['generate_command'],env=env,cwd=LLAVA,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        check_predictions(output,questions)
        with open(job['score_log'],'x') as log:
            subprocess.run(job['score_command'],env=env,cwd=LLAVA,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(output.with_suffix('.result.json').read_text())
        if result['count']!=2017:
            raise RuntimeError('Scorer denominator changed')
        print(json.dumps(dict(model=job['model'],arm=job['arm'],n=result['count'],
                             correct=result['correct'],IMG=result['acc'])),flush=True)


if __name__=='__main__':
    main()
