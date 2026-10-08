"""CPU-only independent diagnosis; never writes original evidence or starts models."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path('/media/disk2/YZX/research/EADP_amp')
STABILITY=ROOT/'Qwen_vl/outputs/audit_followup_20261008/next_gap_diagnosis_20261008/stability'
WORKER=STABILITY/'next_text_full_streamwait.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def probe():
    spec=importlib.util.spec_from_file_location('independent_frozen_fullworker',WORKER)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    report=dict(python=sys.version,python_executable=sys.executable,hashseed=os.environ.get('PYTHONHASHSEED'),worker_sha256=sha(WORKER),arms={})
    for arm in ('EADP_beta2','AZ_beta2'):
        paths=worker.paths(arm,1);protocol=json.loads(paths['control_protocol'].read_text())
        questions=[json.loads(line) for line in Path(protocol['parameters']['question_file']).open()]
        stored=json.loads(paths['score'].read_text())
        fresh=worker.validate_and_score(paths['prediction'],paths['runtime'],questions,protocol['parameters'])
        canonical=json.loads(json.dumps(fresh))
        exact_diff={key:dict(stored=stored.get(key),fresh=value) for key,value in fresh.items() if stored.get(key)!=value}
        json_diff={key:dict(stored=stored.get(key),fresh=value) for key,value in canonical.items() if stored.get(key)!=value}
        # Compute each question's official leave-one-out score independently;
        # its mathematical values are tenths, so also report the integer total.
        rows=[json.loads(line) for line in paths['prediction'].open()]
        module_path=worker.LLAVA/'llava/eval/m4c_evaluator.py'
        es=importlib.util.spec_from_file_location('independent_official_m4c',module_path)
        em=importlib.util.module_from_spec(es);es.loader.exec_module(em)
        tree=ast.parse((worker.LLAVA/'llava/eval/eval_textvqa.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prompt_processor')
        ns={'re':__import__('re')};exec(compile(ast.Module(body=[fn],type_ignores=[]),'official_prompt_processor','exec'),ns)
        annotations={(q['image_id'],q['question'].lower()):q for q in json.loads(worker.GT.read_text())['data']}
        ev=em.TextVQAAccuracyEvaluator();units=[];off=[]
        for row in rows:
            gt=annotations[(row['question_id'],ns['prompt_processor'](row['prompt']))]['answers']
            pred=ev.answer_processor(row['text']);answers=[ev.answer_processor(a) for a in gt]
            count=answers.count(pred)
            exact=min(count*3,10)
            got=ev._compute_answer_scores(gt).get(pred,0.)
            if abs(got-exact/10)>1e-14:raise AssertionError('Official per-question formula differs')
            units.append(exact);off.append(got)
        unit_sha=hashlib.sha256(json.dumps(units,separators=(',',':')).encode()).hexdigest()
        report['arms'][arm]=dict(exact_diff=exact_diff,json_normalized_diff=json_diff,
            exact_official_score=canonical['accuracy_percent'],stored_score=stored['accuracy_percent'],
            mathematical_score_percent=sum(units)/500,official_per_question_score_tenths_sha256=unit_sha,
            official_per_question_float_sha256=hashlib.sha256(json.dumps(off,separators=(',',':')).encode()).hexdigest(),
            max_per_question_roundoff=max(abs(a-b/10) for a,b in zip(off,units)),
            prediction_sha256=sha(paths['prediction']),runtime_sha256=sha(paths['runtime']),score_sha256=sha(paths['score']),
            n=len(rows),no_GPU=True)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();r=probe()
    output=json.dumps(r,ensure_ascii=False,indent=2)
    if args.output:
        if args.output.exists():raise FileExistsError('Preserve triage output')
        args.output.write_text(output)
    print(output)
if __name__=='__main__':main()
