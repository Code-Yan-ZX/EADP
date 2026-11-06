"""Bind all historical τ=.2 records to both finished no-GT reviews."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
ORIGINAL=BASE/'e6/pixelprune_t020_dev384'
FORMAL=BASE/'e6/pixelprune_t020_accept'
REVIEW=BASE/'e1_independent_review'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervised(mon, script, out=None):
    s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
    if (s.get('worker_pid')!=t.get('worker_pid') or type(s.get('worker_pid')) is not int
        or s['worker_pid']<=0 or t.get('reason')!='completed' or t.get('worker_returncode')!=0
        or t.get('remaining_process_group')!=[]
        or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]):
        raise ValueError('supervisor exit/command mismatch')
    if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):
        raise ValueError('review output path mismatch')
    return s,t

def validate():
    plan=ORIGINAL/'plan001.json';formal=FORMAL/'verification001.json';proof=REVIEW/'pixel_t020_dev384_result_review.json'
    verifier=FORMAL/'verify.py'; reviewer=REVIEW/'review_pixel_t020_dev384_result.py'
    fmon=FORMAL/'monitor_verify001';rmon=REVIEW/'monitor_pixel_t020_dev384_result';run=ORIGINAL/'monitor_smoke001'
    supervised(fmon,verifier);supervised(rmon,reviewer,proof);supervised(run,ORIGINAL/'launch.py')
    f=read(formal);r=read(proof); p=read(plan)
    if (f.get('status')!='passed' or f.get('questions')!=384 or f.get('physical_generations')!=384
        or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[]
        or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=384
        or r.get('physical_generations')!=384 or r.get('GT_opened') is not False
        or r.get('scoring_performed') is not False or r.get('errors')!=[]):raise ValueError('baseline prior not accepted')
    actual={'formal_sha256':formal,'formal_started_sha256':fmon/'started.json','formal_terminal_sha256':fmon/'terminal.json',
            'reviewer_sha256':reviewer,'run_started_sha256':run/'started.json','run_terminal_sha256':run/'terminal.json',
            'run_exit_sha256':ORIGINAL/'run_smoke001/exit.json','verification_plan_sha256':FORMAL/'verification_plan001.json'}
    if any(r.get(k)!=sha(v) for k,v in actual.items()) or f.get('verifier_sha256')!=sha(verifier):
        raise ValueError('baseline evidence changed')
    if len(p['anchors'])!=384 or len({(x['task'],str(x['qid']),x['group_id']) for x in p['anchors']})!=384:
        raise ValueError('baseline cohort')
    files={}; directory=ORIGINAL/'run_smoke001/observations'
    for anchor in p['anchors']:
        name=anchor['task']+'_'+str(anchor['qid'])
        for suffix in ('.json','_generated.json'):
            path=str((directory/(name+suffix)).resolve());digest=sha(path)
            if f.get('files_sha256',{}).get(path)!=digest or r.get('per_question_files_sha256',{}).get(path)!=digest:
                raise ValueError('baseline file not doubly bound: '+path)
            files[path]=digest
    if len(files)!=768:raise ValueError('baseline file denominator')
    return {'plan_sha256':sha(plan),'formal_sha256':sha(formal),'review_sha256':sha(proof),
            'files_sha256':files,'questions':384,'GT_opened':False,
            'prediction_scope':'existing DEV predictions opened only for source hashes'}

def validate_wavefront_prior():
    run=BASE/'e6/pixel_wavefront_model_t020_run_v2';formal=BASE/'e6/pixel_wavefront_t020_accept_v2';proof=REVIEW/'pixel_wavefront_t020_run_v2_result_review.json'
    reviewer=REVIEW/'review_pixel_wavefront_t020_run_v2_result.py';reviewmon=REVIEW/'monitor_pixel_wavefront_t020_run_v2_result'
    supervised(run/'monitor_run001',run/'controller.py');supervised(formal/'monitor_verify001',formal/'verify.py')
    supervised(reviewmon,reviewer,proof)
    f=read(formal/'verification001.json');r=read(proof);execution=read(run/'execution_plan001.json')
    if (f.get('status')!='passed' or r.get('status')!='passed' or r.get('independent_review') is not True
        or any(x.get('images')!=3 or x.get('jobs')!=6 or x.get('pairs')!=3 or x.get('errors')!=[]
               or x.get('GT_opened') is not False or x.get('scoring_performed') is not False for x in (f,r))):
        raise ValueError('three-anchor reference not accepted')
    paths={'formal_sha256':formal/'verification001.json','formal_started_sha256':formal/'monitor_verify001/started.json',
           'formal_terminal_sha256':formal/'monitor_verify001/terminal.json','run_started_sha256':run/'monitor_run001/started.json',
           'run_terminal_sha256':run/'monitor_run001/terminal.json','reviewer_sha256':reviewer}
    if any(r.get(k)!=sha(p) for k,p in paths.items()) or f.get('plan_sha256')!=sha(run/'execution_plan001.json'):
        raise ValueError('three-anchor evidence changed')
    candidate=BASE/'e6/pixel_wavefront_cpu/candidate.py'
    if execution['source_sha256'].get(str(candidate))!=sha(candidate):raise ValueError('candidate source differs from three-anchor run')
    return {'formal_sha256':sha(formal/'verification001.json'),'review_sha256':sha(proof),'candidate_sha256':sha(candidate)}
