"""Fail-closed host/container admission for the exploratory full panel."""
import json
from pathlib import Path
import baseline

D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'
sha=baseline.sha;read=baseline.read

def check_plan():
    p=read(D/'plan001.json')
    if p.get('questions')!=384 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False:
        raise ValueError('invalid plan')
    if any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('frozen source changed')
    current=baseline.validate()
    if any(current[k]!=p['baseline'][k] for k in ('plan_sha256','formal_sha256','review_sha256','files_sha256')):
        raise ValueError('baseline binding changed')
    if baseline.validate_wavefront_prior()!=p['wavefront_prior']:raise ValueError('three-anchor reference changed')
    return p

def check_review(stage):
    if stage not in ('preflight','final'):raise ValueError('bad stage')
    proof=D/('independent_'+stage+'_review.json')
    reviewer=R/('review_pixel_wavefront_full_'+stage+'.py')
    mon=R/('monitor_pixel_wavefront_full_'+stage)
    baseline.supervised(mon,reviewer,proof)
    baseline.supervised(D/'monitor_cpu001',D/'cpu_validation.py')
    cpu=read(D/'cpu_proof001.json')
    if (cpu.get('status')!='passed' or cpu.get('plan_sha256')!=sha(D/'plan001.json')
        or cpu.get('source_sha256')!=read(D/'plan001.json')['source_sha256']):
        raise ValueError('CPU proof or frozen sources mismatch')
    v=read(proof)
    if (v.get('status')!='passed' or v.get('independent_review') is not True
        or v.get('scope')!=stage or v.get('errors')!=[]
        or v.get('plan_sha256')!=sha(D/'plan001.json')
        or v.get('reviewer_sha256')!=sha(reviewer)
        or v.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json')
        or v.get('cpu_started_sha256')!=sha(D/'monitor_cpu001/started.json')
        or v.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu001/terminal.json')):
        raise ValueError('independent review binding failed')
    if stage=='final':
        expected={'preflight_sha256':D/'preflight001.json','preflight_exit_sha256':D/'run_preflight001/exit.json',
                  'preflight_launch_sha256':D/'run_preflight001/launch.json','preflight_started_sha256':D/'monitor_preflight001/started.json',
                  'preflight_terminal_sha256':D/'monitor_preflight001/terminal.json'}
        if any(v.get(k)!=sha(x) for k,x in expected.items()):raise ValueError('preflight binding failed')
        baseline.supervised(D/'monitor_preflight001',D/'launch.py')
        pre=read(D/'preflight001.json');ex=read(D/'run_preflight001/exit.json')
        if (pre.get('status')!='passed' or pre.get('cuda_initialized') is not False
            or pre.get('model_loaded') is not False or pre.get('weight_sha256_verified')!=4
            or ex.get('container_stopped_verified') is not True or ex.get('container_exit_code')!=0
            or ex.get('returncode')!=0 or ex.get('failure_type') is not None):
            raise ValueError('preflight failed')
    return v

def require_preflight():
    check_plan();check_review('preflight')

def require_gpu():
    check_plan();check_review('preflight');check_review('final')
