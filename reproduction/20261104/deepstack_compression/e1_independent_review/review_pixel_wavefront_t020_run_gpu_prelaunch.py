"""Independent run-chain prelaunch review; reviewer authored wavefront candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B/'e6/pixel_wavefront_model_t020_run'
A = B/'e6/pixel_wavefront_t020_accept'
U = B/'e6/pixel_wavefront_t020_run_prepare'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def terminal(folder, script):
    s, t = read(folder/'started.json'), read(folder/'terminal.json')
    if (t.get('reason') != 'completed' or t.get('worker_returncode') != 0 or
        t.get('remaining_process_group') != [] or s.get('worker_pid') != t.get('worker_pid') or
        (Path(s['cwd'])/s['command'][1]).resolve() != script.resolve()):
        raise ValueError('supervisor mismatch '+str(folder))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    out = ap.parse_args().out.resolve()
    if out != D/'independent_gpu_prelaunch_review.json':
        raise ValueError('unexpected review output')
    plan_path, cpu_path = D/'execution_plan001.json', D/'cpu_check007/verification.json'
    plan, cpu, preplan = read(plan_path), read(cpu_path), read(D/'preflight_plan005.json')
    sources = {path:sha(path) for path in plan['source_sha256']}
    if (len(sources) != 436 or sources != plan['source_sha256'] or sources != preplan['source_sha256'] or
        sources != cpu['source_sha256_before'] or sources != cpu['source_sha256_after'] or
        cpu.get('status') != 'passed' or cpu.get('tests_returncode') != 0 or
        'Ran 31 tests' not in cpu.get('tests_stderr','') or cpu.get('GT_opened') is not False or
        cpu.get('model_loaded') is not False or cpu.get('gpu_used') is not False or
        cpu.get('existing_DEV_predictions_opened') is not True):
        raise ValueError('source or CPU identity')
    cs, ct = D/'monitor_cpu007/started.json', D/'monitor_cpu007/terminal.json'
    terminal(D/'monitor_cpu007',D/'cpu_validation.py')
    fs = terminal(U/'monitor_freeze_execution002',U/'freeze_execution_002.py')
    if fs['command'].count('--out') != 1 or (Path(fs['cwd'])/fs['command'][fs['command'].index('--out')+1]).resolve() != plan_path:
        raise ValueError('execution freeze output mismatch')
    if (plan.get('stage') != 't020 three-anchor six-container identity run' or plan.get('threshold') != .2 or
        plan.get('GT_opened') is not False or plan.get('scoring_performed') is not False or
        plan.get('existing_DEV_predictions_opened') is not True or
        (plan.get('inner_timeout_seconds'),plan.get('outer_timeout_seconds'),plan.get('per_arm_timeout_seconds'),
         plan.get('stale_seconds'),plan.get('grace_seconds'),plan.get('acceptance_reserve_seconds')) != (6300,6480,900,1200,90,3600) or
        plan.get('budget_source_sha256') != sha(B/'initial_plan.json') or
        plan.get('cpu_sha256') != sha(cpu_path) or plan.get('cpu_started_sha256') != sha(cs) or
        plan.get('cpu_terminal_sha256') != sha(ct)):
        raise ValueError('execution plan contract')
    anchors = read(B/'e6/pixel_wavefront_model_t020/preflight_plan002.json')['images']
    expected = {(x['anchor']['task'],str(x['anchor']['qid'])):x['anchor'] for x in anchors}
    jobs = plan['jobs']
    if (len(expected) != 3 or len(jobs) != 6 or [j['order'] for j in jobs] != list(range(6)) or
        {(j['task'],str(j['qid']),j['arm']) for j in jobs} != {(t,q,a) for t,q in expected for a in ('author','wavefront')}):
        raise ValueError('six-arm inventory')
    for j in jobs:
        ref = expected[(j['task'],str(j['qid']))]
        if any(j[k] != ref[k] for k in ('group_id','image_sha256','message_sha256')):
            raise ValueError('anchor mismatch')
    pre = R/'pixel_wavefront_t020_run_preflight_result_003_review.json'
    prevcode = R/'review_pixel_wavefront_t020_run_preflight_result_003.py'
    pm = R/'monitor_pixel_wavefront_t020_run_preflight_result_003'
    terminal(pm,prevcode)
    pr = read(pre)
    bound = {'plan_sha256':D/'preflight_plan005.json','worker_preflight_sha256':D/'run_preflight002/worker_preflight.json',
             'launch_sha256':D/'run_preflight002/launch.json','exit_sha256':D/'run_preflight002/exit.json',
             'started_sha256':D/'monitor_preflight002/started.json','terminal_sha256':D/'monitor_preflight002/terminal.json'}
    if (pr.get('status') != 'passed' or pr.get('independent_review') is not True or pr.get('gpu_authorized') is not False or
        pr.get('GT_opened') is not False or pr.get('source_sha256') != sources or pr.get('errors') != [] or
        pr.get('reviewer_sha256') != sha(prevcode) or any(pr.get(k)!=sha(v) for k,v in bound.items()) or
        plan.get('preflight_result_sha256') != sha(pre) or
        plan.get('preflight_result_started_sha256') != sha(pm/'started.json') or
        plan.get('preflight_result_terminal_sha256') != sha(pm/'terminal.json')):
        raise ValueError('reviewed no-GPU preflight changed')
    va = read(A/'verification_plan001.json')
    terminal(A/'monitor_cpu004',A/'test_verify.py')
    terminal(A/'monitor_freeze001',A/'freeze_plan.py')
    if (va.get('stage') != 'pre-run formal identity acceptance' or va.get('verifier_sha256') != sha(A/'verify.py') or
        va.get('test_sha256') != sha(A/'test_verify.py') or va.get('cpu_started_sha256') != sha(A/'monitor_cpu004/started.json') or
        va.get('cpu_terminal_sha256') != sha(A/'monitor_cpu004/terminal.json') or va.get('execution_plan_sha256') != sha(plan_path) or
        va.get('GPU_results_opened') is not False or va.get('GT_opened') is not False or
        va.get('scoring_performed') is not False or va.get('timing_claims') is not False or
        'Ran 3 tests' not in (A/'monitor_cpu004/stderr.log').read_text()):
        raise ValueError('formal acceptance plan changed')
    proof = {'status':'passed','independent_review':True,'scope':'t020_identity_gpu_prelaunch',
             'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,
             'source_sha256':sources,'plan_sha256':sha(plan_path),'preflight_result_sha256':sha(pre),
             'cpu_sha256':sha(cpu_path),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),
             'reviewer_sha256':sha(__file__),'acceptance_plan_sha256':sha(A/'verification_plan001.json'),
             'acceptance_freeze_started_sha256':sha(A/'monitor_freeze001/started.json'),
             'acceptance_freeze_terminal_sha256':sha(A/'monitor_freeze001/terminal.json'),
             'errors':[],'limitations':['Review author wrote the wavefront candidate; not an independent candidate implementation audit.',
               'Three DEV anchors, six isolated author/wavefront processes; identity only, not timing or quality.',
               'Per-step full KV arrays are not archived; runtime fingerprints are checked.',
               'Weight file metadata before and after cannot exclude transient changes.',
               'Earlier accepted DEV prediction evidence was opened for input and output identity.']}
    with out.open('x') as f:
        json.dump(proof,f,indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__=='__main__':
    main()
