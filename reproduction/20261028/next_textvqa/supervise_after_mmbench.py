"""CPU-only waiter for root-preallocated slot B, followed by exactly one run.

Terminal failures stop this controller; no automatic rerun or slot-A takeover.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import repro_full_default as full

HERE=Path(__file__).resolve().parent
MMB=HERE.parent/'mmbench/streamwait_control'
EXPECTED_PLAN='7de9048ad555683ca95d1bba27193c30c641b2decd5ae2d4626a359b11e9462d'
EXPECTED_WORKER='466e3c7be72cf5db3a12409e50fcb47286f6d054af4fa6f6f0fad351a6d4e94d'
MMB_PID=2367889
MMB_TICKS=1331417519
PYTHON='/home/dell/miniconda3/envs/llava_pruner/bin/python'
AUTHORIZATION=('Parent /root explicitly preallocated subsequent slot B to NeXT public-default K128/alpha.5/beta1/full5000/identical three waits. '
 'Launch only after MMB streamwait_control all4377/1176groups, author XLSX, same scoring and independent verification succeed; '
 'native worker2367889/startticks1331417519 actually exits; all NeXT plan/source/weights/inputs remain frozen. '
 'Root retains slot A for Qwen. Loading counts as a model; global max2. Preserve any terminal failure, no rerun or slot takeover.')


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def save(name,data):full.save(HERE/name,data)


def process_identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    try:
        stat=path.read_text();fields=stat[stat.rindex(')')+2:].split()
        return dict(pid=pid,startticks=int(fields[19]))
    except (FileNotFoundError,ProcessLookupError):return None


def frozen_gate():
    assert full.sha(full.PLAN)==EXPECTED_PLAN, 'NeXT registered plan SHA changed'
    assert full.sha(full.__file__)==EXPECTED_WORKER, 'NeXT worker SHA changed'
    full.check_plan(read(full.PLAN))
    assert not full.RUN.exists(), 'Never restart an existing NeXT full run'
    assert 'torch' not in sys.modules, 'Waiting controller must not import/load torch'


def predecessor_gate():
    for name in ['supervisor_failure.json','CPU_finalizer_failure.json']:
        if (MMB/name).exists():raise RuntimeError('Predecessor failure preserved: '+str(MMB/name))
    paths={name:MMB/name for name in ['state.json','independent_verification.json',
        'supervisor_state.json','paired_full_comparison.json','CPU_finalizer_state.json']}
    try:
        available={name:read(path) for name,path in paths.items() if path.exists()}
    except json.JSONDecodeError:
        return False,dict(reason='waiting_predecessor_record_write')
    for name,record in available.items():
        status=str(record.get('status','')).lower()
        if 'fail' in status or status in ['error','aborted','cancelled']:
            raise RuntimeError('Predecessor terminal state: '+name+' '+status)
        if record.get('exit_code') not in [None,0]:
            raise RuntimeError('Predecessor nonzero exit: '+name)
    if len(available)<len(paths):return False,dict(reason='waiting_complete_predecessor_files',present=list(available))
    state=available['state.json'];independent=available['independent_verification.json']
    supervisor=available['supervisor_state.json'];paired=available['paired_full_comparison.json']
    finalizer=available['CPU_finalizer_state.json']
    if any(record.get('status')!='complete' for record in [state,supervisor,finalizer]):
        return False,dict(reason='waiting_predecessor_terminal_validation')
    assert state['exit_code']==supervisor['exit_code']==0
    assert state['native_process']['pid']==MMB_PID and state['native_process']['start_ticks']==MMB_TICKS
    assert state['author_code_unchanged']
    assert independent['success'] and independent['all_rows_checked'] and not independent['errors']
    assert independent['n_rows']==4377 and independent['n_circular_groups']==1176
    assert paired['success'] and paired['n_rows']==4377 and paired['n_circular_groups']==1176
    assert finalizer['GPU_slot_B_released'] and finalizer['comparison_sha256']==full.sha(paths['paired_full_comparison.json'])
    assert all(evidence['original_worker_exited'] for evidence in paired['worker_exit'])
    assert supervisor['independent_verification_sha256']==full.sha(paths['independent_verification.json'])
    for record in paired['sources']:assert full.sha(record['path'])==record['sha256']
    score_path=MMB/'local_circular_score.json';score=read(score_path)
    assert full.sha(score_path)==state['score_sha256']
    assert score['n_rows']==4377 and score['n_groups']==1176
    assert abs(score['value']-independent['circular_accuracy_percent'])<1e-12
    conversion=read(MMB/'conversion.json')
    assert conversion['exit_code']==0
    assert full.sha(conversion['output'])==independent['submission_sha256']
    protocol=read(MMB/'protocol.json')
    for record in protocol['source_files']:assert full.sha(record['path'])==record['sha256']
    current=process_identity(MMB_PID)
    if current and current['startticks']==MMB_TICKS:
        return False,dict(reason='waiting_original_MMB_worker_exit')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                  '--format=csv,noheader'],text=True)
    gpu_lines=[s for s in apps.splitlines() if s.strip()]
    assert not any(int(s.split(',',1)[0])==MMB_PID for s in gpu_lines), 'MMB worker still on GPU'
    if len(gpu_lines)>1:return False,dict(reason='waiting_global_gpu_model_count_below2',GPU_processes=apps)
    return True,dict(reason='all_parent_gates_passed',verified_utc=utc(),
        predecessor={name:dict(path=str(path),sha256=full.sha(path)) for name,path in paths.items()},
        author_XLSX=dict(path=conversion['output'],sha256=independent['submission_sha256']),
        MMB_original_worker_exited=True,current_identity_if_pid_reused=current,
        GPU_processes_before_launch=apps,NeXT_plan_sha256=EXPECTED_PLAN,NeXT_worker_sha256=EXPECTED_WORKER)


def main(check_only=False):
    frozen_gate()
    if check_only:
        passed,evidence=predecessor_gate();print(json.dumps(dict(CPU_only=True,would_launch=False,gate_passed=passed,gate=evidence)))
        return
    assert not (HERE/'full_default_K128.supervisor_registration.json').exists(), 'No automatic supervisor restart'
    registration=dict(registered_utc=utc(),authorization=AUTHORIZATION,identity=process_identity(os.getpid()),
        plan_sha256=EXPECTED_PLAN,worker_sha256=EXPECTED_WORKER,supervisor_sha256=full.sha(__file__),
        GPU_loaded_by_waiter=False,command=[PYTHON,str(HERE/'repro_full_default.py'),'run','--gpu-authorized'])
    save('full_default_K128.supervisor_registration.json',registration)
    state=dict(status='waiting_MMB_validation_and_exit',**registration)
    save('full_default_K128.supervisor_state.json',state);print(json.dumps(state),flush=True)
    previous_reason=None
    while True:
        passed,evidence=predecessor_gate()
        if evidence['reason']!=previous_reason:
            state.update(gate=evidence,observed_utc=utc());save('full_default_K128.supervisor_state.json',state)
            print(json.dumps(dict(event='gate',**evidence)),flush=True);previous_reason=evidence['reason']
        if passed:break
        time.sleep(15)
    frozen_gate()
    passed,evidence=predecessor_gate();assert passed
    save('full_default_K128.launch_gate.json',dict(authorization=AUTHORIZATION,**evidence))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    with (HERE/'full_default_K128.generation.log').open('x') as log:
        worker=subprocess.Popen(registration['command'],cwd=HERE.parents[2],env=env,
                                stdout=log,stderr=subprocess.STDOUT)
        state.update(status='full_run_started',launched_utc=utc(),worker=process_identity(worker.pid))
        save('full_default_K128.supervisor_state.json',state);print(json.dumps(dict(event='worker_started',worker=state['worker'])),flush=True)
        first=False
        while worker.poll() is None:
            answer_path=full.RUN/'answers.jsonl'
            if not first and answer_path.exists() and answer_path.stat().st_size:
                with answer_path.open() as f:
                    line=f.readline()
                try:answer=json.loads(line)
                except json.JSONDecodeError:
                    time.sleep(2);continue
                state['first_prediction']=answer;state['first_prediction_observed_utc']=utc()
                save('full_default_K128.supervisor_state.json',state)
                print(json.dumps(dict(event='first_prediction',question_id=answer['question_id'],text=answer['text'])),flush=True)
                first=True
            time.sleep(5)
    code=worker.wait()
    state.update(status='complete' if code==0 else 'terminal_failure_no_retry',exit_code=code,finished_utc=utc())
    if code==0:
        run_state=read(full.RUN/'state.json');score=read(full.RUN/'official.score.json')
        assert run_state['status']=='complete' and score['n']==5000
        state.update(score_sha256=full.sha(full.RUN/'official.score.json'),score_value=score['value'],
            actual_worker_exited=process_identity(worker.pid) is None)
    save('full_default_K128.supervisor_state.json',state);print(json.dumps(state),flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:main(args.check_only)
    except Exception as error:
        save('full_default_K128.supervisor_failure.json',dict(status='terminal_failure_no_retry',utc=utc(),
            error=repr(error),identity=process_identity(os.getpid()),automatic_retry=False))
        raise
