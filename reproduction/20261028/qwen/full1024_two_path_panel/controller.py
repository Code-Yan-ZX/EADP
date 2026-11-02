"""Explicit one-shot future root allocation; no automatic launch/retry/queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import common as c
sys.path.insert(0,str(c.PREV))
from run_panel import container_command,gpu_pids,identity
from namespace_guards import verify_mounts

def run(note_path):
    p=c.read(c.PLAN);c.plan_checks(p);note=c.read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID
    assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_slot'] in ('A','B') and note['GPU_device']==0
    assert note['single_attempt'] and not note['automatic_retry'] and note['maximum_concurrent_models_including_loading']==2
    assert note['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    pre=c.read(c.HERE/'CPU_preflight_pass.json')
    assert pre['status']=='passed' and pre['plan_sha256']==c.sha(c.PLAN)
    assert note['CPU_preflight_sha256']==c.sha(c.HERE/'CPU_preflight_pass.json')
    assert pre['container_record_sha256']==c.sha(c.HERE/'CPU_preflight_container_record.json')
    others=note['allowed_other_gpu_processes'];assert len(others)<2
    assert gpu_pids()==sorted(x['pid'] for x in others)
    for x in others:
        actual=identity(x['pid']);assert actual and all(actual[k]==x[k] for k in ('pid','start_ticks','boot_id'))
    assert not (c.HERE/'execution').exists() and not (c.HERE/'state.json').exists()
    old=c.read(c.PREV/'run_plan.json');name='eadp-qwen-doc16-full1024-paths'
    cmd=container_command(old,[str(c.HERE/'worker.py'),'--execute','--allocation-note',str(note_path)],True,name)
    controller=identity(__import__('os').getpid())
    claim=c.HERE/'launch.claim.json'
    with claim.open('x') as f:json.dump(dict(controller=controller,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)),f,indent=2)
    state=dict(status='starting',controller=controller,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),command=cmd,automatic_retry=False)
    c.save(c.HERE/'state.json',state)
    worker=None
    try:
        with (c.HERE/'execution.log').open('x') as log:
            worker=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',docker_client=identity(worker.pid));c.save(c.HERE/'state.json',state)
            while worker.poll() is None:
                if 'container_identity' not in state:
                    probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
                    if probe.returncode==0:
                        container=json.loads(probe.stdout)[0];pid=container['State']['Pid'];actual=identity(pid) if pid else None
                        if actual:
                            assert container['Image']==old['container_image']
                            group=Path('/proc',str(pid),'cgroup').read_text();assert container['Id'] in group
                            state['container_identity']=dict(container_id=container['Id'],host_init=actual,host_cgroup=group)
                            state['mount_proof']=verify_mounts(old,container,gpu=True);c.save(c.HERE/'state.json',state)
                time.sleep(1)
            code=worker.wait()
        state.update(status='container_exited_validation_pending',exit_code=code);c.save(c.HERE/'state.json',state);assert code==0
        original=state['container_identity']['host_init'];now=identity(original['pid'])
        assert now is None or now['start_ticks']!=original['start_ticks']
        assert original['pid'] not in gpu_pids()
        check=subprocess.run(['docker','inspect',state['container_identity']['container_id']],capture_output=True,text=True)
        assert check.returncode!=0 or not json.loads(check.stdout)[0]['State']['Running']
        with (c.HERE/'verification.log').open('x') as log:
            ver=subprocess.run(['/home/dell/miniconda3/envs/llava_pruner/bin/python','-B',str(c.HERE/'verify_terminal.py')],stdout=log,stderr=subprocess.STDOUT)
        state['CPU_verifier_exit_code']=ver.returncode;c.save(c.HERE/'state.json',state);assert ver.returncode==0
        report=c.read(c.HERE/'execution/independent_verification.json');assert report['success'] and report['n_native_generations']==32
        c.save(c.HERE/'execution/release.json',dict(success=True,plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),
            independent_verification_sha256=c.sha(c.HERE/'execution/independent_verification.json'),
            actual_native_container_exited=True,controller_exit_requires_separate_root_pty_proof=True,benchmark_result=False))
        state.update(status='complete',release_sha256=c.sha(c.HERE/'execution/release.json'));c.save(c.HERE/'state.json',state)
    except BaseException as e:
        state.update(status='terminal_failure_no_retry',error=repr(e));c.save(c.HERE/'state.json',state)
        if worker is not None and worker.poll() is None:
            stopped=subprocess.run(['docker','stop','--time','10',name],capture_output=True,text=True)
            state['emergency_stop']=dict(exit_code=stopped.returncode,stdout=stopped.stdout,stderr=stopped.stderr)
            try:state['docker_client_exit_after_stop']=worker.wait(timeout=30)
            except subprocess.TimeoutExpired:state['docker_client_still_running']=True
        probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
        state['container_after_failure']=dict(inspect_exit=probe.returncode,stdout=probe.stdout,stderr=probe.stderr)
        state['failure_does_not_prove_GPU_released']=True
        c.save(c.HERE/'state.json',state);c.save(c.HERE/'failure.json',state);raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['check','run']);ap.add_argument('--allocation-note',type=Path)
    a=ap.parse_args()
    if a.action=='check':c.plan_checks(c.read(c.PLAN));print(json.dumps(dict(status='CPU_only',GPU_started=False)))
    else:assert a.allocation_note is not None;run(a.allocation_note.resolve())
