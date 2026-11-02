"""Explicitly allocated, single-attempt MMB panel controller; never auto-queues."""
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path
import common as c
import cpu_container


def gpu_pids():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    return sorted({int(x.strip()) for x in p.stdout.splitlines() if x.strip()})

def write_state(value):
    target=c.OUT/'state.json';tmp=c.OUT/'state.json.tmp'
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,target)

def main(note_path):
    assert Path(note_path).resolve()==(c.HERE/'root_gpu_allocation.json').resolve()
    plan=c.read(c.PLAN);c.check(plan);note=c.read(note_path)
    assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==plan['run_id']
    assert note['plan_sha256']==c.sha(c.PLAN) and note['GPU_device']==0
    assert note['single_attempt'] is True and note['automatic_retry'] is False
    assert note['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert note['cpu_preflight_sha256']==c.sha(c.HERE/'cpu_preflight.json')
    pre=c.read(c.HERE/'cpu_preflight.json');assert pre['success'] and pre['plan_sha256']==c.sha(c.PLAN)
    others=note['allowed_other_gpu_processes'];assert len(others)<=1
    assert gpu_pids()==sorted(x['pid'] for x in others)
    assert all(c.identity(x['pid'])==x for x in others)
    c.OUT.mkdir(exist_ok=False)
    controller=c.identity(os.getpid())
    c.save_new(c.HERE/'launch.claim.json',dict(plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),controller=controller))
    state=dict(status='controller_started',plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path),controller=controller,original_worker_exited=False,GPU_pid_absent=False)
    write_state(state);process=None;worker_id=None;verifier=None;container_name=None
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(note['GPU_device']),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        env.pop('USE_LLAVA_ARCH_CDPRUNER',None);env.pop('USE_LLAVA_ARCH_DIVPRUNE',None);env.pop('USE_LLAVA_ARCH_HIPRUNE',None);env.pop('USE_LLAVA_ARCH_ABLATION',None)
        with (c.OUT/'worker.log').open('xb') as log:
            process=subprocess.Popen([c.PYTHON,'-B',str(c.HERE/'worker.py'),'--allocation-note',str(Path(note_path).resolve())],cwd=str(c.HERE),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            worker_id=c.identity(process.pid)
            if worker_id is None:raise RuntimeError('worker exited before identity could be established')
            state.update(status='native_running',worker=worker_id);write_state(state)
            rc=process.wait()
        alive=(c.identity(process.pid)==worker_id)
        current_gpu=gpu_pids();absent=process.pid not in current_gpu
        state.update(status='native_exited_validation_pending' if rc==0 and not alive and absent else 'native_failed',native_exit_code=rc,
                     original_worker_exited=not alive,GPU_pid_absent=absent,gpu_pids_after=current_gpu,worker_log_sha256=c.sha(c.OUT/'worker.log'))
        write_state(state)
        if state['status']!='native_exited_validation_pending':raise RuntimeError('native worker failed or GPU release not established')
        c.save_new(c.OUT/'native_release.json',state)
        # CPU validation runs in a container with no device passthrough and no network.
        container_name='mmb-full-verifier-'+str(os.getpid())
        cidfile=c.OUT/'verifier.cid'
        command=cpu_container.command([str(c.HERE/'verify_generation.py'),'--terminal'],container_name,write_output=True,cidfile=cidfile)
        with (c.OUT/'verifier.log').open('xb') as log:
            verifier=subprocess.Popen(command,cwd=str(c.HERE),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(verifier_client=c.identity(verifier.pid),verifier_container_name=container_name);write_state(state)
            verifier_rc=verifier.wait()
        inspected=subprocess.run(['docker','inspect',container_name],capture_output=True,text=True)
        container_absent=inspected.returncode!=0 and 'No such' in inspected.stderr
        state.update(status='complete' if verifier_rc==0 and container_absent else 'verification_failed',verifier_exit_code=verifier_rc,verifier_container_absent=container_absent,
                     verifier_log_sha256=c.sha(c.OUT/'verifier.log'))
        write_state(state)
        if verifier_rc or not container_absent:raise RuntimeError('independent CPU verifier failed or container remains')
    except BaseException as exc:
        # Reap only our own child and record unknown release when driver query fails.
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if verifier is not None and verifier.poll() is None:
            if container_name:subprocess.run(['docker','stop','--time','5',container_name],capture_output=True,text=True)
            try:verifier.wait(timeout=15)
            except subprocess.TimeoutExpired:verifier.kill();verifier.wait()
        if container_name:
            inspected=subprocess.run(['docker','inspect',container_name],capture_output=True,text=True)
            state['verifier_container_absent']=(inspected.returncode!=0 and 'No such' in inspected.stderr)
        alive=(worker_id is not None and c.identity(worker_id['pid'])==worker_id)
        try:current_gpu=gpu_pids();absent=(worker_id is not None and worker_id['pid'] not in current_gpu)
        except Exception:current_gpu=None;absent=False
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),original_worker_exited=(worker_id is not None and not alive),GPU_pid_absent=absent,gpu_pids_after=current_gpu)
        write_state(state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allocation-note',required=True);a=p.parse_args();main(a.allocation_note)
