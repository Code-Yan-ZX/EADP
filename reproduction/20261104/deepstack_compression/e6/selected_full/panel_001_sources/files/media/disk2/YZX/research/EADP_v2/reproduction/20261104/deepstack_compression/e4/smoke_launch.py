"""Bounded E4-A Docker smoke launcher; intentionally locked pending worker review."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from identity_worker import HERE, PLAN, check_e1, sha

# An absent external manifest is a hard lock. The manifest can only name a CPU
# proof and independent review of these exact immutable source hashes.
APPROVAL_PATH = HERE/'smoke_approval.json'

def check_approval():
    if not APPROVAL_PATH.is_file():raise ValueError('independent source-bundle review is not complete')
    bundle=json.loads(APPROVAL_PATH.read_text())
    if set(bundle)!={'cpu_proof','review_proof'}:raise ValueError('incomplete approval bundle')
    cpu_path=Path(bundle['cpu_proof']);review_path=Path(bundle['review_proof'])
    cpu=json.loads(cpu_path.read_text());review=json.loads(review_path.read_text())
    if (cpu.get('status')!='passed' or cpu.get('timeout') is not False or cpu.get('returncode')!=0
        or review.get('status')!='passed' or review.get('independent_review') is not True
        or review.get('cpu_proof_sha256')!=sha(cpu_path)):
        raise ValueError('CPU or independent review proof failed')
    names=('smoke_launch.py','smoke_worker.py','smoke_contract.py','budget_runner.py','budget_adapter.py',
           'off_adapter.py','skip_deepstack.py','identity_records.py','identity_worker.py','verify_k256.py',
           'decode_observer.py','test_budget_adapter.py','test_off_adapter.py','test_skip_deepstack.py','run_cpu_checks.py')
    paths=[HERE/x for x in names]+[HERE.parent/'e1_analysis/completion.py',HERE.parent/'e1_analysis/score.py',HERE.parent/'e1/contract.py']
    paths += [HERE.parent/'e1_analysis/metrics.py',HERE.parent/'e1_analysis/paired_stats.py']
    from run_cpu_checks import AUTHOR,PRUNER,VISION,CONFIG,LAUNCH_DEPENDENCIES
    paths += [AUTHOR,PRUNER,VISION,CONFIG]+LAUNCH_DEPENDENCIES
    current={str(x):sha(x) for x in paths}
    for path,digest in current.items():
        if (cpu.get('source_sha256_before',{}).get(path)!=digest or cpu.get('source_sha256_after',{}).get(path)!=digest
            or review.get('source_sha256',{}).get(path)!=digest):raise ValueError('unapproved dependency: '+path)
    return {'approval_manifest':{'path':str(APPROVAL_PATH.resolve()),'sha256':sha(APPROVAL_PATH)},
            'cpu_proof':{'path':str(cpu_path.resolve()),'sha256':sha(cpu_path)},
            'review_proof':{'path':str(review_path.resolve()),'sha256':sha(review_path)}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--e1-acceptance',type=Path,required=True)
    p.add_argument('--e1-acceptance-started',type=Path,required=True);p.add_argument('--e1-acceptance-terminal',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--timeout',type=int,required=True)
    a=p.parse_args()
    if not a.run_id.isalnum() or not 60 <= a.timeout <= 14400:raise ValueError('invalid id/deadline')
    proof=a.e1_acceptance.resolve();started=a.e1_acceptance_started.resolve();terminal=a.e1_acceptance_terminal.resolve()
    check_e1(proof,started,terminal)
    worker=HERE/'smoke_worker.py'
    approvals=check_approval()
    if not worker.is_file():raise RuntimeError('smoke worker missing')
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('frozen pre-launch source changed: '+path)
    # Import the launcher helper only after validating it, its imported
    # diagnose.py, and the dynamically loaded runtime against the frozen plan.
    sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command, gpu_pids, identity
    if gpu_pids():raise RuntimeError('GPU already occupied')
    out=HERE/('smoke_'+a.run_id);out.mkdir(exist_ok=False)
    name='eadp-e4a-'+a.run_id+'-'+str(os.getpid())
    command=container_command(plan,[str(worker),'--execute','--e1-acceptance',str(proof),
        '--e1-acceptance-started',str(started),'--e1-acceptance-terminal',str(terminal),'--out',str(out)],True,name)
    # Preserve the stopped container for independent post-exit inspection.
    if command.count('--rm')!=1:raise ValueError('unexpected Docker cleanup policy')
    command.remove('--rm')
    controller=identity(os.getpid()); launch={'command':command,'controller':controller,'timeout_seconds':a.timeout,
        'e1_acceptance_sha256':sha(proof),'launcher_sha256':sha(__file__),'worker_sha256':sha(worker),
        'approval_bundle':approvals,'started_unix':time.time()}
    (out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    interrupted=[None]
    def stop(signum,_frame):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
            'client_exited_while_container_running':False,'container_identity':None}
    with (out/'launcher.log').open('x') as log:
        client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic()
        try:
            while client.poll() is None:
                if result['container_identity'] is None:
                    probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=5)
                    if probe.returncode==0:
                        state=json.loads(probe.stdout)[0];pid=state['State']['Pid']
                        if pid:result['container_identity']={'container_id':state['Id'],'image':state['Image'],'host_init':identity(pid)}
                if interrupted[0] is not None or time.monotonic()-begin>=a.timeout:
                    result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None
                    subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                    try:client.wait(timeout=30)
                    except subprocess.TimeoutExpired:client.kill();client.wait()
                    break
                time.sleep(1)
            result['returncode']=client.wait()
            probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode==0 and json.loads(probe.stdout)[0]['State']['Running']:
                result['client_exited_while_container_running']=True
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode!=0 or json.loads(probe.stdout)[0]['State']['Running']:
                raise RuntimeError('could not confirm container stopped')
            if result['container_identity'] is None or json.loads(probe.stdout)[0]['Id']!=result['container_identity']['container_id']:
                raise RuntimeError('container identity unavailable or changed')
            result['container_exit_code']=json.loads(probe.stdout)[0]['State']['ExitCode']
            if result['container_exit_code']!=result['returncode']:raise RuntimeError('Docker client/container exit mismatch')
            result['container_stopped_verified']=True
        finally:
            if not result['container_stopped_verified']:
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if client.poll() is None:client.kill();client.wait()
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception:result['gpu_pids_after']=None
            (out/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
    if (result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or
        result['client_exited_while_container_running'] or result['gpu_pids_after']!=[]):return 1
    return 0

if __name__=='__main__':sys.exit(main())
