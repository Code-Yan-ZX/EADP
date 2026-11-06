"""Draft B launcher: hard-bounded; the B approval gate remains locked."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4';NATIVE=HERE.parent/'e4_native'
sys.path[:0]=[str(E4),str(NATIVE)]
from native_gate import check_gates
from identity_worker import PLAN,sha
from b_imports import local
check_approval=local('b_gate').check_approval

def check_budget(run_id, timeout):
    budget=json.loads((HERE/'budget_plan.json').read_text())
    initial=HERE.parent/'initial_plan.json'
    if (not run_id.isalnum() or timeout!=budget['inner_timeout_seconds'] or sha(initial)!=budget['budget_source_sha256'] or
        json.loads(initial.read_text()).get('budget_end_unix')!=budget['budget_end_unix'] or
        time.time()+budget['outer_timeout_seconds']>budget['budget_end_unix']):
        raise ValueError('invalid or exceeded B time budget')
    started=HERE/('monitor_b'+run_id)/'started.json'
    v=json.loads(started.read_text());expected=(HERE/('b_'+run_id)/'generated.jsonl').resolve()
    if (v.get('worker_pid')!=os.getpid() or v.get('timeout_seconds')!=budget['outer_timeout_seconds'] or
        v.get('stale_seconds')!=budget['journal_stale_seconds'] or v.get('progress_paths')!=[str(expected)] or
        v.get('snapshot_seconds')!=budget['snapshot_seconds'] or v.get('poll_seconds')!=budget['poll_seconds'] or
        v.get('grace_seconds')!=budget['grace_seconds']):raise ValueError('B supervisor configuration mismatch')
    return budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--timeout',type=int,required=True)
    for name in ('e1-acceptance','e1-started','e1-terminal','smoke-dir','smoke-proof','smoke-started','smoke-terminal'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    budget=check_budget(a.run_id,a.timeout)
    approval=check_approval();host_panel=local('b_gate').host_validate_panel();check_gates(a);plan=json.loads(PLAN.read_text())
    if shutil.disk_usage(HERE).free<budget['minimum_free_bytes']:raise ValueError('insufficient free disk space')
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('pre-launch source changed: '+path)
    sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command,gpu_pids,identity
    if gpu_pids():raise ValueError('GPU occupied')
    out=HERE/('b_'+a.run_id);out.mkdir(exist_ok=False)
    receipt=out/'host_panel_gate.json'
    with receipt.open('x') as f:json.dump(host_panel,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    receipt_link={'path':str(receipt.resolve()),'sha256':sha(receipt)}
    name='eadp-b-'+a.run_id+'-'+str(os.getpid());worker=HERE/'b_worker.py';args=[str(worker),'--execute']
    for field in ('e1_acceptance','e1_started','e1_terminal','smoke_dir','smoke_proof','smoke_started','smoke_terminal'):
        args += ['--'+field.replace('_','-'),str(getattr(a,field).resolve())]
    args += ['--out',str(out)]
    command=container_command(plan,args,True,name)
    if command.count('--rm')!=1:raise ValueError('unexpected container cleanup policy')
    command.remove('--rm')
    if time.time()+budget['outer_timeout_seconds']>budget['budget_end_unix']:raise ValueError('B deadline expired before launch')
    controller=identity(os.getpid());launch={'command':command,'controller':controller,'timeout_seconds':a.timeout,'host_panel_gate':receipt_link,
        'approval_bundle':approval,'launcher_sha256':sha(__file__),'worker_sha256':sha(worker),'started_unix':time.time()}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    interrupted=[None]
    def stop(signum,_frame):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
            'client_exited_while_container_running':False,'container_identity':None}
    with (out/'launcher.log').open('x') as log:
        client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
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
            result['returncode']=client.wait();probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode==0 and json.loads(probe.stdout)[0]['State']['Running']:
                result['client_exited_while_container_running']=True
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                probe=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if probe.returncode!=0 or json.loads(probe.stdout)[0]['State']['Running']:raise RuntimeError('container stop not confirmed')
            if result['container_identity'] is None or json.loads(probe.stdout)[0]['Id']!=result['container_identity']['container_id']:
                raise RuntimeError('container identity mismatch')
            result['container_exit_code']=json.loads(probe.stdout)[0]['State']['ExitCode']
            if result['container_exit_code']!=result['returncode']:raise RuntimeError('client/container exit mismatch')
            result['container_stopped_verified']=True
        finally:
            if not result['container_stopped_verified']:
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if client.poll() is None:client.kill();client.wait()
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception:result['gpu_pids_after']=None
            with (out/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if (result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or
        result['client_exited_while_container_running'] or result['gpu_pids_after']!=[]):return 1
    return 0

if __name__=='__main__':sys.exit(main())
