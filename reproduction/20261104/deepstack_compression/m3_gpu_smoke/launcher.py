"""Hard-bounded no-GT real-model gradient feasibility launcher."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4'
sys.path.insert(1,str(E4))
from identity_worker import PLAN
from local_imports import local
sha=local('contract').sha;check_approval=local('gate').check_approval

def check_budget(run_id, timeout):
    budget=json.loads((HERE/'plan.json').read_text())
    initial=HERE.parent/'initial_plan.json'
    if (not run_id.isalnum() or timeout!=budget['inner_timeout_seconds'] or sha(initial)!=budget['budget_source_sha256'] or
        json.loads(initial.read_text()).get('budget_end_unix')!=budget['budget_end_unix'] or
        time.time()+budget['outer_timeout_seconds']>budget['budget_end_unix']):
        raise ValueError('invalid or exceeded B time budget')
    started=HERE/('monitor_smoke'+run_id)/'started.json'
    v=json.loads(started.read_text());expected=(HERE/('smoke_'+run_id)/'events.jsonl').resolve()
    if (v.get('worker_pid')!=os.getpid() or v.get('timeout_seconds')!=budget['outer_timeout_seconds'] or
        v.get('stale_seconds')!=budget['progress_stale_seconds'] or v.get('progress_paths')!=[str(expected)] or
        v.get('snapshot_seconds')!=budget['snapshot_seconds'] or v.get('poll_seconds')!=budget['poll_seconds'] or
        v.get('grace_seconds')!=budget['grace_seconds']):raise ValueError('B supervisor configuration mismatch')
    return budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--timeout',type=int,required=True)
    a=p.parse_args()
    budget=check_budget(a.run_id,a.timeout)
    approval=check_approval();host_upstream=local('gate').upstream(host=True);plan=json.loads(PLAN.read_text())
    if shutil.disk_usage(HERE).free<budget['minimum_free_bytes']:raise ValueError('insufficient free disk space')
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('pre-launch source changed: '+path)
    sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command,gpu_pids,identity
    if gpu_pids():raise ValueError('GPU occupied')
    out=HERE/('smoke_'+a.run_id);out.mkdir(exist_ok=False)
    receipt=out/'host_upstream.json'
    with receipt.open('x') as f:json.dump(host_upstream,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    receipt_link={'path':str(receipt.resolve()),'sha256':sha(receipt)}
    name='eadp-m3-'+a.run_id+'-'+str(os.getpid());worker=HERE/'worker.py';args=[str(worker),'--execute','--out',str(out)]
    command=container_command(plan,args,True,name)
    if command.count('--rm')!=1:raise ValueError('unexpected container cleanup policy')
    command.remove('--rm')
    if time.time()+budget['outer_timeout_seconds']>budget['budget_end_unix']:raise ValueError('B deadline expired before launch')
    controller=identity(os.getpid());launch={'command':command,'controller':controller,'timeout_seconds':a.timeout,'host_upstream':receipt_link,
        'approval_bundle':approval,'launcher_sha256':sha(__file__),'worker_sha256':sha(worker),'started_unix':time.time()}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    interrupted=[None]
    def stop(signum,_frame):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result=supervise(command,name,out,controller,a.timeout,gpu_pids,identity,interrupted)
    if (result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or result['errors'] or
        not result['container_stopped_verified'] or result['client_exited_while_container_running'] or result['gpu_pids_after']!=[]):return 1
    return 0

def supervise(command,name,out,controller,timeout,gpu_pids,identity,interrupted):
    """Preserve exit evidence even if Docker/client operations fail; never assume unknown state is stopped."""
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,
            'client_exited_while_container_running':False,'container_identity':None,'errors':[]}
    errors=result['errors'];client=None;state=None
    def note(stage,exc):errors.append(stage+': '+str(exc))
    def inspect(allow_missing=False):
        try:
            q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if q.returncode:
                if not allow_missing:note('inspect',f'exit {q.returncode}')
                return None
            values=json.loads(q.stdout)
            if len(values)!=1:raise ValueError('unexpected inspect result')
            return values[0]
        except Exception as e:note('inspect',e);return None
    def stop_and_kill(log):
        try:
            q=subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if q.returncode:note('stop',f'exit {q.returncode}')
        except Exception as e:note('stop',e)
        v=inspect()
        if v is None or v.get('State',{}).get('Running') is not False:
            try:
                q=subprocess.run(['docker','kill',name],stdout=log,stderr=log,timeout=15,check=False)
                if q.returncode:note('kill',f'exit {q.returncode}')
            except Exception as e:note('kill',e)
            v=inspect()
        return v
    with (out/'launcher.log').open('x') as log:
        try:
            client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
            while client.poll() is None:
                if result['container_identity'] is None:
                    v=inspect(allow_missing=True)
                    if v is not None:
                        pid=v['State']['Pid']
                        if pid:result['container_identity']={'container_id':v['Id'],'image':v['Image'],'host_init':identity(pid)}
                if interrupted[0] is not None or time.monotonic()-begin>=timeout:
                    result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None
                    state=stop_and_kill(log);break
                time.sleep(1)
            try:result['returncode']=client.wait(timeout=30)
            except Exception as e:note('client wait',e)
            if state is None:state=inspect()
            if state is not None and state.get('State',{}).get('Running') is True:
                result['client_exited_while_container_running']=True;state=stop_and_kill(log)
        except BaseException as e:note('supervision',repr(e))
        finally:
            if state is None or state.get('State',{}).get('Running') is not False:state=stop_and_kill(log)
            if client is not None:
                try:
                    if client.poll() is None:client.kill()
                    result['returncode']=client.wait(timeout=10)
                except Exception as e:note('client cleanup',e)
            try:
                ident=result['container_identity']
                if state is None or state.get('State',{}).get('Running') is not False:raise ValueError('container stop not confirmed')
                if ident is None or state.get('Id')!=ident['container_id']:raise ValueError('container identity mismatch')
                result['container_exit_code']=state['State']['ExitCode']
                if result['container_exit_code']!=result['returncode']:raise ValueError('client/container exit mismatch')
                result['container_stopped_verified']=True
            except Exception as e:note('termination',e)
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception as e:result['gpu_pids_after']=None;note('gpu snapshot',e)
            with (out/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return result

if __name__=='__main__':sys.exit(main())
