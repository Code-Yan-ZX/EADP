"""Hard-bounded host Docker launcher; fails closed without reviewed approval."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
from local_imports import load_local
gate=load_local('gate');HERE=gate.HERE;BASE=gate.BASE;check_approval=gate.check_approval;sha=gate.sha
sys.path.insert(0,str(BASE/'e4'))
from identity_worker import PLAN

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    for name in ('e1-acceptance','e1-started','e1-terminal'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.run_id.isalnum():raise ValueError('invalid run id')
    approval=check_approval()
    budget=json.loads((BASE/'initial_plan.json').read_text())
    if time.time()+3720>budget['budget_end_unix']:raise ValueError('three-day deadline')
    out=HERE/('run_'+a.run_id);started=json.loads((HERE/('monitor_'+a.run_id)/'started.json').read_text())
    if (started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=3720 or started.get('stale_seconds')!=1800 or
        started.get('snapshot_seconds')!=3600 or started.get('progress_paths')!=[str(out/'generated.jsonl')]):raise ValueError('bounded supervisor required')
    if shutil.disk_usage(HERE).free<30*1024**3:raise ValueError('insufficient disk')
    plan=json.loads(PLAN.read_text())
    for group in ('source_sha256','implementation_sha256'):
        for path,digest in plan[group].items():
            if sha(path)!=digest:raise ValueError('upstream source changed')
    sys.path.insert(0,str(PLAN.parent))
    from run_panel import container_command,gpu_pids,identity
    if gpu_pids():raise ValueError('GPU occupied')
    out.mkdir(exist_ok=False);worker=HERE/'smoke_worker.py';name='eadp-e2e3-'+a.run_id+'-'+str(os.getpid())
    args=[str(worker),'--execute','--e1-acceptance',str(a.e1_acceptance.resolve()),'--e1-started',str(a.e1_started.resolve()),'--e1-terminal',str(a.e1_terminal.resolve()),'--out',str(out)]
    command=container_command(plan,args,True,name)
    if command.count('--rm')!=1:raise ValueError('container removal policy')
    command.remove('--rm');controller=identity(os.getpid())
    launch={'command':command,'controller':controller,'approval':approval,'worker_sha256':sha(worker),'launcher_sha256':sha(__file__),'started_unix':time.time()}
    if launch['started_unix']+3720>budget['budget_end_unix']:raise ValueError('three-day deadline')
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    interrupted=[None]
    def stop(sig,frame):interrupted[0]=sig
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result={'returncode':None,'timed_out':False,'interrupted':None,'container_stopped_verified':False,'client_exited_while_container_running':False,'container_identity':None}
    with (out/'launcher.log').open('x') as log:
        client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);begin=time.monotonic()
        try:
            while client.poll() is None:
                if result['container_identity'] is None:
                    p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=5)
                    if p.returncode==0:
                        state=json.loads(p.stdout)[0];pid=state['State']['Pid']
                        if pid:result['container_identity']={'container_id':state['Id'],'image':state['Image'],'host_init':identity(pid)}
                if interrupted[0] is not None or time.monotonic()-begin>=3600:
                    result['interrupted']=interrupted[0];result['timed_out']=interrupted[0] is None
                    subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                    try:client.wait(timeout=30)
                    except subprocess.TimeoutExpired:client.kill();client.wait()
                    break
                time.sleep(1)
            result['returncode']=client.wait();p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if p.returncode==0 and json.loads(p.stdout)[0]['State']['Running']:
                result['client_exited_while_container_running']=True
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
            if p.returncode!=0:raise RuntimeError('container inspection failed')
            state=json.loads(p.stdout)[0]
            if state['State']['Running'] or result['container_identity'] is None or state['Id']!=result['container_identity']['container_id']:raise RuntimeError('container stop/identity missing')
            result['container_exit_code']=state['State']['ExitCode']
            if result['container_exit_code']!=result['returncode']:raise RuntimeError('exit mismatch')
            result['container_stopped_verified']=True
        finally:
            if not result['container_stopped_verified']:subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
            if client.poll() is None:client.kill();client.wait()
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception:result['gpu_pids_after']=None
            with (out/'exit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return int(result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or result['client_exited_while_container_running'] or result['gpu_pids_after']!=[])
if __name__=='__main__':sys.exit(main())
