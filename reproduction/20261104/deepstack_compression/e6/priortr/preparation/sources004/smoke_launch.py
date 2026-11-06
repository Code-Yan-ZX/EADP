"""Review-gated bounded Docker launcher for PriorTR single DEV smoke."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from admission import BASE,E1,admit,sha
from smoke_worker import admit_review
HERE=Path(__file__).resolve().parent
PLAN=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
PYTHON=HERE.parent/'install001/env/bin/python'
def exit_code(r):
    if (r.get('returncode')!=0 or r.get('container_exit_code')!=0 or r.get('timed_out') is not False or r.get('interrupted') is not None or r.get('client_exited_while_container_running') is not False or r.get('gpu_pids_after')!=[] or r.get('container_stopped_verified') is not True or not r.get('container_identity')):return 1
    return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--review-proof',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--timeout',type=int,required=True)
    a=p.parse_args()
    if not a.run_id.isalnum() or not 60 <= a.timeout <= 14400:raise ValueError('invalid id/deadline')
    proof=a.review_proof.resolve();admit();worker=HERE/'smoke_worker.py';approvals=admit_review(proof)
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
    cache=out/'runtime_cache';cache.mkdir()
    name='eadp-e6prior-'+a.run_id+'-'+str(os.getpid())
    command=container_command(plan,[str(worker),'--execute','--review-proof',str(proof),'--out',str(out)],True,name)
    idx=command.index('--entrypoint');command[idx+1]=str(PYTHON)
    overrides={'PYTHONPATH':str(HERE),'HF_HOME':str(cache/'hf'),'XDG_CACHE_HOME':str(cache/'xdg'),'MPLCONFIGDIR':str(cache/'mpl')}
    for i,x in enumerate(command):
        key=x.split('=',1)[0]
        if key in overrides:command[i]=key+'='+overrides[key]
    at=command.index('--workdir');command[at:at]=['--env','TMPDIR='+str(cache),'--env','TORCH_EXTENSIONS_DIR='+str(cache/'extensions')]
    # Preserve the stopped container for independent post-exit inspection.
    if command.count('--rm')!=1:raise ValueError('unexpected Docker cleanup policy')
    command.remove('--rm')
    controller=identity(os.getpid()); launch={'command':command,'controller':controller,'timeout_seconds':a.timeout,
        'review_proof_sha256':sha(proof),'launcher_sha256':sha(__file__),'worker_sha256':sha(worker),
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
                result['cleanup_error']=None
                try:
                    stopped=subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=log,timeout=30,check=False)
                    if stopped.returncode!=0:raise RuntimeError('docker stop failed')
                except Exception as exc:
                    result['cleanup_error']=repr(exc)
                    try:subprocess.run(['docker','kill',name],stdout=log,stderr=log,timeout=15,check=False)
                    except Exception as kill_exc:result['cleanup_error']+='; kill: '+repr(kill_exc)
            if client.poll() is None:client.kill();client.wait()
            result.update(controller=controller,name=name,finished_unix=time.time())
            if interrupted[0] is not None:result['interrupted']=interrupted[0]
            try:result['gpu_pids_after']=gpu_pids()
            except Exception:result['gpu_pids_after']=None
            (out/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
    return exit_code(result)

if __name__=='__main__':sys.exit(main())
