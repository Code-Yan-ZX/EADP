"""Host-supervised no-GPU container preflight; bounded Docker cleanup."""
import argparse,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
from cpu_validation import sha
import runtime_gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SMOKE=ROOT/'m3_gpu_smoke'
sys.path.insert(0,str(SMOKE))
spec=importlib.util.spec_from_file_location('m3_oracle_exec_safe_docker',SMOKE/'launcher.py')
safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)

def clean_exit(result):
    return (result.get('returncode')==0 and result.get('container_exit_code')==0 and result.get('timed_out') is False and
            result.get('interrupted') is None and result.get('container_stopped_verified') is True and
            result.get('client_exited_while_container_running') is False and result.get('errors')==[])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--cpu-id',required=True);a=ap.parse_args()
    if not a.run_id.isalnum() or not a.cpu_id.isalnum():raise ValueError('invalid run id')
    proof=HERE/('cpu_check_'+a.cpu_id)/'verification.json';v=json.loads(proof.read_text());term=json.loads((HERE/('monitor_cpu'+a.cpu_id)/'terminal.json').read_text())
    if v['status']!='passed' or v['source_sha256_before']!=v['source_sha256_after'] or term['reason']!='completed' or term['worker_returncode']!=0 or term['remaining_process_group']!=[]:
        raise ValueError('synthetic CPU proof not completed')
    if any(sha(p)!=d for p,d in v['source_sha256_after'].items()):raise ValueError('preflight source changed')
    approval=runtime_gate.validate(host=False,preflight_only=True)
    if approval['cpu_sha256']!=sha(proof):raise ValueError('provisional approval binds different CPU proof')
    previous=json.loads((ROOT/'m3_gpu_smoke/preflight_001/launch.json').read_text());cmd=previous['command'][:]
    name='eadp-m3-oracle-preflight-'+a.run_id;cmd[cmd.index('--name')+1]=name
    i=cmd.index(str(ROOT/'m3_gpu_smoke/worker.py'));out=HERE/('preflight_'+a.run_id)
    cmd=cmd[:i]+[str(HERE/'runtime_worker.py'),'--execute','--preflight-only','--out',str(out)]
    if any('/dev/nvidia' in str(x) for x in cmd) or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('GPU exposed')
    out.mkdir(exist_ok=False);receipt=out/'host_receipt.json'
    with receipt.open('x') as f:json.dump(approval,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    launch={'command':cmd,'cpu_proof_sha256':sha(proof),'worker_sha256':sha(HERE/'runtime_worker.py'),
            'host_receipt':{'path':str(receipt.resolve()),'sha256':sha(receipt)}}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    sys.path.insert(0,str(safe.PLAN.parent))
    from run_panel import identity
    interrupted=[None]
    def stop(signum,_):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    start=time.time();r=safe.supervise(cmd,name,out,identity(os.getpid()),600,lambda:[],identity,interrupted)
    result=None
    for line in reversed((out/'launcher.log').read_text().splitlines()):
        try:result=json.loads(line);break
        except ValueError:pass
    try:
        q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
        info=json.loads(q.stdout)[0] if q.returncode==0 else {};state=info.get('State',{})
    except Exception:info={};state={}
    if interrupted[0] is not None:r['interrupted']=interrupted[0]
    ok=(clean_exit(r) and state.get('Running') is False and state.get('ExitCode')==0 and state.get('Pid')==0 and
        info.get('Id')==(r.get('container_identity') or {}).get('container_id') and result and result.get('status')=='passed' and
        result.get('cuda_initialized') is False and result.get('model_loaded') is False and
        result.get('GT_opened') is False and result.get('predictions_opened') is False)
    d={'status':'passed' if ok else 'failed','exit':r,'container_id':info.get('Id'),'container_running':state.get('Running'),
       'container_pid':state.get('Pid'),'container_exit_code':state.get('ExitCode'),'launch_sha256':sha(out/'launch.json'),
       'log_sha256':sha(out/'launcher.log'),'exit_sha256':sha(out/'exit.json'),'receipt_sha256':sha(receipt),'cpu_proof_sha256':sha(proof),
       'preflight_sha256':sha(__file__),'result':result,'started_unix':start,'finished_unix':time.time(),
       'scope':'no GPU, no model loading, no ground truth or predictions; not approval for real execution'}
    with (HERE/('preflight_'+a.run_id+'.json')).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
