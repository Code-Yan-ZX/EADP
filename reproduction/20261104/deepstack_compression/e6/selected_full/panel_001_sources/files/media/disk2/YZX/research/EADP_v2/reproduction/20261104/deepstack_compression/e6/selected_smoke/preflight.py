"""Same worker model-before-load path in a bounded CPU-only real container."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from local_imports import HERE,ROOT,E4,pinned
from freeze_plan import sha
from docker_supervisor import supervise
mounts=pinned('eadp_selected_container_mount',HERE/'container_mount.py');add_audit_mount=mounts.add

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run id')
 gate=pinned('eadp_selected_gate',HERE/'gate.py');approval=gate.check(preflight_only=True,require_future=True)
 old=ROOT/'m3_gpu_smoke/preflight_001/launch.json';cmd=json.loads(old.read_text())['command'][:]
 name='eadp-selected-preflight-'+a.run_id;cmd[cmd.index('--name')+1]=name;index=cmd.index(str(ROOT/'m3_gpu_smoke/worker.py'))
 out=HERE/('preflight_'+a.run_id);cmd=add_audit_mount(cmd[:index]+[str(HERE/'worker.py'),'--execute','--preflight-only','--out',str(out)])
 if any('/dev/nvidia' in str(x) for x in cmd) or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('GPU exposed')
 out.mkdir(exist_ok=False);receipt=out/'host_receipt.json'
 with receipt.open('x') as f:json.dump(approval,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 launch={'command':cmd,'cpu_proof_sha256':approval['cpu_sha256'],'worker_sha256':sha(HERE/'worker.py'),
         'host_receipt':{'path':str(receipt.resolve()),'sha256':sha(receipt)},'base_launch_sha256':sha(old)}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 identity_worker=pinned('identity_worker',E4/'identity_worker.py');sys.path.insert(0,str(identity_worker.PLAN.parent))
 pinned('diagnose',identity_worker.PLAN.parent/'diagnose.py');run_panel=pinned('eadp_selected_run_panel',identity_worker.PLAN.parent/'run_panel.py')
 interrupted=[None]
 def stop(signum,_):interrupted[0]=signum
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 begin=time.time();ex=supervise(cmd,name,out,run_panel.identity(os.getpid()),600,lambda:[],run_panel.identity,interrupted)
 result=None
 for line in reversed((out/'launcher.log').read_text().splitlines()):
  try:result=json.loads(line);break
  except ValueError:pass
 try:
  q=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
  info=json.loads(q.stdout)[0] if q.returncode==0 else {};state=info.get('State',{})
 except Exception:info={};state={}
 try:mount_count=mounts.verify_actual(cmd,info)
 except Exception:mount_count=None
 ok=(ex.get('returncode')==0 and ex.get('container_exit_code')==0 and ex.get('timed_out') is False and ex.get('interrupted') is None and
     ex.get('container_stopped_verified') is True and ex.get('client_exited_while_container_running') is False and ex.get('errors')==[] and
     state.get('Running') is False and state.get('Pid')==0 and state.get('ExitCode')==0 and
     info.get('Id')==(ex.get('container_identity') or {}).get('container_id') and isinstance(result,dict) and result.get('status')=='passed' and
     all(result.get(k) is False for k in ('cuda_initialized','model_loaded','GT_opened','predictions_opened')) and
     result.get('anchors_verified')==3 and result.get('weights_sha256_verified')==4 and
     result.get('source_count_verified')==approval['source_count_verified'] and
     result.get('source_manifest_sha256')==approval['source_manifest_sha256'] and mount_count is not None)
 proof={'status':'passed' if ok else 'failed','exit':ex,'container_id':info.get('Id'),'container_running':state.get('Running'),
        'container_pid':state.get('Pid'),'container_exit_code':state.get('ExitCode'),'launch_sha256':sha(out/'launch.json'),
        'log_sha256':sha(out/'launcher.log'),'exit_sha256':sha(out/'exit.json'),'receipt_sha256':sha(receipt),
        'cpu_proof_sha256':approval['cpu_sha256'],'preflight_sha256':sha(__file__),'result':result,
        'actual_mount_count_verified':mount_count,'started_unix':begin,'finished_unix':time.time(),
        'scope':'no GPU, model load, GT or predictions; no GPU authorization'}
 with (HERE/('preflight_'+a.run_id+'.json')).open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
