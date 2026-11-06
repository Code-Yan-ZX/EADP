"""Same-image no-GPU complete worker preflight; requires preflight review."""
import importlib.util,json,os,signal,sys,time
from pathlib import Path
import gate
from launch import configure
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PREP=HERE.parent/'preparation'

def lifecycle():
 root=BASE/'e2_e3_full';sys.path.insert(0,str(root));path=root/'preflight.py'
 spec=importlib.util.spec_from_file_location('_priortr_shared_preflight',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 if Path(mod.__file__).resolve()!=path.resolve():raise ImportError('wrong preflight lifecycle')
 return mod.supervise_container

def main():
 approval=gate.execution_gate(preflight=True);out=HERE/'preflight_003';out.mkdir(exist_ok=False);cache=out/'runtime_cache';cache.mkdir()
 prior=PREP/'smoke_001/launch.json';cmd=json.loads(prior.read_text())['command'][:];name='eadp-priortr32-preflight-003'
 if cmd.count('--name')!=1:raise ValueError('container name')
 cmd[cmd.index('--name')+1]=name
 for device in ('/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
  i=cmd.index(device);del cmd[i-1:i+1]
 cmd[cmd.index('CUDA_VISIBLE_DEVICES=0')]='CUDA_VISIBLE_DEVICES='
 worker=str(PREP/'smoke_worker.py');idx=cmd.index(worker);cmd=cmd[:idx]+[str(HERE/'panel_worker.py'),'--execute','--preflight-only','--out',str(out)]
 cmd=configure(cmd,cache)
 launch={'command':cmd,'prior_launch_sha256':gate.sha(prior),'approval':approval}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def handle(sig,_frame):interrupted[0]=sig
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,handle)
 start=time.time();result=lifecycle()(cmd,name,out,interrupted,timeout=1200)
 try:output=json.loads((out/'stdout.log').read_text().strip().splitlines()[-1])
 except Exception:output=None
 passed=(not result['timed_out'] and result['interrupted'] is None and not result['errors'] and result['returncode']==0 and result['container_stopped_verified'] and result['container_exit_code']==0 and not result['client_exited_while_container_running'] and isinstance(output,dict) and output.get('status')=='preflight_passed' and output.get('cuda_initialized') is False and output.get('model_loaded') is False and output.get('predictions_parsed') is False and output.get('labels_opened') is False and not (out/'frozen.json').exists() and not (out/'data').exists())
 proof={'status':'passed' if passed else 'failed',**result,'worker_sha256':gate.sha(HERE/'panel_worker.py'),'preflight_source_sha256':gate.sha(__file__),'launch_sha256':gate.sha(out/'launch.json'),'stdout_sha256':gate.sha(out/'stdout.log'),'stderr_sha256':gate.sha(out/'stderr.log'),'result':output,'started_unix':start,'finished_unix':time.time()}
 with (HERE/'container_preflight_003.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':proof['status']}),flush=True);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
