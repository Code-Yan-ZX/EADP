"""Independent review of execution-layer no-device import preflight."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_wavefront_model_t020_execution'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 run=D/'run_preflight001';plan=read(D/'preflight_plan002.json');w=read(run/'worker_preflight.json');s=read(D/'monitor_preflight001/started.json');t=read(D/'monitor_preflight001/terminal.json');launch=read(run/'launch.json');ex=read(run/'exit.json');ident=read(run/'container_identity.json')
 if (t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or s['worker_pid']!=launch['controller']['pid'] or (Path(s['cwd'])/s['command'][1]).resolve()!=D/'preflight_launch.py'):raise ValueError('supervisor')
 if (ex['returncode']!=0 or ex['container_exit_code']!=0 or ex['container_stopped_verified'] is not True or ex['timed_out'] is not False or ex['container_identity']!=ident or ex['controller']!=launch['controller'] or ex['last_inspect']['host_devices']!=[] or ex['last_inspect']['device_requests'] is not None):raise ValueError('exit or device')
 obj=json.loads(subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];state=obj['State'];host=obj['HostConfig']
 if obj['Id']!=ident['container_id'] or obj['Image']!=ident['image'] or state['Running'] or state['Pid']!=0 or state['ExitCode']!=0 or state['OOMKilled'] or host['NetworkMode']!='none' or host['Devices'] not in (None,[]) or host['DeviceRequests'] not in (None,[]):raise ValueError('actual docker state')
 if 'CUDA_VISIBLE_DEVICES=' not in obj['Config']['Env'] or obj['Config']['Cmd']!=[str(D/'preflight_entry.py'),'--out',str(run/'worker_preflight.json')]:raise ValueError('container command')
 if w['status']!='passed' or w['gpu_authorized'] is not False or w['cuda_initialized'] is not False or w['model_loaded'] is not False or w['GT_opened'] is not False or w['existing_DEV_predictions_opened'] is not True or w['approval']!=launch['approval'] or w['approval']['plan_sha256']!=sha(D/'preflight_plan002.json') or w['approval']['review_sha256']!=sha(D/'independent_preflight_review.json') or w['upstream_review_sha256']!=plan['upstream_review_sha256']:raise ValueError('worker scope or approval')
 expected=('gpu_admission','worker_entry','single_arm','controller_draft','bounded_runner_draft','schedule_contract','upstream_identity')
 if w['module_paths']!={x:str(D/(x+'.py')) for x in expected} or any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('source mismatch')
 result={'status':'passed','independent_review':True,'scope':'execution import no-GPU preflight only','gpu_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'plan_sha256':sha(D/'preflight_plan002.json'),'worker_preflight_sha256':sha(run/'worker_preflight.json'),'launch_sha256':sha(run/'launch.json'),'exit_sha256':sha(run/'exit.json'),'started_sha256':sha(D/'monitor_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_preflight001/terminal.json'),'source_sha256':plan['source_sha256'],'reviewer_sha256':sha(__file__),'errors':[],'limitations':['No real model or GPU identity run is authorized.','Reviewer authored wavefront candidate but did not author this execution preflight.','Prior DEV evidence was hashed for identity.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
