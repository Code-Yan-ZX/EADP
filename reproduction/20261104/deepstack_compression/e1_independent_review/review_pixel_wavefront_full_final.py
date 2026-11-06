#!/usr/bin/env python3
"""Independent actual no-device container + final 384-panel GPU admission review."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/pixel_wavefront_t020_full_dev'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (Path(s['cwd'])/s['command'][1]).resolve()==script.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==D/'independent_final_review.json' and not out.exists() and not (D/'run_run001').exists()
 term(D/'monitor_freeze002',D/'freeze.py');term(D/'monitor_cpu002',D/'cpu_validation.py');term(D/'monitor_preflight001',D/'launch.py');term(I/'monitor_pixel_wavefront_full_preflight',I/'review_pixel_wavefront_full_preflight.py')
 sys.path.insert(0,str(D));import admission
 p=admission.check_plan();admission.check_review('preflight');c=read(D/'cpu_proof002.json');pre=read(D/'preflight001.json');la=read(D/'run_preflight001/launch.json');ex=read(D/'run_preflight001/exit.json')
 assert pre['status']=='passed' and pre['cuda_initialized'] is False and pre['model_loaded'] is False and pre['weight_sha256_verified']==4 and pre['GT_opened'] is False and pre['patch_before_model'] is True and len(pre['weight_stats'])==4
 assert {x['task'] for x in pre['processor_verified']}=={'DocVQA_VAL','TextVQA_VAL','AI2D_TEST'} and len(pre['processor_verified'])==3
 assert la['stage']=='preflight' and la['plan_sha256']==sha(D/'plan002.json') and ex['returncode']==0 and ex['container_stopped_verified'] is True and ex['container_exit_code']==0 and ex['failure_type'] is None and ex['timed_out'] is False and ex['controller']==la['controller']
 cmd=la['command'];obj=json.loads(subprocess.run(['docker','inspect',la['container_name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0];st=obj['State'];host=obj['HostConfig'];cfg=obj['Config']
 assert obj['Id']==ex['container_identity']['container_id'] and obj['Image']==ex['container_identity']['image'] and st['Running'] is False and st['Pid']==0 and st['ExitCode']==0 and st['OOMKilled'] is False and not (host.get('Devices') or []) and host.get('DeviceRequests') in (None,[]) and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
 image=cfg['Image'];assert cfg['Cmd']==cmd[cmd.index(image)+1:]
 for i,x in enumerate(cmd[:-1]):
  if x=='--env':assert [y for y in cfg.get('Env',[]) if y.startswith(cmd[i+1].split('=',1)[0]+'=')]==[cmd[i+1]]
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--mount':
   parts=cmd[i+1].split(',');d=dict(v.split('=',1) for v in parts if '=' in v);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in parts))
 actual=obj.get('Mounts') or [];bind=[(str(Path(m['Source']).resolve()),m['Destination'],m['RW']) for m in actual if m.get('Type')=='bind'];assert len(actual)==len(mounts)==12 and sorted(bind)==sorted(mounts)
 # Final stage must not create the GPU container before this proof exists.
 z=subprocess.run([sys.executable,str(D/'launch.py'),'--stage','run'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=120)
 assert z.returncode!=0 and 'monitor_pixel_wavefront_full_final' in z.stderr and not (D/'run_run001').exists(),z.stderr[-1000:]
 budget=read(B/'initial_plan.json');assert sha(B/'initial_plan.json')==p['budget_source_sha256'] and time.time()+p['outer_timeout_seconds']+p['grace_seconds']<budget['budget_end_unix']
 fields={'plan_sha256':D/'plan002.json','reviewer_sha256':Path(__file__),'cpu_proof_sha256':D/'cpu_proof002.json','cpu_started_sha256':D/'monitor_cpu002/started.json','cpu_terminal_sha256':D/'monitor_cpu002/terminal.json','preflight_sha256':D/'preflight001.json','preflight_exit_sha256':D/'run_preflight001/exit.json','preflight_launch_sha256':D/'run_preflight001/launch.json','preflight_started_sha256':D/'monitor_preflight001/started.json','preflight_terminal_sha256':D/'monitor_preflight001/terminal.json'}
 v={'status':'passed','independent_review':True,'scope':'final','errors':[],'GT_opened':False,'GPU_used':False,'checks':['real Docker exited/PID0/no device/non-OOM and exact image/env/12 mounts','three actual processor checks and four payload hashes','preflight source, baseline and candidate prior admission','GPU launch refuses before final review and respects absolute budget','formal verifier retains all 384 equality and mismatch rows'],'prediction_scope':'existing E1 DEV mixed journal parsed for identity; 768 accepted baseline DEV files opened for SHA only','limitations':['same DEV posthoc exploratory test','candidate selector and saved output/indices checked, full earlier high-dimensional tensors unavailable','not a latency experiment']};v.update({k:sha(x) for k,x in fields.items()})
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
