"""Independent actual no-GPU preflight audit for Div/CD full DEV panel."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'e6/selected_full'
sys.path.insert(0,str(HERE))
import cpu_validation,freeze_plan,gate
from full_imports import load
container_mount=load('review_selected_full_mount',ROOT/'e6/selected_smoke/container_mount.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 assert a.out.resolve()==HERE/'independent_final_review.json'
 cpu=HERE/'cpu_check_005/verification.json';cs=HERE/'monitor_cpu005/started.json';ct=HERE/'monitor_cpu005/terminal.json'
 previous=HERE/'independent_preflight_review.json';ps=ROOT/'e1_independent_review/monitor_selected_full_preflight/started.json';pt=ps.parent/'terminal.json'
 gate.supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);gate.supervised(ps,pt,ROOT/'e1_independent_review/review_selected_full_preflight.py',previous)
 c=read(cpu);p=read(previous);src={str(x):sha(x) for x in cpu_validation.sources()}
 assert c['status']=='passed' and c['source_sha256_before']==c['source_sha256_after']==src
 assert p['status']=='passed' and p['preflight_only'] is True and p['source_sha256']==src and p['reviewer_sha256']==sha(ROOT/'e1_independent_review/review_selected_full_preflight.py')
 assert p['cpu_proof_sha256']==sha(cpu) and p['cpu_started_sha256']==sha(cs) and p['cpu_terminal_sha256']==sha(ct)
 approval=gate.check(preflight_only=True,host=True);assert approval['cpu_sha256']==sha(cpu) and approval['review_sha256']==sha(previous)
 proof=HERE/'preflight_001.json';started=HERE/'monitor_preflight001/started.json';terminal=HERE/'monitor_preflight001/terminal.json';d=HERE/'preflight_001'
 s,t=read(started),read(terminal);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']==[(HERE/'preflight.py').resolve()] and cmd.count('--run-id')==1 and cmd[cmd.index('--run-id')+1]=='001'
 f=read(proof);launch=read(d/'launch.json');ex=read(d/'exit.json');receipt=read(d/'host_receipt.json')
 assert f['status']=='passed' and f['exit']==ex and receipt==approval and launch['cpu_proof_sha256']==sha(cpu) and launch['worker_sha256']==sha(HERE/'worker.py')
 for k,path in [('launch_sha256',d/'launch.json'),('exit_sha256',d/'exit.json'),('log_sha256',d/'launcher.log'),('receipt_sha256',d/'host_receipt.json'),('preflight_sha256',HERE/'preflight.py'),('cpu_proof_sha256',cpu)]:assert f[k]==sha(path)
 assert launch['host_receipt']=={'path':str((d/'host_receipt.json').resolve()),'sha256':sha(d/'host_receipt.json')}
 logs=(d/'launcher.log').read_text().splitlines();actual=next(json.loads(x) for x in reversed(logs) if x.startswith('{'))
 assert actual==f['result'] and actual['status']=='passed' and all(actual[k] is False for k in ('cuda_initialized','model_loaded','GT_opened','predictions_opened'))
 assert actual['inputs_verified']==384 and actual['weights_sha256_verified']==4 and actual['source_count_verified']==len(src) and actual['source_manifest_sha256']==approval['source_manifest_sha256']
 assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['timed_out'] is False and ex['interrupted'] is None and ex['errors']==[]
 docker=launch['command'];prior=read(ROOT/'m3_gpu_smoke/preflight_001/launch.json')['command'];expected=prior[:];expected[expected.index('--name')+1]='eadp-selected-full-preflight-001';i=expected.index(str(ROOT/'m3_gpu_smoke/worker.py'));expected=container_mount.add(expected[:i]+[str(HERE/'worker.py'),'--execute','--preflight-only','--out',str(d)])
 assert docker==expected and 'CUDA_VISIBLE_DEVICES=' in docker and not any('/dev/nvidia' in str(x) for x in docker)
 obj=json.loads(subprocess.check_output(['docker','inspect',f['container_id']],text=True))[0];state=obj['State'];cfg=obj['Config'];host=obj['HostConfig'];image=docker[docker.index('--entrypoint')+2]
 assert obj['Id']==ex['container_identity']['container_id']==f['container_id'] and obj['Image']==ex['container_identity']['image']==image
 assert state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False and not host['Devices'] and not host['DeviceRequests'] and host['NetworkMode']=='none'
 assert cfg['Entrypoint']==[docker[docker.index('--entrypoint')+1]] and cfg['Cmd']==docker[docker.index(image)+1:] and cfg['User']==docker[docker.index('--user')+1] and cfg['WorkingDir']==docker[docker.index('--workdir')+1]
 assert container_mount.verify_actual(docker,obj)==f['actual_mount_count_verified']==13
 envs=[docker[i+1] for i,x in enumerate(docker[:-1]) if x=='--env'];assert all(x in cfg['Env'] for x in envs)
 result={'status':'passed','independent_review':True,'preflight_only':False,'GT_opened':False,'gpu_used':False,'reviewer_sha256':sha(__file__),
         'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),'source_sha256':src,
         'preflight_sha256':sha(proof),'preflight_started_sha256':sha(started),'preflight_terminal_sha256':sha(terminal),
         'limitations':['No full-panel GPU output or accuracy has been evaluated.','Full panel retains only tensor summaries; prior fixed-anchor smoke archives full arrays.','GPU requires coordinator scheduling and clean exclusive device.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','sources':len(src),'mounts':13,'scope':'full DEV panel prelaunch'}))
if __name__=='__main__':main()
