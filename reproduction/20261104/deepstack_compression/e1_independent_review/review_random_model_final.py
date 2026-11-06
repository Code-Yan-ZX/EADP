"""Review the retained real no-GPU random-model preflight and launcher."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/random_smoke';I=B/'e1_independent_review';O=D/'model_preflight002';ROOT=B.parents[2]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main(out):
 gate=load('_independent_random_gate',D/'smoke_gate.py');plan=gate.plan();cpu=gate.cpu(plan);approval=gate.admit(True)
 s=read(D/'monitor_model_preflight002/started.json');t=read(D/'monitor_model_preflight002/terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']==s['process_group'] and (cwd/cmd[1]).resolve()==D/'model_launch.py' and cmd.count('--preflight-only')==1 and s['timeout_seconds']==2100 and s['grace_seconds']==90
 launch=read(O/'launch.json');ex=read(O/'exit.json');w=read(O/'worker_preflight.json');assert launch['approval_bundle']==approval and launch['worker_sha256']==sha(D/'model_worker.py') and launch['launcher_sha256']==sha(D/'model_launch.py') and launch['run_out']==str(O) and launch['controller']['pid']==s['worker_pid'] and ex['controller']==launch['controller']
 assert w['status']=='passed' and w['plan_sha256']==sha(D/'execution_plan009.json') and w['anchors']==3 and w['random_choices']==9 and w['cuda_initialized'] is False and w['model_loaded'] is False and w['GT_opened'] is False and w['predictions_opened'] is True
 assert ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['interrupted'] is None and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
 cid=ex['container_identity']['container_id'];assert len(cid)==64 and cid in ex['container_identity']['host_cgroup'] and ex['container_identity']['host_init']['pid']>0 and ex['container_identity']['host_init']['start_ticks']>0
 doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15,check=True).stdout)[0];command=launch['command'];idx=command.index('--entrypoint');assert doc['Id']==cid and doc['Image']==command[idx+2] and doc['State']['Running'] is False and doc['State']['Pid']==0 and doc['State']['ExitCode']==0 and doc['State']['OOMKilled'] is False and doc['HostConfig']['NetworkMode']=='none' and not doc['HostConfig']['Devices'] and not doc['HostConfig']['DeviceRequests'] and doc['Config']['Entrypoint']==[command[idx+1]] and doc['Config']['Cmd']==command[idx+3:]==[str(D/'model_worker.py'),'--out',str(O),'--preflight-only']
 def values(flag):return [command[i+1] for i,x in enumerate(command[:-1]) if x==flag]
 assert '--device' not in command and '--gpus' not in command and doc['Config']['User']==values('--user')[0] and doc['Config']['WorkingDir']==values('--workdir')[0]
 env=dict(x.split('=',1) for x in doc['Config']['Env'] if '=' in x)
 for item in values('--env'):
  k,v=item.split('=',1);assert env.get(k)==v
 assert env['CUDA_VISIBLE_DEVICES']==''
 mounts={}
 for x in values('--mount'):
  parts=x.split(',');v=dict(y.split('=',1) for y in parts if '=' in y);assert v['type']=='bind';mounts[v['dst']]=(v['src'],'readonly' not in parts)
 assert mounts=={m['Destination']:(m['Source'],m['RW']) for m in doc['Mounts'] if m['Type']=='bind'}
 source=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';runtime=load('_independent_random_runtime',source/'run_panel.py');ref=runtime.container_command(read(source/'run_plan.json'),[str(D/'model_worker.py'),'--out',str(O),'--preflight-only'],False,launch['container_name']);assert ref.count('--rm')==1;ref.remove('--rm');assert ref==command
 p={'status':'passed','independent_review':True,'scope':'GPU diagnostic prelaunch','GT_opened':False,'scoring_performed':False,'predictions_opened':True,'plan_sha256':sha(D/'execution_plan009.json'),'reviewer_sha256':sha(__file__),'source_sha256':plan['source_sha256'],**cpu,'preflight_started_sha256':sha(D/'monitor_model_preflight002/started.json'),'preflight_terminal_sha256':sha(D/'monitor_model_preflight002/terminal.json'),'worker_preflight_sha256':sha(O/'worker_preflight.json'),'preflight_exit_sha256':sha(O/'exit.json'),'preflight_launch_sha256':sha(O/'launch.json'),'preflight_review_sha256':sha(I/'random_model_preflight_review.json'),'errors':[],'limitations':['GPU is not started or authorized by this review; root schedules any model diagnostic.','Existing E1 DEV mixed prediction journal parsed for input identity; no GT.','EADP greedy selection is computed and discarded; no timing claim.','Runtime per-step KV represented by fingerprints, not archived full tensors.']}
 with out.open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'random_model_final_review.json' and not a.out.exists();main(a.out)
