"""Review retained no-GPU random baseline import/weight preflight only."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
B=Path(__file__).resolve().parents[1];D=B/'e6/random_smoke';I=B/'e1_independent_review';RUN=D/'run_preflight002';OUT=I/'random_smoke_preflight002_result_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script,out):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervisor')
 return s
def run():
 proof=I/'random_smoke_preflight_review_002.json';reviewer=I/'review_random_smoke_preflight_002.py';rs=I/'monitor_random_smoke_preflight_002';term(rs,reviewer,proof);r=read(proof);plan=D/'runtime_plan003.json';p=read(plan)
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or r.get('predictions_opened') is not True or r.get('GT_opened') is not False or r.get('GPU_used') is not False or r.get('runtime_plan_sha256')!=sha(plan) or r.get('reviewer_sha256')!=sha(reviewer) or r.get('review_started_sha256')!=sha(rs/'started.json') or r.get('source_sha256')!=p['source_sha256']:raise ValueError('prior independent review')
 for path,h in p['source_sha256'].items():
  if sha(path)!=h:raise ValueError('source mismatch '+path)
 mon=D/'monitor_preflight002';s=term(mon,D/'preflight_launch.py',RUN);launch=read(RUN/'launch.json');ex=read(RUN/'exit.json');worker=read(RUN/'worker_preflight.json')
 if s['timeout_seconds']!=1500 or s['grace_seconds']!=90 or launch['controller']!=ex['controller'] or launch['controller']['pid']!=s['worker_pid'] or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False:raise ValueError('run exit')
 if worker.get('status')!='passed' or worker.get('GT_opened') is not False or worker.get('predictions_opened') is not True or worker.get('prediction_scope')!=p['prediction_scope'] or worker.get('model_loaded') is not False or worker.get('cuda_initialized') is not False or worker.get('weight_count')!=4 or worker.get('runtime_plan_sha256')!=sha(plan):raise ValueError('worker proof')
 w=read(B.parents[2]/'reproduction/20261028/qwen/qwen_weight_identity.json')['files']
 if len(w)!=4 or any(a['sha256']!=b['expected_lfs_sha256'] or a['bytes']!=b['bytes'] for a,b in zip(worker['weights'],w)):raise ValueError('four weight identities')
 cid=ex['container_identity']['container_id'];obj=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=10,check=True).stdout)[0];cfg=obj['Config'];host=obj['HostConfig'];st=obj['State'];cmd=launch['command'];img=obj['Image']
 if obj['Id']!=cid or obj['Name']!='/'+launch['name'] or st['Running'] or st['Pid']!=0 or st['ExitCode']!=0 or st['OOMKilled'] or cfg['Image']!=img or img!=ex['container_identity']['image'] or host.get('Devices') or host.get('DeviceRequests') or cfg['Entrypoint']!=[cmd[cmd.index('--entrypoint')+1]] or cfg['Cmd']!=cmd[cmd.index(img)+1:] or cfg['User']!=cmd[cmd.index('--user')+1] or cfg['WorkingDir']!=cmd[cmd.index('--workdir')+1] or host['NetworkMode']!=cmd[cmd.index('--network')+1]:raise ValueError('actual docker identity')
 for j,v in enumerate(cmd[:-1]):
  if v=='--env':
   key=cmd[j+1].split('=',1)[0]
   if [x for x in cfg['Env'] if x.startswith(key+'=')]!=[cmd[j+1]]:raise ValueError('env mismatch')
 mounts=[]
 for j,v in enumerate(cmd[:-1]):
  if v=='--mount':
   chunks=cmd[j+1].split(',');d=dict(x.split('=',1) for x in chunks if '=' in x);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in chunks))
 actual=[(str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind']
 if len(actual)!=len(obj['Mounts']) or sorted(actual)!=sorted(mounts):raise ValueError('actual mounts')
 return {'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'predictions_opened':True,'prediction_scope':p['prediction_scope'],'GPU_used':False,'model_loaded':False,'weights_verified':4,'runtime_plan_sha256':sha(plan),'preflight_review_sha256':sha(proof),'started_sha256':sha(mon/'started.json'),'terminal_sha256':sha(mon/'terminal.json'),'worker_sha256':sha(RUN/'worker_preflight.json'),'launch_sha256':sha(RUN/'launch.json'),'exit_sha256':sha(RUN/'exit.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Same-image import/four-weight preflight only; no actual random model or GPU identity, no latency result.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('unexpected output')
 v=run()
 with a.out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
