#!/usr/bin/env python3
"""Independent verification of real same-image no-device runtime-import probe."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_runtime_preflight_v2';R=D/'run001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (cwd/cmd[1]).resolve()==script.resolve()
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out.resolve();assert out==I/'rres_k256_runtime_v2_preflight_result_review.json' and not out.exists()
 term(D/'monitor_run001',D/'launch.py',R);term(I/'monitor_rres_k256_runtime_v2_preflight',I/'review_rres_k256_runtime_v2_preflight.py',I/'rres_k256_runtime_v2_preflight_review.json')
 p=read(D/'plan001.json');q=read(I/'rres_k256_runtime_v2_preflight_review.json');w=read(R/'worker.json');l=read(R/'launch.json');e=read(R/'exit.json');cid=read(R/'container_identity.json')
 assert q['status']=='passed' and q['independent_review'] is True and q['plan_sha256']==sha(D/'plan001.json') and q['source_sha256']==p['source_sha256'] and all(sha(k)==v for k,v in p['source_sha256'].items())
 assert w['status']=='passed' and w['GPU_locked'] is True and w['GPU_authorized'] is False and w['GT_opened'] is False and w['model_loaded'] is False and w['cuda_initialized'] is False and w['imports']==['worker','gpu_gate','launch','gpu_dispatch','arm','capture','resolution','decode_observer','author'] and not w['errors']
 assert w['approval']==l['approval'] and w['container_id']==cid['container_id'] and e['returncode']==0 and e['container_exit_code']==0 and e['container_stopped_verified'] is True and e['failure_type'] is None
 cmd=l['command'];o=json.loads(subprocess.run(['docker','inspect',l['name']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=o['State'];host=o['HostConfig'];cfg=o['Config']
 assert o['Id']==cid['container_id'] and o['Image']==cid['image'] and st['Running'] is False and st['Pid']==0 and st['ExitCode']==0 and st['OOMKilled'] is False and host['NetworkMode']=='none' and not (host.get('Devices') or []) and host.get('DeviceRequests') in (None,[])
 assert cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Cmd']==cmd[cmd.index(cfg['Image'])+1:]
 mounts=[]
 for i,x in enumerate(cmd[:-1]):
  if x=='--env':assert [y for y in cfg.get('Env',[]) if y.startswith(cmd[i+1].split('=',1)[0]+'=')]==[cmd[i+1]]
  if x=='--mount':
   v=cmd[i+1].split(',');d=dict(z.split('=',1) for z in v if '=' in z);mounts.append((str(Path(d['src']).resolve()),d['dst'],'readonly' not in v))
 actual=o.get('Mounts') or [];bind=[(str(Path(v['Source']).resolve()),v['Destination'],v['RW']) for v in actual if v.get('Type')=='bind'];assert len(actual)==len(mounts)==12 and sorted(bind)==sorted(mounts)
 v={'status':'passed','independent_review':True,'GPU_authorized':False,'GT_opened':False,'errors':[],'worker_sha256':sha(R/'worker.json'),'reviewer_sha256':sha(__file__),'plan_sha256':sha(D/'plan001.json'),'launch_sha256':sha(R/'launch.json'),'identity_sha256':sha(R/'container_identity.json'),'exit_sha256':sha(R/'exit.json'),'started_sha256':sha(D/'monitor_run001/started.json'),'terminal_sha256':sha(D/'monitor_run001/terminal.json'),'preflight_review_sha256':sha(I/'rres_k256_runtime_v2_preflight_review.json'),'checks':['211 source hashes and completed supervisors','real retained container image/process/env/12 mounts/no device/exit','nine actual imports; no model/CUDA, GPU lock active'],'limitations':['no actual GPU execution or full-model DeepStack evidence']}
 with out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
