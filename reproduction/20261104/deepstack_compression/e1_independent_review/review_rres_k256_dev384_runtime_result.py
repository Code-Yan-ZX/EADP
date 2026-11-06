"""Independent retained no-device 384 worker-import preflight review."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_k256_dev384_exec';F=B/'e6/rres_k256_dev384';P=B/'e6/rres_k256_pilot_v2';ROOT=B.parents[2];R=D/'runtime_preflight001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 assert type(s['worker_pid']) is int and s['worker_pid']>0 and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
 planfile=D/'runtime_preflight_plan001.json';p=read(planfile);review=I/'rres_k256_dev384_runtime_preflight_review.json';reviewer=I/'review_rres_k256_dev384_runtime_preflight.py';terminal(I/'monitor_rres_k256_dev384_runtime_preflight',reviewer,review);q=read(review);s=terminal(D/'monitor_runtime_preflight001',D/'runtime_preflight_launch.py',R)
 assert q['status']=='passed' and q['independent_review'] is True and q['preflight_only'] is True and q['GPU_authorized'] is False and q['GT_opened'] is False and q['errors']==[] and q['reviewer_sha256']==sha(reviewer) and q['source_sha256']==p['source_sha256'] and q['plan_sha256']==sha(planfile) and p['scope']=='rres384_runtime_import_noGPU' and p['GPU_authorized'] is False and all(sha(k)==v for k,v in p['source_sha256'].items())
 launch=read(R/'launch.json');identity=read(R/'container_identity.json');ex=read(R/'exit.json');worker=read(R/'worker_preflight.json')
 assert launch['approval']=={'plan_sha256':sha(planfile),'review_sha256':sha(review)} and launch['controller']['pid']==s['worker_pid']==ex['controller']['pid'] and ex['container_identity']==identity and ex['returncode']==ex['container_exit_code']==0 and ex['timed_out'] is False and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False
 assert worker['status']=='passed' and worker['approval']==launch['approval'] and worker['container_id']==identity['container_id'] and worker['GPU_authorized'] is False and worker['model_loaded'] is False and worker['cuda_initialized'] is False and worker['GT_opened'] is False and worker['existing_DEV_predictions_opened'] is True and worker['scoring_performed'] is False and worker['errors']==[] and worker['worker_sha256']==sha(D/'worker.py')
 expected={'arm':P/'arm.py','capture':P/'capture.py','resolution':P/'resolution.py','panel_core':F/'panel_core.py','decode_observer':B/'e4/decode_observer.py','input_preflight':B/'e4_c_latency/input_preflight.py','vlmeval.vlm.qwen3_vl.model_fixed_res':ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
 assert set(worker['module_paths'])==set(expected)
 for name,path in expected.items():assert Path(worker['module_paths'][name])==path.resolve() and p['source_sha256'][str(path.resolve())]==sha(path)
 obj=json.loads(subprocess.run(['docker','inspect',identity['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];assert obj['Id']==identity['container_id']==ex['last_inspect']['id'] and obj['Image']==identity['image'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['HostConfig']['NetworkMode']=='none' and not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests']
 cmd=launch['command'];assert '--gpus' not in cmd and '--device' not in cmd and obj['Config']['Cmd']==[str(D/'runtime_probe.py'),'--out',str(R/'worker_preflight.json')] and obj['Config']['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and obj['Config']['User']==cmd[cmd.index('--user')+1] and obj['Config']['WorkingDir']==cmd[cmd.index('--workdir')+1]
 mounts=[]
 for i,c in enumerate(cmd[:-1]):
  if c=='--env':assert cmd[i+1] in obj['Config']['Env']
  if c=='--mount':
   v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mounts.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 assert sorted(mounts)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind') and all(sha(k)==v for k,v in p['source_sha256'].items())
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'errors':[],'plan_sha256':sha(planfile),'worker_preflight_sha256':sha(R/'worker_preflight.json'),'launch_sha256':sha(R/'launch.json'),'identity_sha256':sha(R/'container_identity.json'),'exit_sha256':sha(R/'exit.json'),'started_sha256':sha(D/'monitor_runtime_preflight001/started.json'),'terminal_sha256':sha(D/'monitor_runtime_preflight001/terminal.json'),'preflight_review_sha256':sha(review),'reviewer_sha256':sha(__file__),'scope':'actual no-device worker/module imports; no model or GPU authorization','limitations':['worker dependency imports do not execute the actual model or generate','accepted DEV source artifacts hashed/parsed only for identity']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('384 noGPU runtime result review passed')
if __name__=='__main__':main()
