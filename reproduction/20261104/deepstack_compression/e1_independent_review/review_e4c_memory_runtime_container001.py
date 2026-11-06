"""Independent retained-container CPU preflight verification; no GPU authority."""
import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e4_c_memory_runtime';I=R/'e1_independent_review';D=H/'container_preflight001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
s=mon(H/'monitor_container_preflight001');mon(I/'monitor_e4c_memory_runtime_preflight');launch=read(D/'launch.json');ex=read(D/'exit.json');w=read(D/'worker_preflight.json');review=read(I/'e4c_memory_runtime_preflight_review.json');cmd=launch['command']
assert s['cwd']==str(R) and s['command'][1:]==['e4_c_memory_runtime/runtime_preflight_launch.py','--run-id','001']
assert review['status']=='passed' and review['independent_review'] and review['gpu_authorized'] is False and review['reviewer_sha256']==sha(I/'review_e4c_memory_runtime_preflight.py') and all(sha(p)==h for p,h in review['source_sha256'].items())
assert launch['approval_bundle']['review_sha256']==sha(I/'e4c_memory_runtime_preflight_review.json') and launch['source_sha256']==review['source_sha256'] and len(launch['source_sha256'])==88 and launch['gpu_requested'] is False
assert cmd[:2]==['docker','run'] and '--gpus' not in cmd and '--device' not in cmd and '--rm' not in cmd
assert ex['returncode']==0 and ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['client_exited_while_container_running'] is False
cid=ex['container_identity']['container_id'];data=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,timeout=30,check=True).stdout);assert len(data)==1;c=data[0];state=c['State'];cfg=c['Config'];host=c['HostConfig']
assert c['Id']==cid and c['Name']=='/'+launch['name'] and c['Image']==ex['container_identity']['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
assert host['NetworkMode']=='none' and host.get('Devices') in ([],None) and host.get('DeviceRequests') in ([],None) and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1]
entry=cmd[cmd.index('--entrypoint')+1];image_index=cmd.index(c['Image']);assert cfg['Entrypoint']==[entry] and cfg['Cmd']==cmd[image_index+1:]
env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x);expected_env=[cmd[i+1].split('=',1) for i,x in enumerate(cmd) if x=='--env'];assert all(env.get(k)==v for k,v in expected_env) and env.get('CUDA_VISIBLE_DEVICES')==''
expected=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount'];actual={x['Destination']:x for x in c['Mounts']};assert len(expected)==len(actual)
for spec in expected:
 parts=spec.split(',');attrs=dict(z.split('=',1) for z in parts if '=' in z);m=actual[attrs['dst']];assert m['Type']=='bind' and os.path.realpath(m['Source'])==os.path.realpath(attrs['src']) and m['RW']==('readonly' not in parts)
assert (w['status'],w['gpu_authorized'],w['cuda_initialized'],w['model_loaded'],w['GT_opened'],w['scoring_performed'],w['predictions'])==('passed',False,False,False,False,False,False)
assert w['approval_bundle']==launch['approval_bundle'] and w['sampler_import']['status']=='passed' and w['sampler_import']['process_isolated'] is True
previous=w['prior_worker_preflight'];assert previous['status']=='passed' and previous['cuda_initialized'] is False and previous['model_loaded'] is False and len(previous['weight_stats_before'])==4
assert previous['memory_job_counts']=={'C-ON':24,'C-DROP':24,'C-SKIP':24}
out=I/'e4c_memory_runtime_container001_review.json';proof={'status':'passed','independent_review':True,'scope':'no-GPU runtime preflight only; no actual sampler handshake','container_id':cid,'sources_verified':88,'weights_preflighted':4,'launch_sha256':sha(D/'launch.json'),'exit_sha256':sha(D/'exit.json'),'worker_preflight_sha256':sha(D/'worker_preflight.json'),'started_sha256':sha(H/'monitor_container_preflight001/started.json'),'terminal_sha256':sha(H/'monitor_container_preflight001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'gpu_authorized':False,'errors':[]}
with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
