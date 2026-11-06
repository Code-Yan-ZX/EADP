"""Independent source/CPU/container review for one forced-prefix diagnostic."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e6/ziprerank_smoke_v3';I=R/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s,t
a=argparse.ArgumentParser();a.add_argument('--out',required=True);o=Path(a.parse_args().out).resolve();assert o==(H/'independent_prelaunch_review.json').resolve()
p=read(H/'plan001.json');pre=read(H/'preflight001.json');launch=read(H/'run_preflight001/launch.json');ex=read(H/'run_preflight001/exit.json');s,t=mon(H/'monitor_preflight001');ts,tt=mon(H/'monitor_test001');mon(I/'monitor_ziprerank_scope001')
assert p['forced_generated_tokens']==33 and p['forced_decode_calls_after_prefill']==32 and p['stale_seconds']==1200 and p['arms']==['disabled','keep_all','keep_256']
assert all(sha(k)==v for k,v in p['sha256'].items()) and pre['source_plan_sha256']==launch['plan_sha256']==sha(H/'plan001.json')
assert s['cwd']==str(R) and s['command'][1:]==['e6/ziprerank_smoke_v3/launch.py','--stage','preflight'] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid']
assert ts['command'][1:]==['e6/ziprerank_smoke_v3/test_worker.py'] and 'Ran 8 tests' in (H/'monitor_test001/stderr.log').read_text()
assert (pre['status'],pre['cuda_initialized'],pre['model_loaded'],pre['GT_opened'],pre['weight_sha256_verified'])==('passed',False,False,False,4)
assert ex['returncode']==0 and ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['failure_type'] is None and ex['timed_out'] is False and ex['client_exited_while_container_running'] is False
cmd=launch['command'];cid=ex['container_identity']['container_id'];data=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,timeout=30,check=True).stdout);assert len(data)==1;c=data[0];state=c['State'];cfg=c['Config'];host=c['HostConfig']
assert c['Id']==cid and c['Name']=='/'+launch['container_name'] and c['Image']==ex['container_identity']['image']==ex['last_inspect']['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
assert '--gpus' not in cmd and '--device' not in cmd and '--rm' not in cmd and host['NetworkMode']=='none' and host.get('Devices') in (None,[]) and host.get('DeviceRequests') in (None,[])
assert cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(c['Image'])+1:]
env=dict(x.split('=',1) for x in cfg['Env'] if '=' in x);ee=[cmd[i+1].split('=',1) for i,x in enumerate(cmd) if x=='--env'];assert all(env.get(k)==v for k,v in ee) and env.get('CUDA_VISIBLE_DEVICES')==''
mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount'];actual={x['Destination']:x for x in c['Mounts']};assert len(mounts)==len(actual)
for text in mounts:
 parts=text.split(',');attrs=dict(x.split('=',1) for x in parts if '=' in x);m=actual[attrs['dst']];assert m['Type']=='bind' and os.path.realpath(m['Source'])==os.path.realpath(attrs['src']) and m['RW']==('readonly' not in parts)
proc=subprocess.run([sys.executable,str(H/'test_worker.py')],cwd=R,capture_output=True,text=True,timeout=120,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});assert proc.returncode==0 and 'Ran 8 tests' in proc.stderr and 'OK' in proc.stderr and all(sha(k)==v for k,v in p['sha256'].items())
for _ in range(100):
 try:started=read(I/'monitor_ziprerank_smoke_v3_review/started.json');break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
else:raise ValueError('review supervisor absent')
assert started['worker_pid']==started['process_group']==os.getpid() and started['cwd']==str(R) and started['command'][-2:]==['--out',str(o)]
proof={'status':'passed','independent_review':True,'scope':'single DEV image, three forced diagnostics; no quality or timing claim','plan_sha256':sha(H/'plan001.json'),'preflight_sha256':sha(H/'preflight001.json'),'preflight_started_sha256':sha(H/'monitor_preflight001/started.json'),'preflight_terminal_sha256':sha(H/'monitor_preflight001/terminal.json'),'preflight_exit_sha256':sha(H/'run_preflight001/exit.json'),'preflight_launch_sha256':sha(H/'run_preflight001/launch.json'),'reviewer_sha256':sha(__file__),'reviewer_path':str(Path(__file__).resolve()),'review_started_sha256':sha(I/'monitor_ziprerank_smoke_v3_review/started.json'),'review_terminal_path':str(I/'monitor_ziprerank_smoke_v3_review/terminal.json'),'container_id':cid,'synthetic_tests':8,'GT_opened':False,'scoring_performed':False,'gpu_used':False,'limitations':['Completion records diagnostic attempts, not necessarily successful paths.','Single fixed image and forced prefix; author exceptions and all failed checks must remain visible.','FA2 matches E1 backend, but complete generation recipe and method are not automatically equivalent.'],'errors':[]}
with o.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
