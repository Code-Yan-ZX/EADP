"""Independently audit 32x3 archived-input scatter-graph results, CPU only."""
import argparse,hashlib,json,math,os,subprocess
from pathlib import Path
import torch
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e5/author_graph_scatter_real32';O=D/'run_smoke001/observations'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'author_graph_scatter_real32_result_review.json'
formal=D/'smoke001_verification.json';f=read(formal);fs=terminal(D/'monitor_verify001',D/'verify.py',formal);rs=terminal(D/'monitor_smoke001',D/'launch.py');terminal(R/'monitor_author_graph_scatter_real32_prelaunch',R/'review_author_graph_scatter_real32_prelaunch.py',D/'independent_prelaunch_review.json');plan=read(D/'plan001.json');vp=read(D/'verification_plan001.json');launch=read(D/'run_smoke001/launch.json');ex=read(D/'run_smoke001/exit.json')
assert f['status']=='passed' and f['GT_opened'] is False and f['scoring_performed'] is False and f['errors']==[] and f['verifier_sha256']==vp['verifier_sha256']==sha(D/'verify.py') and vp['plan_sha256']==sha(D/'plan001.json') and sha(D/'verification_plan001.json')=='2ac3a9a91a420150e155be0cd9db1740bc6ca327d51ed16a34985bbb5fe9ea94'
assert len(plan['sha256'])==53 and all(sha(x)==h for x,h in plan['sha256'].items()) and all(sha(x)==h for x,h in f['files_sha256'].items());assert rs['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['gpu_pids_after']==[]
obj=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];st=obj['State'];host=obj['HostConfig'];cfg=obj['Config'];cmd=launch['command'];assert obj['Id']==ex['container_identity']['container_id'] and not st['Running'] and st['Pid']==st['ExitCode']==0 and not st['OOMKilled'] and obj['Image']==ex['container_identity']['image'] and cfg['Cmd']==[str(D/'worker.py'),'--out',str(O)] and host['NetworkMode']=='none' and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]]
assert {x['PathOnHost'] for x in host['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'}
mount=[]
for i,x in enumerate(cmd[:-1]):
 if x=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
 if x=='--env':assert cmd[i+1] in cfg['Env']
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in obj['Mounts'] if x['Type']=='bind')
manifest=plan['case_manifest'];assert len(manifest)==96 and len({(x['image_index'],x['budget']) for x in manifest})==96 and {(x['image_index'],x['budget']) for x in manifest}=={(i,k) for i in range(32) for k in (128,256,512)}
prior=read(B/'e1_independent_review/selector_real32_run001_review.json');assert prior['status']=='passed' and prior['independent_review'] is True and all(prior['array_sha256'].get(x['path'])==x['sha256'] for x in manifest)
complete=read(O/'worker_complete.json');attempts=complete['cases'];assert complete['status']=='attempts_recorded' and [x['case'] for x in attempts]==['k_128','k_256','k_512'];progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert [x['case'] for x in progress if x['event']=='case_started']==[x['case'] for x in attempts] and [x['case'] for x in progress if x['event']=='case_finished']==[x['case'] for x in attempts]
counts=[];all_equal=True
for attempt,budget in zip(attempts,(128,256,512)):
 name=attempt['case'];path=O/(name+'.jsonl');rows=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [];comp=[x for x in rows if x.get('event')=='case_compared'];failed=[x for x in rows if x.get('event')=='failed'];expected=[x for x in manifest if x['budget']==budget];assert len(expected)==32 and [x['image_index'] for x in comp]==list(range(len(comp)))
 failures=[]
 for row,item in zip(comp,expected):
  assert all(row.get(k)==item[k] for k in ('image_index','task','qid','group_id')) and row['case_sha256']==item['sha256'];path=Path(item['path']);assert sha(path)==item['sha256'];saved=torch.load(path,map_location='cpu',weights_only=True)['ordered'].tolist();original=row['author_order'];graph=row['graph_order'];assert len(original)==len(graph)==1 and len(original[0])==len(graph[0])==budget and len(set(original[0]))==len(set(graph[0]))==budget and row['author_quota']==row['graph_quota']==[budget]
  equal=original==graph and original==saved and row['repeat_same_input_equal'] is True;assert row['graph_equals_author'] is (graph==original) and row['author_equals_archive'] is (original==saved) and all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in ('input_copy_plus_replay_device_ms','graph_only_device_ms','host_wall_ms'))
  if not equal:failures.append(item['image_index'])
 success=attempt['returncode']==0 and not attempt['timed_out'] and len(comp)==32 and not failures and not failed;all_equal=all_equal and success
 assert len([x for x in rows if x.get('event')=='capture_completed'])<2
 if success:assert len([x for x in rows if x.get('event')=='budget_completed' and x.get('all_identity_equal') is True])==1
 else:assert failed or attempt['timed_out'] or failures or attempt['returncode']!=0
 formalrow=next(x for x in f['budgets'] if x['k']==budget);assert formalrow['outcome']==('completed' if attempt['returncode']==0 else 'timeout' if attempt['timed_out'] else 'failed') and formalrow['compared_images']==len(comp) and formalrow['identity_mismatch_images']==failures
 counts.append({'budget':budget,'compared':len(comp),'identity_mismatch_images':failures,'returncode':attempt['returncode'],'timed_out':attempt['timed_out']})
assert all_equal==f['complete_96_identity']
out={'status':'passed','independent_review':True,'complete_96_identity':all_equal,'budgets':counts,'formal_sha256':sha(formal),'formal_started_sha256':sha(D/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(D/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'plan_sha256':sha(D/'plan001.json'),'verification_plan_sha256':sha(D/'verification_plan001.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['32 fixed images only; not general equivalence, model integration or TTFT','setup and device replay separate; allocator figures exclude driver and graph metadata','runtime repeated output checks do not prove correctness on unseen inputs'],'errors':[]}
with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
print('passed')
