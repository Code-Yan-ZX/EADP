"""Independent artifact review of unmodified author's no-forward-hook control."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e6/ziprerank_nohook_v2';I=R/'e1_independent_review';D=H/'run_smoke001';O=D/'observations'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
plan=read(H/'plan001.json');proof=read(H/'independent_prelaunch_review.json');launch=read(D/'launch.json');ex=read(D/'exit.json');s=mon(H/'monitor_smoke001');end=read(O/'worker_complete.json')
assert proof['status']=='passed' and proof['independent_review'] and proof['plan_sha256']==launch['plan_sha256']==sha(H/'plan001.json') and all(sha(k)==v for k,v in plan['sha256'].items())
assert s['command'][1:]==['e6/ziprerank_nohook_v2/launch.py','--stage','smoke'] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and s['stale_seconds']==1200 and s['progress_paths']==[str(O/'progress.jsonl')]
assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['failure_type'] is None
cid=ex['container_identity']['container_id'];c=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,check=True,timeout=30).stdout)[0];state=c['State'];cfg=c['Config'];host=c['HostConfig'];cmd=launch['command']
assert c['Id']==cid and c['Name']=='/'+launch['container_name'] and c['Image']==ex['container_identity']['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
assert {cmd[i+1] for i,v in enumerate(cmd) if v=='--device'}=={x['PathOnHost'] for x in host['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'} and host['NetworkMode']=='none' and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(c['Image'])+1:]
a=plan['anchor'];rows=[json.loads(x) for x in (R/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];rs=[x for x in rows if x['task']==a['task'] and str(x['qid'])==a['qid']];assert len(rs)==1;r=rs[0]
assert r['message_sha256']==a['message_sha256']==hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest() and sha(r['image_path'])==r['image_sha256']==a['image_sha256']
assert end['status']=='diagnostic_completed' and end['arm_statuses']=={'disabled':'completed','keep_all':'failed','keep_256':'failed'}
files={}
for name in ('disabled','keep_all','keep_256'):
 x=read(O/(name+'.json'));st=read(O/(name+'_started.json'));assert st['arm']==x['arm']==name and st['model_forward_hooks']==x['model_forward_hooks_after']==0
 for n in (name+'.json',name+'_started.json'):files[n]=sha(O/n)
 if name=='disabled':
  g=read(O/(name+'_generated.json'));prev=read(R/'e6/ziprerank_smoke_v3/run_smoke001/observations/disabled_generated.json');assert x['status']=='completed' and x['generated_token_count']==33 and x['forced_tokens_match'] and g['tokens']==prev['tokens'] and g['forced_token_id']==prev['forced_token_id'];files[name+'_generated.json']=sha(O/(name+'_generated.json'))
 else:
  e=read(O/(name+'_exception.json'));prev=read(R/'e6/ziprerank_smoke_v3/run_smoke001/observations'/(name+'_exception.json'));assert x==e and x['status']=='failed' and x['error_type']==prev['error_type']=='IndexError' and x['error']==prev['error'] and 'get_rope_index' in x['traceback'];files[name+'_exception.json']=sha(O/(name+'_exception.json'))
progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert [x['arm'] for x in progress if x['event']=='arm_started']==['disabled','keep_all','keep_256'] and [x['arm'] for x in progress if x['event']=='arm_recorded']==['disabled','keep_all','keep_256']
files.update({n:sha(O/n) for n in ('started.json','progress.jsonl','worker_complete.json')});result={'status':'passed','independent_review':True,'scope':'artifact review; enabled author paths failed','arm_statuses':end['arm_statuses'],'all_instance_and_global_forward_hook_counts_zero_at_boundaries':True,'disabled_tokens_equal_observed_run':True,'enabled_errors_equal_observed_run':True,'files_sha256':files,'plan_sha256':sha(H/'plan001.json'),'run_started_sha256':sha(H/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(H/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'exit.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['One image and forced prefix only; failure confirmed without module/global forward hooks at the recorded boundaries.','No-hook run cannot observe internal per-step states and cannot prove the particular rope-state mechanism by itself.'],'errors':[]}
with (I/'ziprerank_nohook_v2_result_review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
