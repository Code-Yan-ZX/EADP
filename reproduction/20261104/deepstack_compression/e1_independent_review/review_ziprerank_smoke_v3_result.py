"""Verify ZipRerank diagnostic artifacts without calling the model or labels."""
import hashlib,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e6/ziprerank_smoke_v3';I=R/'e1_independent_review';D=H/'run_smoke001';O=D/'observations'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
plan=read(H/'plan001.json');proof=read(H/'independent_prelaunch_review.json');s=mon(H/'monitor_smoke001');mon(I/'monitor_ziprerank_smoke_v3_review');launch=read(D/'launch.json');ex=read(D/'exit.json');end=read(O/'worker_complete.json')
assert proof['status']=='passed' and proof['independent_review'] and proof['plan_sha256']==launch['plan_sha256']==sha(H/'plan001.json') and all(sha(k)==v for k,v in plan['sha256'].items())
assert s['command'][1:]==['e6/ziprerank_smoke_v3/launch.py','--stage','smoke'] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid']
assert s['stale_seconds']==1200 and s['progress_paths']==[str(O/'progress.jsonl')] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['failure_type'] is None
cid=ex['container_identity']['container_id'];c=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,check=True,timeout=30).stdout)[0];state=c['State'];cfg=c['Config'];host=c['HostConfig'];cmd=launch['command']
assert c['Id']==cid and c['Name']=='/'+launch['container_name'] and c['Image']==ex['container_identity']['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
assert '--gpus' in cmd and host['DeviceRequests'] and host['NetworkMode']=='none' and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(c['Image'])+1:]
anchors=plan['anchor'];rows=[json.loads(x) for x in (R/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];selected=[x for x in rows if x['task']==anchors['task'] and str(x['qid'])==anchors['qid']];assert len(selected)==1;row=selected[0]
assert row['message_sha256']==anchors['message_sha256']==hashlib.sha256(json.dumps(row['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest() and sha(row['image_path'])==anchors['image_sha256']==row['image_sha256']
start=read(O/'started.json');assert start['plan_sha256']==sha(H/'plan001.json') and start['task']==anchors['task'] and start['qid']==anchors['qid'] and start['GT_opened'] is False
assert end['status']=='diagnostic_completed' and end['arms']==['disabled','keep_all','keep_256'] and end['arm_statuses']=={'disabled':'completed','keep_all':'failed','keep_256':'failed'}
arms={x:read(O/(x+'.json')) for x in end['arms']};files={}
for name,x in arms.items():
 assert read(O/(name+'_started.json'))['arm']==name
 for kind in (name+'.json',name+'_started.json'):files[kind]=sha(O/kind)
 if name=='disabled':
  gen=read(O/(name+'_generated.json'));assert len(gen['tokens'])==33 and all(z==gen['forced_token_id'] for z in gen['tokens']) and len(x['language_model_calls'])==len(x['logits_trace'])==33 and all(x['diagnostic_checks'].values());files[name+'_generated.json']=sha(O/(name+'_generated.json'))
 else:
  e=read(O/(name+'_exception.json'));assert e['arm']==name and e['error_type']==x['error_type']=='IndexError' and e['error']==x['error'] and e['traceback']==x['traceback'] and 'get_rope_index' in e['traceback'] and 'does not match the shape' in e['error']
  assert len(x['language_model_calls'])==len(x['logits_trace'])==1 and x['prefill_checks']['positions_match'] and x['prefill_checks']['image_main_matches'] and x['prefill_checks']['text_embeddings_match'] and x['prefill_checks']['ds_matches']==[True]*3
  assert x['prefill_checks']['selected_count']==(1024 if name=='keep_all' else 256) and x['logits_trace'][0]['rope_deltas']==x['original_rope_deltas'] and x['logits_trace'][0]['model_rope_deltas'] is None
  files[name+'_exception.json']=sha(O/(name+'_exception.json'))
progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert [x['arm'] for x in progress if x['event']=='arm_started']==end['arms'] and [x['arm'] for x in progress if x['event']=='arm_recorded']==end['arms']
assert arms['keep_all']['keep_all_vs_disabled_logit_checks'][0]['allclose_atol_0.01_rtol_0.01'] and arms['keep_all']['keep_all_vs_disabled_logit_checks'][0]['argmax_equal']
files.update({n:sha(O/n) for n in ('worker_complete.json','started.json','progress.jsonl')})
proofout={'status':'passed','independent_review':True,'scope':'artifact verification only; enabled diagnostic arms failed','arm_statuses':end['arm_statuses'],'disabled_decode_calls':32,'enabled_completed_decode_calls':0,'formal_limitations':['Prefill comparisons/logit differences are runtime summaries; full arrays are not available for independent reconstruction.','No-hook control not yet run; observer effects cannot be excluded empirically.','One fixed image/forced prefix; no natural quality, speed or generalization conclusion.'],'files_sha256':files,'plan_sha256':sha(H/'plan001.json'),'run_started_sha256':sha(H/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(H/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'exit.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'errors':[]}
with (I/'ziprerank_smoke_v3_result_review.json').open('x') as f:json.dump(proofout,f,indent=2);f.write('\n')
