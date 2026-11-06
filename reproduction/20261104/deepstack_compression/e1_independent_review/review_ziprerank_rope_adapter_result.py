"""Independent audit of recorded RoPE-adapted forced-prefix diagnostic."""
import hashlib,json,re,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];H=R/'e6/ziprerank_rope_adapter';I=R/'e1_independent_review';D=H/'run_smoke001';O=D/'observations'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
plan=read(H/'plan001.json');proof=read(H/'independent_prelaunch_review.json');launch=read(D/'launch.json');ex=read(D/'exit.json');s=mon(H/'monitor_smoke001');mon(I/'monitor_ziprerank_rope_adapter_review');mon(I/'monitor_ziprerank_nohook_v2_result');end=read(O/'worker_complete.json')
assert proof['status']=='passed' and proof['independent_review'] and proof['plan_sha256']==launch['plan_sha256']==sha(H/'plan001.json') and proof['nohook_control_review_sha256']==sha(I/'ziprerank_nohook_v2_result_review.json') and all(sha(k)==v for k,v in plan['sha256'].items())
assert s['command'][1:]==['e6/ziprerank_rope_adapter/launch.py','--stage','smoke'] and s['worker_pid']==launch['controller']['pid']==ex['controller']['pid'] and s['stale_seconds']==1200 and s['progress_paths']==[str(O/'progress.jsonl')]
assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['failure_type'] is None
cid=ex['container_identity']['container_id'];c=json.loads(subprocess.run(['docker','inspect','--type','container',cid],capture_output=True,text=True,check=True,timeout=30).stdout)[0];state=c['State'];cfg=c['Config'];host=c['HostConfig'];cmd=launch['command']
assert c['Id']==cid and c['Name']=='/'+launch['container_name'] and c['Image']==ex['container_identity']['image'] and state['Running'] is False and state['Pid']==0 and state['ExitCode']==0 and state['OOMKilled'] is False
assert {cmd[i+1] for i,v in enumerate(cmd) if v=='--device'}=={x['PathOnHost'] for x in host['Devices']}=={'/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'} and host['NetworkMode']=='none' and cfg['User']==cmd[cmd.index('--user')+1] and cfg['WorkingDir']==cmd[cmd.index('--workdir')+1] and cfg['Entrypoint']==[cmd[cmd.index('--entrypoint')+1]] and cfg['Cmd']==cmd[cmd.index(c['Image'])+1:]
a=plan['anchor'];rows=[json.loads(x) for x in (R/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];rr=[x for x in rows if x['task']==a['task'] and str(x['qid'])==a['qid']];assert len(rr)==1;r=rr[0];assert r['message_sha256']==a['message_sha256']==hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest() and sha(r['image_path'])==r['image_sha256']==a['image_sha256']
assert end['status']=='diagnostic_completed' and end['arm_statuses']=={'disabled':'completed','keep_all':'completed','keep_256':'completed'}
files={};all_tokens=[]
for name in ('disabled','keep_all','keep_256'):
 x=read(O/(name+'.json'));st=read(O/(name+'_started.json'));g=read(O/(name+'_generated.json'));assert x['status']=='completed' and st['arm']==g['arm']==x['name']==name and st['adapter_sets_model_rope_delta_before_generate']==x['adapter_sets_model_rope_delta_before_generate']==(name!='disabled')
 assert len(g['tokens'])==33 and all(v==g['forced_token_id'] for v in g['tokens']);all_tokens.append(g['tokens']);trace=x['language_model_calls'];logs=x['logits_trace'];assert len(trace)==len(logs)==33 and all(y['finite'] is True for y in logs)
 pre=x['prefill_checks'];expected=256 if name=='keep_256' else 1024;assert pre['positions_match'] and pre['image_main_matches'] and pre['text_embeddings_match'] and pre['ds_matches']==[True]*3 and pre['ds_count']==pre['expected_ds_count']==3 and pre['selected_count']==expected and pre['selected_unique'] and pre['selected_in_range']
 if name!='disabled':
  assert len(x['selected_indices'])==1 and len(x['selected_indices'][0])==expected and len(set(x['selected_indices'][0]))==expected and all(0<=v<1024 for v in x['selected_indices'][0]) and x['selected_indices'][0]==sorted(x['selected_indices'][0])
 n=x['original_sequence_length'];kept=n-(768 if name=='keep_256' else 0);delta=x['original_rope_deltas']
 for i,(tr,lg) in enumerate(zip(trace,logs)):
  assert tr['input_length']==(kept if i==0 else 1) and tr['cache_position']==(list(range(n)) if i==0 else [n+i-1]) and lg['rope_deltas']==lg['model_rope_deltas']==delta
  if i>0:assert tr['position_ids']==[[[x['original_max_position']+i]]]*3 and len(tr['all_layer_cache_before'])==36
  for z in tr['all_layer_cache_before'] if i else []:assert len(z['key_shape'])==len(z['value_shape'])==4 and z['key_shape'][-2]==z['value_shape'][-2]==kept+i-1
  assert len(lg['all_layer_cache_after'])==36
  for z in lg['all_layer_cache_after']:assert len(z['key_shape'])==len(z['value_shape'])==4 and z['key_shape'][-2]==z['value_shape'][-2]==kept+i
 fps=x['cache_fingerprints'];assert len(fps)==2 and [v['call_index'] for v in fps]==[0,32] and len(fps[0]['layers'])==len(fps[1]['layers'])==36
 for b,e in zip(fps[0]['layers'],fps[1]['layers']):
  for k in ('key_prefix_sha256','value_prefix_sha256'):assert re.fullmatch('[0-9a-f]{64}',b[k]) and b[k]==e[k]
 assert all(x['diagnostic_checks'].values())
 if name=='keep_all':assert len(x['keep_all_vs_disabled_logit_checks'])==33 and all(z['comparable'] and z['allclose_atol_0.01_rtol_0.01'] and z['argmax_equal'] for z in x['keep_all_vs_disabled_logit_checks'])
 for n in (name+'.json',name+'_started.json',name+'_generated.json'):files[n]=sha(O/n)
assert all(t==all_tokens[0] for t in all_tokens);original=read(I/'ziprerank_nohook_v2_result_review.json');assert original['arm_statuses']=={'disabled':'completed','keep_all':'failed','keep_256':'failed'}
progress=[json.loads(x) for x in (O/'progress.jsonl').read_text().splitlines()];assert [x['arm'] for x in progress if x['event']=='arm_started']==end['arms'] and [x['arm'] for x in progress if x['event']=='arm_recorded']==end['arms']
files.update({n:sha(O/n) for n in ('started.json','progress.jsonl','worker_complete.json')});result={'status':'passed','independent_review':True,'scope':'single-image local RoPE compatibility adaptation, forced prefix only','arm_statuses':end['arm_statuses'],'calls_per_arm':33,'decode_calls_per_arm':32,'cache_layers':36,'logit_comparisons_keep_all':33,'files_sha256':files,'plan_sha256':sha(H/'plan001.json'),'nohook_original_review_sha256':sha(I/'ziprerank_nohook_v2_result_review.json'),'run_started_sha256':sha(H/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(H/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'exit.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['This is a local adaptation, not an unmodified author baseline.','Full high-dimensional inputs, DS, KV and logits are not archived; elementwise and logit comparisons are runtime observations, with recorded KV prefix hashes independently checked.','Single fixed image and forced prefix only; no natural quality, latency or generalization claim.'],'errors':[]}
with (I/'ziprerank_rope_adapter_result_review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
