"""Independent reconstruction of archived PixelPrune 384-question PixelPrune DEV natural panel metadata."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_t010_dev384';O=D/'run_smoke001/observations'
def get(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='4081eb373545b5e9dfcc2069806c053023e39c11fcdd0b71fc09b42b9b7ab682'
for f,h in plan['sha256'].items():assert sha(f)==h,f
term(B/'e1_independent_review/monitor_pixel_t010_dev384_prelaunch');s=term(D/'monitor_smoke001');l=get(D/'run_smoke001/launch.json');ex=get(D/'run_smoke001/exit.json');assert l['plan_sha256']==sha(D/'plan001.json') and s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[]
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'];assert ins['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and ins['HostConfig']['NetworkMode']=='none'
devices=ins['HostConfig']['Devices'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}=={(x,x,'rwm') for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']}
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
import struct,statistics
A=B/'e6/pixelprune_t010_accept';formal=A/'verification001.json';fs=A/'monitor_verify001/started.json';ft=A/'monitor_verify001/terminal.json';term(A/'monitor_verify001');fp=get(formal)
vp=A/'verification_plan001.json';pre=get(D/'independent_prelaunch_review.json');assert sha(vp)=='c76c7c004259dc9d667eac50ba28d543f5be85032620a5949b6ad4ec412dda1b'==pre['acceptance_plan_sha256'];vplan=get(vp);assert vplan['panel_plan_sha256']==sha(D/'plan001.json') and all(sha(f)==h for f,h in vplan['source_sha256'].items())
assert fp['status']=='passed' and fp['questions']==fp['physical_generations']==384 and fp['GT_opened'] is False and fp['scoring_performed'] is False and fp['verifier_sha256']==sha(A/'verify.py')
done=get(O/'worker_complete.json');assert done['n_expected']==done['n_recorded']==384 and done['GT_opened'] is False and done['scoring_performed'] is False
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
e1=get(B/'e1/e1full001_completion.json');ep=Path(e1['files']['journal']['path']);assert sha(ep)==e1['files']['journal']['sha256'];refs={}
with ep.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('arm')=='P3-D1' and x.get('forced') is False:refs[(x['task'],str(x['qid']))]={'scalars':x['generation_scalars'],'eos':x['eos_ids'],'group_id':x['group_id']}
assert len(refs)==384
progress=[json.loads(line) for line in (O/'progress.jsonl').read_text().splitlines()];starts=[(x['task'],str(x['qid'])) for x in progress if x['event']=='arm_started'];ends=[(x['task'],str(x['qid'])) for x in progress if x['event']=='arm_recorded'];expected=[(x['task'],x['qid']) for x in plan['anchors']];assert starts==ends==expected and len(set(expected))==384
observed={};task_counts={};expected_status={};checked=[]
for anchor in plan['anchors']:
 task=anchor['task'];qid=anchor['qid'];key=task+'_'+qid;r=get(O/(key+'.json'));raw=get(O/(key+'_generated.json'));ref=refs[(task,qid)]
 assert all(str(r.get(k))==str(anchor[k]) for k in ('task','qid','group_id','message_sha256','image_sha256')) and r['status']=='completed' and ref['group_id']==anchor['group_id']
 assert r['pixelprune_config']=={'PIXELPRUNE_ENABLED':'true','PIXELPRUNE_METHOD':'pred_2d','PIXELPRUNE_METRIC':'max','PIXELPRUNE_THRESHOLD':'0.1','PIXELPRUNE_ANCHORED':'true','PIXELPRUNE_VIT_LAYER':'0'}
 ctrl=r['generation_kwargs'];assert ctrl==plan['generation'] and all(ctrl.get(k)==v for k,v in ref['scalars'].items()) and r['eos_token_ids']==ref['eos']
 n=r['selected_count'];ids=r['selected_indices'];assert type(n) is int and 1<=n<=1024 and len(ids)==n and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<1024 for i in ids)
 seq=r['sequence_length'];full=raw['full_sequence_ids'];tokens=raw['generated_token_ids'];assert raw['prompt_length']==seq and full[seq:]==tokens and len(tokens)==r['generated_count']
 digest=hashlib.sha256(struct.pack('<'+'q'*seq,*full[:seq])).hexdigest();expected_prompt=anchor['processor']['input_ids'];assert expected_prompt=={'dtype':'torch.int64','shape':[1,seq],'sha256':digest}
 pi=r['processor_identity'];assert pi['processor']==anchor['processor'] and pi['image_start']==anchor['image_start'] and pi['visual_tokens']==1024
 assert tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['decoded'] and 0<len(tokens)<=2048 and r['ends_eos']==(tokens[-1] in ref['eos']) and (r['ends_eos'] or len(tokens)==2048)
 assert r['visual_rows']==r['prefill_visual_count']==n and r['ds_rows']==r['prefill_ds_rows']==[n]*3 and r['prefill_ds_match']==[True]*3
 assert all(r[k] is True for k in ('prefill_main_match','prefill_text_match','prefill_positions_match','hook_cleanup_valid'))
 trace=r['trace'];assert len(trace)==len(tokens);initial=seq-1024+n
 for i,t in enumerate(trace):
  assert t['rope_delta']==r['original_rope_delta']==t['model_rope_delta'] and len(t['layers'])==36
  assert all(z['key']==[1,8,initial+i,128] and z['value']==[1,8,initial+i,128] for z in t['layers'])
  if i:assert t['position_ids']==[[[r['original_max_position']+i]]]*3 and t['cache_position']==[seq+i-1]
 assert all(r['diagnostic_checks'].values());assert [e['index'] for e in progress if e.get('event')=='forward_complete' and e.get('task')==task and str(e.get('qid'))==qid]==list(range(len(trace)));checked.append({'task':task,'qid':qid,'selected_count':n,'generated_count':len(tokens),'ends_eos':r['ends_eos']});task_counts.setdefault(task,[]).append(n);expected_status[task+'/'+qid]='completed'
assert done['arm_statuses']==expected_status and fp['per_question']==checked
for task,nums in task_counts.items():observed[task]={'questions':len(nums),'min_K':min(nums),'median_K':statistics.median(nums),'max_K':max(nums)}
assert all(v['questions']==128 for v in observed.values())
result={'status':'passed','independent_review':True,'questions':384,'physical_generations':384,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'dynamic_K':observed,'verification_plan_sha256':sha(vp),'formal_sha256':sha(formal),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'reviewer_sha256':sha(__file__),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'run_smoke001/exit.json'),'per_question_files_sha256':{str(O/(a['task']+'_'+a['qid']+suffix+'.json')):sha(O/(a['task']+'_'+a['qid']+suffix+'.json')) for a in plan['anchors'] for suffix in ('','_generated')},'limitations':['threshold chosen after DEV outcomes, and dynamic input-dependent visual K is not equal K256 compute','only processor hashes and runtime visual/DS equality flags retained, not full visual/DS/KV tensors','generation hooks mean no timing claim','accepted E1 prediction journal parsed only for scalar and EOS identity','DEV only, no quality score performed in review'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
