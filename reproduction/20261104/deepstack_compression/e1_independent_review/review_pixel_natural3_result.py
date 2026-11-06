"""Independent reconstruction of archived PixelPrune three-anchor natural diagnostic metadata."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_natural3_v3';O=D/'run_smoke001/observations'
def get(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='972484e2480124136e2e64c671de03dd02602fc9a42408a17962e475d6f834e0'
for f,h in plan['sha256'].items():assert sha(f)==h,f
term(B/'e1_independent_review/monitor_pixel_natural3_prelaunch');s=term(D/'monitor_smoke001');l=get(D/'run_smoke001/launch.json');ex=get(D/'run_smoke001/exit.json');assert l['plan_sha256']==sha(D/'plan001.json') and s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[]
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'];assert ins['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and ins['HostConfig']['NetworkMode']=='none'
devices=ins['HostConfig']['Devices'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}=={(x,x,'rwm') for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']}
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
done=get(O/'worker_complete.json');assert done['status']=='diagnostic_completed' and done['GT_opened'] is False and done['scoring_performed'] is False
progress=[json.loads(line) for line in (O/'progress.jsonl').read_text().splitlines()];observed={}
for anchor in plan['anchors']:
 task=anchor['task'];qid=anchor['qid'];key=task+'_'+qid;r=get(O/(key+'.json'));raw=get(O/(key+'_generated.json'))
 assert r['task']==task and r['qid']==qid and r['message_sha256']==anchor['message_sha256'] and r['image_sha256']==anchor['image_sha256'] and r['status']=='completed'
 assert r['generation_kwargs']==plan['generation']
 ids=r['selected_indices'];n=r['selected_count'];assert len(ids)==n and ids==sorted(set(ids)) and all(type(i) is int and 0<=i<1024 for i in ids)
 seq=r['sequence_length'];generated=raw['generated_token_ids'];assert raw['prompt_length']==seq and raw['full_sequence_ids'][seq:]==generated and len(generated)==r['generated_count']
 assert 0<len(generated)<=2048 and (generated[-1] in r['eos_token_ids'] or len(generated)==2048)
 assert r['visual_rows']==r['prefill_visual_count']==n and r['ds_rows']==r['prefill_ds_rows']==[n]*3
 assert all(r[k] is True for k in ('prefill_main_match','prefill_text_match','prefill_positions_match','hook_cleanup_valid')) and r['prefill_ds_match']==[True]*3
 trace=r['trace'];assert len(trace)==len(generated);initial=seq-1024+n
 for i,t in enumerate(trace):
  assert t['rope_delta']==r['original_rope_delta']==t['model_rope_delta'] and len(t['layers'])==36
  assert all(z['key']==[1,8,initial+i,128] and z['value']==[1,8,initial+i,128] for z in t['layers'])
  if i:assert t['position_ids']==[[[r['original_max_position']+i]]]*3 and t['cache_position']==[seq+i-1]
 assert all(r['diagnostic_checks'].values())
 assert [e['index'] for e in progress if e.get('event')=='forward_complete' and e.get('task')==task and str(e.get('qid'))==qid]==list(range(len(trace)))
 observed[task]={'qid':qid,'visual_tokens':n,'generated_tokens':len(generated),'eos':r['ends_eos']}
assert done['arm_statuses']=={a['task']+'/'+a['qid']:'completed' for a in plan['anchors']}
files=list(O.glob('*.json'))+[O/'progress.jsonl',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'independent_prelaunch_review.json']
result={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'observed':observed,'layers_per_call':36,'files_sha256':{str(f):sha(f) for f in files},'reviewer_sha256':sha(__file__),'limitations':['three natural outputs only; no quality score','visual budget varies by image and is not equal K','top_p=0.8 and top_k=20 differ from E1 0.001 and 1, although do_sample=false; no complete generation recipe identity','full processor tensors, per-step logits, visual/DS and KV contents not archived; elementwise matches are online checks','no timing or general correctness claims'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
