"""Independent reconstruction of archived PixelPrune three-anchor ZipRerank natural diagnostic metadata."""
import argparse,collections,hashlib,json,math,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e6/ziprerank_natural_v4';O=D/'run_smoke001/observations'
def get(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'];return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=get(D/'plan001.json');assert sha(D/'plan001.json')=='b87c850ce9636e3e63d5e34c9a8c188f9031aa76929e1a69bfca290d09e80758'
for f,h in plan['sha256'].items():assert sha(f)==h,f
term(B/'e1_independent_review/monitor_ziprerank_natural_v4_prelaunch');s=term(D/'monitor_smoke001');l=get(D/'run_smoke001/launch.json');ex=get(D/'run_smoke001/exit.json');assert l['plan_sha256']==sha(D/'plan001.json') and s['worker_pid']==l['controller']['pid']==ex['controller']['pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] and ex['client_exited_while_container_running'] is False and ex['gpu_pids_after']==[]
ins=json.loads(subprocess.run(['docker','inspect',ex['container_identity']['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout)[0];state=ins['State'];assert ins['Id']==ex['last_inspect']['id'] and ins['Image']==ex['container_identity']['image'] and not state['Running'] and state['Pid']==0 and state['ExitCode']==0 and not state['OOMKilled'];assert ins['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)] and ins['HostConfig']['NetworkMode']=='none'
devices=ins['HostConfig']['Devices'];assert {(x['PathOnHost'],x['PathInContainer'],x['CgroupPermissions']) for x in devices}=={(x,x,'rwm') for x in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']}
cmd=l['command'];mount=[]
for i,c in enumerate(cmd[:-1]):
 if c=='--mount':
  v=dict(z.split('=',1) if '=' in z else (z,True) for z in cmd[i+1].split(','));mount.append((str(Path(v['src']).resolve()),v['dst'],'readonly' not in v))
assert sorted(mount)==sorted((str(Path(x['Source']).resolve()),x['Destination'],x['RW']) for x in ins['Mounts'] if x['Type']=='bind')
import struct
done=get(O/'worker_complete.json');assert done['status']=='attempts_recorded' and done['no_GT'] is True and done['scoring'] is False
progress=[json.loads(line) for line in (O/'progress.jsonl').read_text().splitlines()];observed={};status={}
for anchor in plan['anchors']:
 task=anchor['task'];qid=anchor['qid'];summary={};digests=[]
 for name in plan['arms']:
  key=task+'_'+qid+'_'+name;r=get(O/(key+'.json'));raw=get(O/(key+'_generated.json'));seq=raw['input_sequence_length'];full=raw['sequences'];assert len(full)==1 and seq<=len(full[0]);tokens=full[0][seq:]
  assert r['task']==task and r['qid']==qid and r['group_id']==anchor['group_id'] and r['arm']==name and r['status']=='completed' and r['output_ids']==tokens
  digest=hashlib.sha256(struct.pack('<'+'q'*seq,*full[0][:seq])).hexdigest();assert r['prompt_sha256']==digest;digests.append(digest)
  assert r['local_rope_adapter']==(name!='disabled') and r['full_user_text_adapter']==(name=='user_text_256') and r['prompt_ids_unchanged'] and r['selector_restored']
  ctrl=r['generation_controls'];assert all(ctrl.get(k)==v for k,v in plan['generation'].items() if k!='return_dict_in_generate')
  count=256 if name.endswith('_256') else 1024;pre=r['prefill'];assert pre['selected_count']==count and pre['selected_unique'] and pre['selected_in_range'] and pre['positions_match'] and pre['visual_count']==count and pre['ds_rows']==[count]*3
  if name!='disabled':
   selected=r['selected_indices'];assert len(selected)==1 and len(selected[0])==count and len(set(selected[0]))==count and all(0<=x<1024 for x in selected[0])
  else:assert not r['selected_indices']
  sel=r['selection_trace'];assert sel.get('calls')==(1 if name=='user_text_256' else 0)
  if name=='user_text_256':assert sel['input_ids_sha256']==digest and sel['returned_exact_selected_embeddings'] is True and len(sel['returned_embedding_sha256'])==64
  calls=r['calls'];initial=seq-1024+count;assert len(calls)==len(tokens) and calls[0]['input_length']==initial
  assert all(row['input_length']==1 and row['cache_length_before']==initial+i-1 for i,row in enumerate(calls) if i)
  assert len(r['final_cache'])==36 and all(z['key_shape']==[1,8,initial+len(tokens)-1,128] and z['value_shape']==[1,8,initial+len(tokens)-1,128] for z in r['final_cache'])
  assert 0<len(tokens)<=2048 and (r['eos'] or r['truncated']) and r['eos']==(tokens[-1] in ctrl['eos_token_id'])
  assert len([e for e in progress if e.get('event')=='arm_started' and e.get('key')==key])==1 and len([e for e in progress if e.get('event')=='arm_recorded' and e.get('key')==key])==1
  status[key]='completed';summary[name]={'visual_tokens':count,'generated_tokens':len(tokens),'eos':r['eos']}
 assert len(set(digests))==1;observed[task]=summary
assert done['statuses']==status
files=list(O.glob('*.json'))+[O/'progress.jsonl',D/'run_smoke001/launch.json',D/'run_smoke001/exit.json',D/'monitor_smoke001/started.json',D/'monitor_smoke001/terminal.json',D/'independent_prelaunch_review.json']
result={'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'observed':observed,'arms':12,'files_sha256':{str(f):sha(f) for f in files},'reviewer_sha256':sha(__file__),'limitations':['three tasks, one image each; no quality scoring or throughput','local RoPE repair on all enabled arms; full user-message text selector is another local intervention and includes instructions/options','per-step cache length is global/first layer; only final all-layer shapes are archived, not KV contents','positions, DS and selected embedding equality are online checks without full high-dimensional tensor archives'],'errors':[]}
a.out.write_text(json.dumps(result,indent=2)+'\n');print('passed')
