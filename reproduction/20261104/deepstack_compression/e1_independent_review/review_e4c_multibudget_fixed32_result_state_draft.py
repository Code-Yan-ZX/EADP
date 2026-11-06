"""Independently reconstruct 12 fixed32 pairs after formal no-GT acceptance."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e4_c_multibudget_fixed32';A=B/'e4_c_multibudget_fixed32_accept';O=D/'run_001';J=O/'diagnostic';E=B/'e4_panel_v2'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(path,script=None):
 s=read(path/'started.json');t=read(path/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
 if script is not None:assert [(Path(s['cwd'])/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()]
 return s
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==R/'e4c_multibudget_fixed32_run001_review.json' and not a.out.exists()
terminal(D/'monitor_run001',D/'launch.py');terminal(A/'monitor_snapshot001',A/'snapshot.py');terminal(A/'monitor_verify001',A/'verify.py')
formal=read(A/'verification001.json');vplan=read(A/'verification_plan001.json');plan=read(D/'execution_plan005.json')
assert formal['status']=='passed' and formal['pairs']==12 and formal['generations']==24 and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['verification_plan_sha256']==sha(A/'verification_plan001.json') and formal['verifier_sha256']==sha(A/'verify.py')
for f,h in vplan['source_sha256'].items():assert sha(f)==h
for k,f in [('run_started_sha256',D/'monitor_run001/started.json'),('run_terminal_sha256',D/'monitor_run001/terminal.json'),('launch_sha256',O/'launch.json'),('exit_sha256',O/'exit.json'),('journal_sha256',J/'journal.jsonl'),('snapshot_manifest_sha256',D/'run_001_sources/manifest.json')]:assert formal[k]==sha(f)
ex=read(O/'exit.json');cid=ex['container_identity']['container_id'];doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]
assert ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['timed_out'] is False and ex['interrupted'] is None and ex['gpu_pids_after']==[] and doc['Id']==cid and doc['State']['Running'] is False and doc['State']['Pid']==0 and doc['State']['ExitCode']==0 and doc['State']['OOMKilled'] is False and doc['HostConfig']['NetworkMode']=='none' and doc['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)]
snapshot=read(D/'run_001_sources/manifest.json');assert set(snapshot['sources'])==set(plan['source_sha256'])
for f,h in plan['source_sha256'].items():
 x=snapshot['sources'][f];assert x['sha256']==h and sha(x['snapshot'])==h
fp=read(E/'panel_003_verification.json');ir=read(R/'e4panel003_review.json');assert fp['status']=='passed' and fp['all_arms_validated']==3072 and ir['status']=='passed' and ir['formal_proof_sha256']==sha(E/'panel_003_verification.json')
terminal(E/'monitor_verify_panel003');terminal(R/'monitor_e4panel003');panel_path=E/'panel_003/records.jsonl';assert fp['files_sha256'][str(panel_path)]==sha(panel_path)
anchors=read(B/'e6/pixel_natural_timing_pilot_v4/plan001.json')['anchors'];assert len(anchors)==3 and len({x['task'] for x in anchors})==3
needed={(x['task'],str(x['qid'])):x for x in anchors};panel={}
for line in panel_path.open():
 x=json.loads(line);key=(x['task'],str(x['qid']),x.get('budget'))
 if key[:2] in needed and key[2] in (128,256,512,1024) and x.get('ds_on') is True and x.get('forced') is False:
  assert key not in panel;panel[key]=x
assert len(panel)==12
events=[json.loads(x) for x in (J/'journal.jsonl').read_text().splitlines()];assert len(events)==84
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
fields=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','prefill_logits','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','forced','forced_token','protocol','min_new_tokens','decode_trace','cache_fingerprints','block_calls','main_merger_calls','ds_calls')
panel_fields=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','block_calls','main_merger_calls','ds_calls')
array_hash={};pairs=0
for ai,anchor in enumerate(anchors):
 for ki,k in enumerate((128,256,512,1024)):
  seg=events[(ai*4+ki)*7:(ai*4+ki+1)*7];assert [e['event'] for e in seg]==['arm_started','raw_generation','arm_completed','arm_started','raw_generation','arm_completed','pair_verified'];fresh,light=seg[2]['record'],seg[5]['record'];ref=panel[(anchor['task'],str(anchor['qid']),k)]
  for index,side,rec in [(0,'fresh',fresh),(3,'light',light)]:
   job={'task':anchor['task'],'qid':str(anchor['qid']),'group_id':anchor['group_id'],'budget':k,'side':side,'protocol':'fixed32'}
   assert all(e['job']==job for e in seg[index:index+3]);raw=seg[index+1]['raw'];assert all(raw.get(z)==rec.get(z) for z in ('task','qid','group_id','budget','ds_on','forced','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens'))
   assert all(rec.get(z)==ref.get(z) for z in panel_fields) and rec['decode_trace'][0]==ref['decode_trace'][0]
   ids=rec['ordered_ids'];sp=rec['spatial_ids'];start=rec['image_start'];n=rec['original_length'];assert len(ids)==len(set(ids))==k and all(type(i) is int and 0<=i<1024 for i in ids) and sp==sorted(ids)
   retained=list(range(start))+[start+i for i in sp]+list(range(start+1024,n));assert rec['retained_ids']==retained
   pos=rec['full_position_ids'];assert rec['prefill_position_ids']==[[[axis[0][i] for i in retained]] for axis in pos];delta=max(max(axis[0]) for axis in pos)+1-n;assert rec['full_delta']==[[delta]] and rec['adjusted_delta']==[[delta+1024-k]]
   mask=[False]*len(retained);mask[start:start+k]=[True]*k;assert rec['injected_mask']==[mask] and len(rec['injected_deepstack'])==3 and rec['protocol']=='fixed32' and rec['forced'] is False and rec['min_new_tokens']==rec['max_new_tokens']==32 and len(rec['output_ids'])==32 and all(i not in rec['eos_ids'] for i in rec['output_ids'][:-1]) and tok.decode(rec['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==rec['answer']
   trace=rec['decode_trace'];cache=rec['cache_fingerprints'];assert len(trace)==len(cache)==32
   for i,(step,layers) in enumerate(zip(trace,cache)):
    prev=0 if not i else len(retained)+i-1;after=len(retained)+i;assert (step['layer_cache_lengths'] in ([],[0]*36) if not i else step['layer_cache_lengths']==[prev]*36) and step['layer_cache_lengths_after']==[after]*36 and step['rope_delta']==[[delta+1024-k]] and step['cache_position']==(list(range(len(retained))) if not i else [prev]) and step['position_ids']==(rec['prefill_position_ids'] if not i else [[[prev+delta+1024-k]]]*3) and len(layers)==36
    for layer in layers:
     assert set(layer)=={'key','value'}
     for z in layer.values():assert z['dtype']=='torch.bfloat16' and z['shape']==[1,8,after,128] and re.fullmatch('[0-9a-f]{64}',z['sha256'])
   path=Path(rec['arrays']['path']).resolve();assert path.is_relative_to((J/'arrays').resolve()) and str(path) not in array_hash and rec['arrays']['kind']=='full_arrays';array_hash[str(path)]=sha(path);assert array_hash[str(path)]==rec['arrays']['sha256']
  assert all(fresh.get(x)==light.get(x) for x in fields) and seg[6]['event']=='pair_verified' and seg[6]['comparison']['status']=='passed'
  with np.load(light['arrays']['path'],allow_pickle=False) as left,np.load(fresh['arrays']['path'],allow_pickle=False) as right:
   required={'visual','prefill','prefill_logits',*[f'ds{i}' for i in range(3)],*[f'injected{i}' for i in range(3)],*['processor_'+i for i in ('input_ids','attention_mask','pixel_values','image_grid_thw')]};assert required<=set(left.files)&set(right.files)
   for name in set(left.files)&set(right.files):assert np.array_equal(left[name],right[name])
   assert np.array_equal(left['pruned'],left['visual'][light['spatial_ids']]) and np.array_equal(left['prefill'][0,start:start+k],left['pruned'])
   for i in range(3):assert np.array_equal(left['injected'+str(i)],left['ds'+str(i)][light['spatial_ids']])
  pairs+=1
assert pairs==12 and len(array_hash)==24 and array_hash==formal['array_sha256']
proof={'status':'passed','independent_review':True,'pairs':12,'generations':24,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_claims':False,'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'run_exit_sha256':sha(O/'exit.json'),'snapshot_manifest_sha256':sha(D/'run_001_sources/manifest.json'),'journal_sha256':sha(J/'journal.jsonl'),'array_sha256':array_hash,'reviewer_sha256':sha(__file__),'scope':'3 DEV anchors; 32 output tokens and 31 decode forwards; 36-layer KV fingerprints compared, full cache arrays not archived; prefill compared to E4, old E4 logits unavailable','errors':[]}
with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
