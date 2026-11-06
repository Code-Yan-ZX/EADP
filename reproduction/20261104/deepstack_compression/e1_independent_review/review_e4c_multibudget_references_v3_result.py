"""Independent post-run rebuild of 62 DEV inputs x 4 fixed32 references."""
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e4_c_multibudget_references_v3';A=B/'e4_c_multibudget_references_accept_v3';O=D/'run_001';J=O/'diagnostic'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(folder,script):
 s=read(folder/'started.json');t=read(folder/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and (cwd/c[1]).resolve()==script.resolve();return s
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
FIELDS=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','greedy_calls','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','injected_mask','injected_deepstack','block_calls','main_merger_calls','ds_calls')
def main(out):
 term(D/'monitor_run001',D/'launch.py');term(A/'monitor_snapshot001',A/'snapshot.py');term(A/'monitor_verify001',A/'verify.py')
 formal=read(A/'verification001.json');plan=read(D/'execution_plan001.json');vp=read(A/'verification_plan001.json');assert formal['status']=='passed' and formal['inputs']==62 and formal['generations']==248 and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['timing_claims'] is False and formal['verification_plan_sha256']==sha(A/'verification_plan001.json') and formal['verifier_sha256']==sha(A/'verify.py')
 for name,h in vp['source_sha256'].items():assert sha(name)==h
 files={'run_started_sha256':D/'monitor_run001/started.json','run_terminal_sha256':D/'monitor_run001/terminal.json','launch_sha256':O/'launch.json','exit_sha256':O/'exit.json','worker_complete_sha256':O/'worker_complete.json','snapshot_manifest_sha256':D/'run_001_sources/manifest.json','snapshot_started_sha256':A/'monitor_snapshot001/started.json','snapshot_terminal_sha256':A/'monitor_snapshot001/terminal.json','journal_sha256':J/'journal.jsonl','records_sha256':J/'records.jsonl'}
 for k,p in files.items():assert formal[k]==sha(p)
 snap=read(D/'run_001_sources/manifest.json');assert set(snap['sources'])==set(plan['source_sha256'])
 for name,h in plan['source_sha256'].items():
  item=snap['sources'][name];assert item['sha256']==h and sha(item['snapshot'])==h
 exit=read(O/'exit.json');cid=exit['container_identity']['container_id'];doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0]
 assert exit['returncode']==exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['timed_out'] is False and exit['interrupted'] is None and exit['gpu_pids_after']==[] and doc['Id']==cid and doc['State']['Running'] is False and doc['State']['Pid']==0 and doc['State']['ExitCode']==0 and doc['State']['OOMKilled'] is False and doc['HostConfig']['NetworkMode']=='none' and doc['Config']['Cmd']==[str(D/'worker.py'),'--out',str(O)]
 refs=load('_independent_reference_loader',B/'e4_c_multibudget_worker/references.py').load_references();rows=sorted(refs['rows'],key=lambda r:(r['task'],str(r['qid'])));assert len(rows)==62 and len({(r['task'],str(r['qid'])) for r in rows})==len({(r['task'],r['group_id']) for r in rows})==62 and all(r['split']=='DEV' for r in rows)
 records=[json.loads(x) for x in (J/'records.jsonl').open()];events=[json.loads(x) for x in (J/'journal.jsonl').open()];assert len(records)==248 and len(events)==992
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 for index,(row,k) in enumerate((row,k) for row in rows for k in (128,256,512,1024)):
  job={'order':index,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':'fixed32'};e=events[index*4:index*4+4];assert [x['event'] for x in e]==['arm_started','raw_generation','arm_captured','arm_completed'] and all(x['job']==job for x in e)
  rec=records[index];raw=e[1]['raw'];assert e[2]['record']==rec
  assert all(raw.get(z)==rec.get(z) for z in ('task','qid','group_id','budget','ds_on','forced','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens'))
  panel=refs['records'][(row['task'],str(row['qid']),k)];assert panel['actual_injection_verified'] is True and all(z in rec and z in panel and rec[z]==panel[z] for z in FIELDS) and rec['decode_trace'][0]==panel['decode_trace'][0]
  selected=rec['ordered_ids'];sp=rec['spatial_ids'];start=rec['image_start'];n=rec['original_length'];assert len(selected)==len(set(selected))==k and all(type(x) is int and 0<=x<1024 for x in selected) and sp==sorted(selected)
  retained=list(range(start))+[start+x for x in sp]+list(range(start+1024,n));assert rec['retained_ids']==retained;full=rec['full_position_ids'];assert rec['prefill_position_ids']==[[[axis[0][x] for x in retained]] for axis in full]
  delta=max(max(axis[0]) for axis in full)+1-n;assert rec['full_delta']==[[delta]] and rec['adjusted_delta']==[[delta+1024-k]]
  mask=[False]*len(retained);mask[start:start+k]=[True]*k;assert rec['injected_mask']==[mask] and len(rec['injected_deepstack'])==3
  ids=rec['output_ids'];assert rec['protocol']=='fixed32' and rec['forced'] is False and rec['min_new_tokens']==rec['max_new_tokens']==32 and rec['generation_scalars']==dict(panel['generation_scalars'],max_new_tokens=32) and rec['eos_ids']==panel['eos_ids'] and len(ids)==32 and all(type(v) is int for v in ids) and all(v not in rec['eos_ids'] for v in ids[:-1]) and tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==rec['answer']
  trace=rec['decode_trace'];assert len(trace)==32
  for i,step in enumerate(trace):
   before=0 if i==0 else len(retained)+i-1;after=len(retained)+i;assert (step['layer_cache_lengths'] in ([],[0]*36) if i==0 else step['layer_cache_lengths']==[before]*36) and step['layer_cache_lengths_after']==[after]*36 and step['rope_delta']==[[delta+1024-k]] and step['cache_position']==(list(range(len(retained))) if i==0 else [before]) and step['position_ids']==(rec['prefill_position_ids'] if i==0 else [[[before+delta+1024-k]]]*3)
  cache=rec['cache_fingerprints'];assert len(cache)==1 and len(cache[0])==36
  for layer in cache[0]:
   assert set(layer)=={'key','value'}
   for v in layer.values():assert v['dtype']=='torch.bfloat16' and v['shape']==[1,8,len(retained),128] and re.fullmatch('[0-9a-f]{64}',v['sha256'])
  assert rec['arrays']['kind']=='summary_only'
 proof={'status':'passed','independent_review':True,'inputs':62,'generations':248,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_claims':False,'errors':[],'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_run001/terminal.json'),'snapshot_manifest_sha256':sha(D/'run_001_sources/manifest.json'),'records_sha256':sha(J/'records.jsonl'),'reviewer_sha256':sha(__file__),'limitations':['No full prefill or DS tensors archived; summaries and selection/position reconstructed against accepted E4.','Prefill KV hashes recorded at runtime; decode steps 1–31 have per-layer lengths/positions only.','Existing DEV mixed prediction records opened for identity; no GT or scoring.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'e4c_multibudget_references_v3_result_review.json' and not a.out.exists();main(a.out)
