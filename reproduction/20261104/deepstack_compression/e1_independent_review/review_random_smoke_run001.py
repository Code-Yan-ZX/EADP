"""Independent post-run validation of 18 random K256 diagnostic generations."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e6/random_smoke';A=B/'e6/random_smoke_accept';O=D/'model_run001'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(d,script):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid'] and (cwd/c[1]).resolve()==script.resolve();return s
def choose(image,seed):
 s=int.from_bytes(hashlib.sha256(('eadp-e6-random-v1|'+str(seed)+'|'+image).encode()).digest()[:8],'big');counter=0;buf=[];ids=list(range(1024))
 for i in range(1023,0,-1):
  bound=i+1;limit=(1<<64)-((1<<64)%bound)
  while True:
   if not buf:
    block=hashlib.sha256(s.to_bytes(8,'big')+counter.to_bytes(8,'big')).digest();counter+=1;buf=[int.from_bytes(block[j:j+8],'big') for j in range(0,32,8)]
   value=buf.pop(0)
   if value<limit:break
  j=value%bound;ids[i],ids[j]=ids[j],ids[i]
 return s,sorted(ids[:256])
def array(a,meta):
 widths={'torch.bfloat16':2,'torch.int64':8,'torch.float32':4,'torch.bool':1};shape=list(meta['shape']);shape[-1]*=widths[meta['dtype']];assert a.dtype==np.uint8 and list(a.shape)==shape and hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==meta['sha256']
def main(out):
 term(D/'monitor_model_run001',D/'model_launch.py');term(A/'monitor_verify001',A/'verify.py');term(A/'monitor_freeze001',A/'freeze.py');term(I/'monitor_random_model_final',I/'review_random_model_final.py')
 formal=read(A/'verification001.json');plan=read(D/'execution_plan009.json');vp=read(A/'verification_plan001.json');assert formal['status']=='passed' and (formal['questions'],formal['arms'],formal['natural'],formal['forced33'])==(3,18,9,9) and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['timing_eligible'] is False and formal['verifier_sha256']==sha(A/'verify.py') and formal['verification_plan_sha256']==sha(A/'verification_plan001.json') and formal['execution_plan_sha256']==sha(D/'execution_plan009.json') and formal['run_started_sha256']==sha(D/'monitor_model_run001/started.json') and formal['run_terminal_sha256']==sha(D/'monitor_model_run001/terminal.json')
 for n,h in vp['source_sha256'].items():assert sha(n)==h
 for n,h in plan['source_sha256'].items():assert sha(n)==h
 for n,h in formal['files_sha256'].items():assert sha(n)==h
 exit=read(O/'exit.json');cid=exit['container_identity']['container_id'];doc=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True,timeout=15).stdout)[0];assert exit['returncode']==exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['timed_out'] is False and exit['interrupted'] is None and exit['gpu_pids_after']==[] and doc['Id']==cid and doc['State']['Running'] is False and doc['State']['Pid']==0 and doc['State']['ExitCode']==0 and doc['State']['OOMKilled'] is False and doc['HostConfig']['NetworkMode']=='none' and doc['Config']['Cmd']==[str(D/'model_worker.py'),'--out',str(O)]
 ready=read(O/'model_ready.json');provenance=read(O/'completion_provenance.json');assert provenance['status']=='passed' and provenance['source_sha256']==plan['source_sha256'] and ready['weight_stats_before']==provenance['weight_stats_before']==provenance['weight_stats_after'] and len(provenance['weight_stats_before'])==4 and ready['timing_eligible'] is False
 anchors=read(D/'plan001.json')['anchors'];assert len(anchors)==len({a['task'] for a in anchors})==3;keys={(x['task'],str(x['qid'])):x for x in anchors};e1=read(B/'e1/e1full001_completion.json');rawpath=B/'e1/e1full001/raw_predictions.jsonl';assert sha(rawpath)==e1['files']['journal']['sha256'];refs={}
 for line in rawpath.open():
  r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
  if key in keys and r.get('arm')=='P3-D1' and r.get('forced') is False:assert key not in refs;refs[key]=r
 assert len(refs)==3
 lines=lambda p:[json.loads(x) for x in p.open()];events=lines(O/'journal.jsonl');generated=lines(O/'generated.jsonl');records=lines(O/'records.jsonl');assert len(events)==36 and len(generated)==len(records)==18
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',local_files_only=True,trust_remote_code=False)
 hashes={};count=0
 for a in anchors:
  for seed in (20261104,20261105,20261106):
   derived,ids=choose(a['image_sha256'],seed)
   for forced in (False,True):
    i=count;count+=1;rec=records[i];r=refs[(a['task'],str(a['qid']))];job={'task':a['task'],'qid':str(a['qid']),'group_id':a['group_id'],'seed':seed,'forced':forced};assert events[2*i]=={'event':'started',**job} and events[2*i+1]=={'event':'recorded',**job} and generated[i]['scheduled']==job
    raw=generated[i]['raw'];assert all(raw.get(k)==rec.get(k)==v for k,v in job.items()) and all(raw.get(k)==rec.get(k) for k in ('output_ids','eos_ids','generation_scalars','protocol'))
    assert rec['budget']==256 and rec['ds_on'] is True and rec['selection_mode']=='e6_random_fixed_seed' and rec['greedy_calls']==1 and rec['extra_author_greedy'] is True and rec['timing_eligible'] is False and rec['actual_injection_verified'] is True and rec['derived_seed64']==derived and rec['ordered_ids']==rec['spatial_ids']==ids
    assert all(rec[x]==a[x] for x in ('task','group_id','image_sha256','message_sha256')) and all(rec[x]==r[y] for x,y in [('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),('full_position_ids','original_position_ids'),('full_delta','full_rope_delta')])
    start=rec['image_start'];n=rec['original_length'];ret=list(range(start))+[start+x for x in ids]+list(range(start+1024,n));pos=rec['full_position_ids'];assert rec['retained_ids']==ret and rec['prefill_position_ids']==[[[axis[0][x] for x in ret]] for axis in pos];delta=max(max(axis[0]) for axis in pos)+1-n;assert rec['full_delta']==[[delta]] and rec['adjusted_delta']==[[delta+768]]
    mask=[False]*len(ret);mask[start:start+256]=[True]*256;assert rec['injected_mask']==[mask]
    outputs=rec['output_ids'];assert outputs and all(type(x) is int for x in outputs) and tok.decode(outputs,skip_special_tokens=True,clean_up_tokenization_spaces=False)==rec['answer'];assert rec['generation_scalars']==dict(r['generation_scalars'],max_new_tokens=33 if forced else 2048)
    if forced:assert rec['protocol']=='forced33' and outputs==[100]*33 and rec['eos_ids']==[]
    else:assert rec['protocol']=='natural' and rec['eos_ids']==r['eos_ids'] and (outputs[-1] in rec['eos_ids'] or len(outputs)==2048)
    trace=rec['decode_trace'];assert len(trace)==len(outputs)
    for j,step in enumerate(trace):
     prev=0 if not j else len(ret)+j-1;after=len(ret)+j;assert (step['layer_cache_lengths'] in ([],[0]*36) if not j else step['layer_cache_lengths']==[prev]*36) and step['layer_cache_lengths_after']==[after]*36 and step['rope_delta']==[[delta+768]] and step['cache_position']==(list(range(len(ret))) if not j else [prev]) and step['position_ids']==(rec['prefill_position_ids'] if not j else [[[prev+delta+768]]]*3)
    if forced:
     cache=rec['cache_fingerprints'];assert len(cache)==33
     for j,layers in enumerate(cache):
      assert len(layers)==36
      for layer in layers:
       assert set(layer)=={'key','value'}
       for v in layer.values():assert v['dtype']=='torch.bfloat16' and v['shape']==[1,8,len(ret)+j,128] and re.fullmatch('[0-9a-f]{64}',v['sha256'])
    path=Path(rec['arrays']['path']).resolve();assert path.is_relative_to((O/'arrays').resolve()) and str(path) not in hashes;h=sha(path);assert h==rec['arrays']['sha256']==formal['array_sha256'][str(path)];hashes[str(path)]=h
    with np.load(path,allow_pickle=False) as z:
     required={'visual','prefill','prefill_logits',*[f'ds{x}' for x in range(3)],*[f'injected{x}' for x in range(3)]};required|=({'processor_'+x for x in ('input_ids','attention_mask','pixel_values','image_grid_thw')} if forced else set());assert set(z.files)==required
     for name,meta in [('visual',rec['main_visual']),('prefill',rec['prefill_embeds']),('prefill_logits',rec['prefill_logits'])]+[(f'ds{x}',rec['full_deepstack'][x]) for x in range(3)]+[(f'injected{x}',rec['injected_deepstack'][x]) for x in range(3)]+([('processor_'+x,rec['processor'][x]) for x in rec['processor']] if forced else []):array(z[name],meta)
     chosen=z['visual'][ids];array(chosen,rec['pruned_visual']);assert np.array_equal(z['prefill'][0,start:start+256],chosen)
     for j in range(3):array(z[f'ds{j}'][ids],rec['selected_deepstack'][j]);assert np.array_equal(z[f'ds{j}'][ids],z[f'injected{j}'])
 assert count==18 and len(hashes)==18
 proof={'status':'passed','independent_review':True,'questions':3,'arms':18,'natural':9,'forced33':9,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'timing_claims':False,'formal_sha256':sha(A/'verification001.json'),'formal_started_sha256':sha(A/'monitor_verify001/started.json'),'formal_terminal_sha256':sha(A/'monitor_verify001/terminal.json'),'run_started_sha256':sha(D/'monitor_model_run001/started.json'),'run_terminal_sha256':sha(D/'monitor_model_run001/terminal.json'),'array_sha256':hashes,'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Three posthoc DEV anchors; diagnostic only, not a quality estimate.','The full author greedy pass is an online check; separate greedy IDs are not archived.','Forced KV tensors only have runtime fingerprints; no raw KV tensor storage.','Weight stat equality cannot rule out transient replacement.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==I/'random_smoke_run001_review.json' and not a.out.exists();main(a.out)
