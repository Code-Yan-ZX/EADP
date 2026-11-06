"""Independent Div/CD 16-arm diagnostic review, no GT or score."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'e6/selected_smoke';RUN=HERE/'smoke_001';sys.path.insert(0,str(HERE))
from local_imports import pinned
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f]
def supervised(start,end,script,out=None):
 s,t=read(start),read(end);cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def key(r):return (r['method'],r['task'],str(r['qid']),r['group_id'],r['path'],r['forced'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/selected_smoke001_review.json'
 proof=HERE/'smoke_001_verification.json';fs=HERE/'monitor_verify001/started.json';ft=fs.parent/'terminal.json';rs=HERE/'monitor_smoke001/started.json';rt=rs.parent/'terminal.json'
 supervised(fs,ft,HERE/'verify.py',proof);start,end=supervised(rs,rt,HERE/'launcher.py');formal=read(proof)
 assert formal['status']=='passed' and formal['questions']==3 and formal['generations']==16 and formal['pairs']==8 and formal['forced_decode_steps']==32 and formal['GT_opened'] is False and formal['scoring'] is False and formal['verifier_sha256']==sha(HERE/'verify.py')
 for path,digest in formal['files_sha256'].items():assert sha(path)==digest
 snap=HERE/'smoke_001_sources';assert formal['source_manifest_sha256']==sha(snap/'manifest.json');sources=read(snap/'manifest.json')['source_sha256'];assert len(sources)==102
 for path,digest in sources.items():assert sha(path)==sha(snap/'files'/str(Path(path)).lstrip('/'))==digest
 launch,ex=read(RUN/'launch.json'),read(RUN/'exit.json');assert ex['controller']==launch['controller'] and ex['controller']['pid']==end['worker_pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['timed_out'] is False and ex['interrupted'] is None and ex['errors']==[] and ex['gpu_pids_after']==[]
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];pinned('_independent_selected_container',HERE/'container_identity.py').validate(RUN,launch,ex,obj)
 plan=read(HERE/'plan.json');anchors=plan['anchors'];expected={(m,x['task'],str(x['qid']),x['group_id'],p,f) for m in ('div','cd') for x in anchors for p in ('author_stock','adapted') for f in ((False,True) if x['task']=='DocVQA_VAL' else (False,))};assert len(expected)==16
 records=lines(RUN/'records.jsonl');gen=lines(RUN/'generated.jsonl');events=lines(RUN/'events.jsonl');comparisons=lines(RUN/'comparisons.jsonl');assert len(records)==len(gen)==16 and len(comparisons)==8
 rec={key(x):x for x in records};generated={key(x):x for x in gen};assert set(rec)==set(generated)==expected
 for typ in ('arm_started','arm_complete'):
  markers=[key(x) for x in events if x['event']==typ];assert len(markers)==16 and set(markers)==expected
 assert events[-1]['event']=='worker_complete' and not any(x['event']=='worker_failed' for x in events)
 for k in expected:
  positions=[i for i,x in enumerate(events) if x['event'] in ('arm_started','arm_complete') and key(x)==k];assert len(positions)==2 and events[positions[0]]['event']=='arm_started' and events[positions[1]]['event']=='arm_complete'
 e1=read(ROOT/'e1/e1full001_completion.json');inputpath=Path(e1['files']['inputs']['path']);journal=Path(e1['files']['journal']['path']);assert sha(inputpath)==e1['files']['inputs']['sha256'] and sha(journal)==e1['files']['journal']['sha256']
 inputs={(r['task'],str(r['qid'])):r for r in lines(inputpath)};erefs={(r['task'],str(r['qid'])):r for r in lines(journal) if r.get('arm')=='P3-D1' and r.get('forced') is False};assert len(inputs)==len(erefs)==384
 from transformers import AutoTokenizer
 author=read(ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tok=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False);forced_token=tok.encode(' the',add_special_tokens=False)[0]
 seen=set();checks=0
 for k,r in rec.items():
  marker=generated[k];assert all(marker.get(x)==r.get(x) for x in ('output_ids','eos_ids','generation_scalars','generation_controls'))
  inp=inputs[(r['task'],str(r['qid']))];ref=erefs[(r['task'],str(r['qid']))];assert all(r[x]==inp[x] for x in ('group_id','image_sha256','message_sha256')) and r['processor']==ref['processor'] and r['main_visual']==ref['visual_sha256'] and r['full_deepstack']==ref['deepstack_full_sha256']
  ids=r['output_ids'];forced=r['forced'];eos=ids[-1] in r['eos_ids'];assert ids and len(ids)<=(33 if forced else 2048) and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'] and r['eos']==eos and r['truncated']==(not forced and not eos and len(ids)==2048)
  if forced:assert ids==[forced_token]*33 and r['forced_token']==forced_token and r['eos_ids']==[]
  else:assert eos or len(ids)==2048
  index=r['spatial_ids'];s,L=r['image_start'],r['original_length'];assert len(index)==256 and index==sorted(set(index)) and all(type(x) is int and 0<=x<1024 for x in index)
  retained=list(range(s))+[s+x for x in index]+list(range(s+1024,L));n=len(retained);assert retained==r['retained_ids']
  full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);d=int(full.max())+1-L;assert r['full_delta']==[[d]]
  if r['path']=='adapted':
   delta=d+768;assert r['adjusted_delta']==[[delta]] and r['prefill_position_ids']==full[:,retained][:,None,:].tolist() and r['selection_mode']=='author_'+r['method'] and r['ordered_ids']==index and r['selector_calls']==1 and r['greedy_calls']==0 and r['actual_injection_verified'] is True
  else:
   delta=0;assert r['rope_delta']==[[0]] and r['prefill_position_ids']==[[list(range(n))]]*3 and r['injected_mask'] is None and r['injected_deepstack'] is None
  trace=r['decode_trace'];assert len(trace)==len(ids)
  for i,t in enumerate(trace):
   pos=n+i-1;assert t['rope_delta']==[[delta]] and t['cache_position']==(list(range(n)) if i==0 else [pos]) and t['layer_cache_lengths'] in (([],[0]*36) if i==0 else ([pos]*36,)) and t['layer_cache_lengths_after']==[n+i]*36
   assert t['position_ids']==(r['prefill_position_ids'] if i==0 else [[[pos+delta]]]*3)
  kv=r['cache_fingerprints'];assert len(kv)==(33 if forced else 1)
  for i,layers in enumerate(kv):
   assert len(layers)==36
   for layer in layers:
    for name in ('key','value'):
     z=layer[name];assert z['dtype']=='torch.bfloat16' and z['shape']==[1,8,n+i,128] and len(z['sha256'])==64 and all(ch in '0123456789abcdef' for ch in z['sha256'])
  descriptor=r['arrays'];path=Path(descriptor['path']).resolve();assert path.parent==(RUN/'arrays').resolve() and path not in seen and sha(path)==descriptor['sha256'];seen.add(path)
  with np.load(path,allow_pickle=False) as z:
   def same(name,meta):assert hashlib.sha256(z[name].tobytes()).hexdigest()==meta['sha256']
   same('visual',r['main_visual']);same('prefill',r['prefill_embeds']);assert np.array_equal(z['prefill'][0,s:s+256],z['visual'][index])
   if 'pruned' in z:same('pruned',r['pruned_visual']);assert np.array_equal(z['pruned'],z['visual'][index])
   for j in range(3):same('ds'+str(j),r['full_deepstack'][j])
   for name,meta in r['processor'].items():
    if 'processor_'+name in z:same('processor_'+name,meta)
   same('prefill_logits',r['prefill_logits']);bits=z['prefill_logits'].tobytes();dtype=r['prefill_logits']['dtype'];assert dtype=='torch.bfloat16' and ((np.frombuffer(bits,dtype='<u2')&0x7f80)!=0x7f80).all()
   if r['path']=='adapted':
    assert r['injected_mask']==[[False]*s+[True]*256+[False]*(n-s-256)]
    for j in range(3):same('injected'+str(j),r['injected_deepstack'][j]);assert np.array_equal(z['injected'+str(j)],z['ds'+str(j)][index])
   else:
    digests=[hashlib.sha256(x.tobytes()).digest() for x in z['visual']];assert all(digests.count(digests[x])==1 for x in index)
  checks+=1
 common=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','full_deepstack','pruned_visual','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_embeds','prefill_mask','generation_scalars','generation_controls')
 for m in ('div','cd'):
  for row in anchors:
   for forced in ((False,True) if row['task']=='DocVQA_VAL' else (False,)):
    prefix=(m,row['task'],str(row['qid']),row['group_id']);stock,adapt=rec[prefix+('author_stock',forced)],rec[prefix+('adapted',forced)];assert all(stock[x]==adapt[x] for x in common)
    if forced:
     for path in ('author_stock','adapted'):
      natural,forcedr=rec[prefix+(path,False)],rec[prefix+(path,True)]
      keys=('processor','main_visual','full_deepstack','pruned_visual','spatial_ids','retained_ids','prefill_embeds','prefill_mask','prefill_position_ids','prefill_logits')
      assert all(natural[x]==forcedr[x] for x in keys) and natural['cache_fingerprints'][0]==forcedr['cache_fingerprints'][0]
 assert len(seen)==checks==16
 result={'status':'passed','independent_review':True,'questions':3,'generations':16,'pairs':8,'forced_decode_steps':32,'GT_opened':False,'scoring_performed':False,'errors':[],
         'formal_proof_sha256':sha(proof),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'run_started_sha256':sha(rs),'run_terminal_sha256':sha(rt),'source_manifest_sha256':sha(snap/'manifest.json'),'reviewer_sha256':sha(__file__),
         'limitations':['Stock P1/D0 and adapted P3/D1 outputs are not asserted equal.','KV content is represented by per-layer fingerprints, not archived raw KV tensors.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','generations':16,'pairs':8}))
if __name__=='__main__':main()
