"""Independent completed Div/CD full-DEV review. No labels or scoring."""
import argparse,collections,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'e6/selected_full';RUN=HERE/'panel_001';sys.path.insert(0,str(HERE))
from full_imports import load
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def supervised(start,end,script,out=None):
 s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
def key(x):return (x['method'],x['task'],str(x['qid']),x['group_id'],x['path'],x['forced'])
def meta(x):return isinstance(x,dict) and set(x)=={'dtype','shape','sha256'} and isinstance(x['shape'],list) and bool(x['shape']) and all(type(i) is int and i>0 for i in x['shape']) and isinstance(x['sha256'],str) and len(x['sha256'])==64 and all(c in '0123456789abcdef' for c in x['sha256'])
def check_trace(r,delta,n):
 ids=r['output_ids'];trace=r['decode_trace'];assert len(trace)==len(ids)
 for i,t in enumerate(trace):
  p=n+i-1;assert t['rope_delta']==[[delta]] and t['cache_position']==(list(range(n)) if i==0 else [p]) and t['layer_cache_lengths'] in (([],[0]*36) if i==0 else ([p]*36,)) and t['layer_cache_lengths_after']==[n+i]*36 and t['position_ids']==(r['prefill_position_ids'] if i==0 else [[[p+delta]]]*3)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();assert args.out.resolve()==ROOT/'e1_independent_review/selected_full_panel001_review.json'
 proof=HERE/'panel_001_verification.json';fs=HERE/'monitor_verify001/started.json';ft=fs.parent/'terminal.json';rs=HERE/'monitor_panel001/started.json';rt=rs.parent/'terminal.json'
 supervised(fs,ft,HERE/'verify.py',proof);start,end=supervised(rs,rt,HERE/'launcher.py');formal=read(proof)
 assert formal['status']=='passed' and formal['stage']=='full_dev' and formal['questions']==384 and formal['generations']==1536 and formal['pairs']==768 and formal['GT_opened'] is False and formal['scoring'] is False and formal['verifier_sha256']==sha(HERE/'verify.py')
 for path,digest in formal['files_sha256'].items():assert sha(path)==digest
 snap=HERE/'panel_001_sources';assert formal['source_manifest_sha256']==sha(snap/'manifest.json');sources=read(snap/'manifest.json')['source_sha256'];assert len(sources)==126
 for path,digest in sources.items():assert sha(path)==sha(snap/'files'/str(Path(path)).lstrip('/'))==digest
 launch,ex=read(RUN/'launch.json'),read(RUN/'exit.json');assert ex['controller']==launch['controller'] and ex['controller']['pid']==end['worker_pid'] and ex['returncode']==ex['container_exit_code']==0 and ex['container_stopped_verified'] is True and ex['client_exited_while_container_running'] is False and ex['timed_out'] is False and ex['interrupted'] is None and ex['errors']==[] and ex['gpu_pids_after']==[]
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],text=True))[0];load('independent_full_docker_identity',HERE/'container_identity.py').validate(RUN,launch,ex,obj)
 frozen=read(RUN/'frozen.json');done=read(RUN/'worker_complete.json');assert frozen['GT_opened'] is False and frozen['scoring'] is False and frozen['full_tensors_archived'] is False and done['status']=='completed' and done['questions']==384 and done['generations']==1536
 e1=read(ROOT/'e1/e1full001_completion.json');inp=Path(e1['files']['inputs']['path']);journal=Path(e1['files']['journal']['path']);assert frozen['e1_completion_sha256']==sha(ROOT/'e1/e1full001_completion.json') and sha(inp)==e1['files']['inputs']['sha256'] and sha(journal)==e1['files']['journal']['sha256']
 rows=lines(inp);refs=[x for x in lines(journal) if x.get('arm')=='P3-D1' and x.get('forced') is False];lookup={(r['task'],str(r['qid']),r['group_id']):r for r in rows};ref={(r['task'],str(r['qid']),r['group_id']):r for r in refs}
 assert len(rows)==len(lookup)==len(refs)==len(ref)==384 and collections.Counter(r['task'] for r in rows)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 expected={(m,r['task'],str(r['qid']),r['group_id'],path,False) for r in rows for m in ('div','cd') for path in ('author_stock','adapted')};assert len(expected)==1536
 records=lines(RUN/'records.jsonl');markers=lines(RUN/'generated.jsonl');events=lines(RUN/'events.jsonl');pairs=lines(RUN/'pairs.jsonl');assert len(records)==len(markers)==1536 and len(pairs)==768
 rec={key(x):x for x in records};gen={key(x):x for x in markers};assert set(rec)==set(gen)==expected
 for typ in ('arm_started','arm_completed'):
  marked=[key(x) for x in events if x['event']==typ];assert len(marked)==1536 and set(marked)==expected
 assert events[-1]['event']=='worker_completed' and not any(x['event']=='worker_failed' for x in events)
 starts={key(x):i for i,x in enumerate(events) if x['event']=='arm_started'};ends={key(x):i for i,x in enumerate(events) if x['event']=='arm_completed'};assert all(starts[k]<ends[k] for k in expected)
 controls=load('independent_full_generation_reference',HERE/'generation_reference.py').controls()
 from transformers import AutoTokenizer
 plan=read(ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tok=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 for k,r in rec.items():
  marker=gen[k];row=lookup[k[1:4]];reference=ref[k[1:4]];assert all(marker.get(q)==r.get(q) for q in ('output_ids','eos_ids','generation_scalars','generation_controls')) and r['generation_controls']==controls[k[0]][k[1]]
  assert all(r[q]==row[q] for q in ('group_id','image_sha256','message_sha256')) and r['processor']==reference['processor'] and r['main_visual']==reference['visual_sha256'] and r['full_deepstack']==reference['deepstack_full_sha256'] and r['generation_scalars']==reference['generation_scalars'] and r['eos_ids']==reference['eos_ids']
  ids=r['output_ids'];assert ids and len(ids)<=2048 and all(type(x) is int for x in ids);eos=ids[-1] in r['eos_ids'];assert r['eos']==eos and r['truncated']==(not eos and len(ids)==2048) and (eos or r['truncated']) and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer']
  selected=r['spatial_ids'];s,L=r['image_start'],r['original_length'];assert len(selected)==256 and selected==sorted(set(selected)) and all(type(x) is int and 0<=x<1024 for x in selected)
  keep=list(range(s))+[s+x for x in selected]+list(range(s+1024,L));n=len(keep);assert r['retained_ids']==keep and r['arrays']['kind']=='summary_only' and all(meta(x) for x in r['arrays']['tensors'].values())
  full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);delta=int(full.max())+1-L;assert r['full_delta']==[[delta]] and r['full_position_ids']==reference['original_position_ids'] and r['full_delta']==reference['full_rope_delta']
  if r['path']=='author_stock':assert r['rope_delta']==[[0]] and r['prefill_position_ids']==[[list(range(n))]]*3 and r['injected_mask'] is None and r['injected_deepstack'] is None;delta=0
  else:
   delta+=768;assert r['adjusted_delta']==[[delta]] and r['prefill_position_ids']==full[:,keep][:,None,:].tolist() and r['selection_mode']=='author_'+r['method'] and r['ordered_ids']==selected and r['selector_calls']==1 and r['greedy_calls']==0 and r['actual_injection_verified'] is True
   assert r['injected_mask']==[[False]*s+[True]*256+[False]*(n-s-256)] and len(r['injected_deepstack'])==3 and r['injected_deepstack']==r['selected_deepstack']
  check_trace(r,delta,n)
 common=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','full_deepstack','pruned_visual','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_embeds','prefill_mask','generation_scalars','generation_controls')
 for row in rows:
  for m in ('div','cd'):
   k=(m,row['task'],str(row['qid']),row['group_id']);s=rec[k+('author_stock',False)];a=rec[k+('adapted',False)];assert all(s[x]==a[x] for x in common)
 pairkeys={(m,row['task'],str(row['qid']),row['group_id']) for row in rows for m in ('div','cd')};assert len(pairkeys)==768 and {(p['method'],p['task'],str(p['qid']),p['group_id']) for p in pairs}==pairkeys and all(p['status']=='passed' for p in pairs)
 out={'status':'passed','independent_review':True,'questions':384,'generations':1536,'pairs':768,'GT_opened':False,'scoring_performed':False,'errors':[],
      'formal_proof_sha256':sha(proof),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'run_started_sha256':sha(rs),'run_terminal_sha256':sha(rt),'source_manifest_sha256':sha(snap/'manifest.json'),'reviewer_sha256':sha(__file__),
      'limits':['Full-panel high-dimensional tensors are not archived and cannot be independently reconstructed; accepted smoke contains full anchor arrays.','Actual KV lengths are recorded; full KV content is not archived.','Stock P1/D0 and adapted P3/D1 outputs are not required to be equal.']}
 with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':384,'generations':1536,'pairs':768}))
if __name__=='__main__':main()
