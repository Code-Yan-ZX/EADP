"""Independent 16-arm light adapter review, no labels or performance claim."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;C=ROOT/'e4_c_timing';RUN=C/'light_smoke_003'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def supervise(start,end,script,out=None):
 s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
COMMON=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual','image_start','original_length','full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars','forced','forced_token','decode_trace','block_calls','main_merger_calls','ds_calls')
def verify_pair(actual,reference,mode,missing_logits,missing_kv):
 assert all(k in actual and k in reference and actual[k]==reference[k] for k in COMMON)
 expected=reference.get('prefill_logits') if mode=='C-ON' else reference.get('first_logits')
 assert actual.get('prefill_logits') is not None
 if expected is None:assert missing_logits and mode=='C-ON' and actual['forced'] is False
 else:assert actual['prefill_logits']==expected
 kv=reference.get('cache_fingerprints')
 if kv is None:assert missing_kv and mode=='C-ON'
 else:assert actual['cache_fingerprints']==kv
 if mode=='C-ON':assert all(actual[k]==reference[k] and actual[k] is not None for k in ('injected_mask','injected_deepstack'))
 else:assert actual.get('injected_mask') is None and actual.get('injected_deepstack') is None
 left,right=actual['arrays'],reference['arrays'];assert sha(left['path'])==left['sha256'] and sha(right['path'])==right['sha256']
 with np.load(left['path'],allow_pickle=False) as a,np.load(right['path'],allow_pickle=False) as b:
  common=set(a.files)&set(b.files);assert {'visual','prefill'}<=common
  if not missing_logits:assert 'prefill_logits' in common
  for name in common:assert np.array_equal(a[name],b[name]),name
 return len(common)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e4c_light_smoke003_review.json'
 proof=C/'light_smoke_003_verification.json';fs=C/'monitor_light_verify003/started.json';ft=fs.parent/'terminal.json';rs=C/'monitor_light_smoke003/started.json';rt=rs.parent/'terminal.json';planpath=C/'light_gpu_smoke_launch_plan003.json'
 supervise(fs,ft,C/'light_verify.py',proof);started,terminal=supervise(rs,rt,C/'light_smoke_launch.py')
 f=read(proof);assert f['status']=='passed' and f['generations']==16 and f['lightweight_generations']==12 and f['comparisons']==16 and f['errors']==[] and all(f[k] is False for k in ('GT_opened','scoring_performed','timing_claims')) and f['verifier_sha256']==sha(C/'light_verify.py')
 for field in ('evidence_sha256','array_sha256'):
  for path,digest in f[field].items():assert sha(path)==digest
 assert len(f['array_sha256'])==16
 plan=read(planpath);sources=f['source_sha256'];assert sources==plan['approval_bundle']['source_sha256'] and set(sources)==set(plan['source_snapshots'])
 for path,digest in sources.items():
  snap=plan['source_snapshots'][path];assert snap['sha256']==digest and sha(snap['snapshot'])==digest
 launch=read(RUN/'launch.json');exit=read(RUN/'exit.json');assert exit['controller']==launch['controller'] and exit['controller']['pid']==terminal['worker_pid']
 assert exit['returncode']==exit['container_exit_code']==0 and exit['container_stopped_verified'] is True and exit['client_exited_while_container_running'] is False and exit['timed_out'] is False and exit['interrupted'] is None and exit['failure_type'] is None and exit['gpu_pids_after']==[]
 obj=json.loads(subprocess.check_output(['docker','inspect',exit['container_identity']['container_id']],text=True))[0];assert obj['State']['Running'] is False and obj['State']['Pid']==0 and obj['State']['ExitCode']==0 and obj['Id']==exit['container_identity']['container_id'] and obj['Image']==exit['container_identity']['image']
 reference=read(C/'light_reference001.json');assert reference['status']=='passed';jobs=reference['plan']['jobs'];assert len(jobs)==16
 old={(x['task'],str(x['qid']),x['forced']):x for x in reference['old_on']};b={(x['task'],str(x['qid']),x['forced'],x['mode']):x for x in reference['b_records']};assert len(old)==4 and len(b)==8
 records=lines(RUN/'records.jsonl');generated=lines(RUN/'generated.jsonl');comparisons=lines(RUN/'comparisons.jsonl');assert len(records)==len(generated)==len(comparisons)==16
 from transformers import AutoTokenizer
 recipe=read(ROOT.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')['recipe'];tok=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False);forced_token=tok.encode(' the',add_special_tokens=False)[0];assert forced_token==f['forced_token']
 fresh={};seen=set();array_checks=0
 for job,item,raw,comparison in zip(jobs,records,generated,comparisons):
  assert item['job']==comparison['job']==job;r=item['record'];mode=job['path'];key=(job['task'],str(job['qid']),job['forced'])
  for k in ('task','qid','group_id','forced','output_ids','eos_ids','generation_scalars'):assert raw[k]==r[k]
  assert raw['path']==mode and all(r[k]==job[k] for k in ('task','qid','group_id','forced'))
  path=Path(r['arrays']['path']).resolve();assert path.parent==(RUN/'arrays').resolve() and path not in seen and f['array_sha256'][str(path)]==sha(path)==r['arrays']['sha256'];seen.add(path)
  ids=r['output_ids'];assert ids and len(ids)<=r['max_new_tokens'] and tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer']
  eos=ids[-1] in r['eos_ids'];assert r['eos']==eos and r['truncated']==(not r['forced'] and not eos and len(ids)==2048)
  if r['forced']:assert ids==[forced_token]*33 and r['eos_ids']==[]
  else:assert eos or len(ids)==2048
  s,L=r['image_start'],r['original_length'];order=r['ordered_ids'];spatial=r['spatial_ids'];assert len(order)==len(set(order))==256 and all(type(x) is int and 0<=x<1024 for x in order) and spatial==sorted(order)
  retained=list(range(s))+[s+x for x in spatial]+list(range(s+1024,L));n=len(retained);full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,L);delta=int(full.max())+1-L
  assert r['retained_ids']==retained and r['full_delta']==[[delta]] and r['adjusted_delta']==[[delta+768]] and r['prefill_position_ids']==full[:,retained][:,None,:].tolist()
  trace=r['decode_trace'];assert len(trace)==len(ids)
  for step,t in enumerate(trace):
   pos=n+step-1;assert t['rope_delta']==[[delta+768]] and t['cache_position']==(list(range(n)) if step==0 else [pos]) and t['layer_cache_lengths'] in (([],[0]*36) if step==0 else ([pos]*36,)) and t['layer_cache_lengths_after']==[n+step]*36
   assert t['position_ids']==(r['prefill_position_ids'] if step==0 else [[[pos+delta+768]]]*3)
  kv=r['cache_fingerprints'];assert len(kv)==(len(ids) if r['forced'] else 1)
  for step,layers in enumerate(kv):
   assert len(layers)==36
   for layer in layers:
    for name in ('key','value'):
     meta=layer[name];assert meta['dtype']=='torch.bfloat16' and meta['shape']==[1,8,n+step,128] and len(meta['sha256'])==64 and all(x in '0123456789abcdef' for x in meta['sha256'])
  with np.load(path,allow_pickle=False) as z:
   def same(name,meta):assert hashlib.sha256(z[name].tobytes()).hexdigest()==meta['sha256']
   same('visual',r['main_visual']);same('prefill',r['prefill_embeds']);assert np.array_equal(z['prefill'][0,s:s+256],z['visual'][spatial])
   if 'pruned' in z:same('pruned',r['pruned_visual']);assert np.array_equal(z['pruned'],z['visual'][spatial])
   if 'prefill_logits' in z:
    same('prefill_logits',r['prefill_logits']);dtype=r['prefill_logits']['dtype'];bits=z['prefill_logits'].tobytes()
    assert ((np.frombuffer(bits,dtype='<u2')&0x7f80)!=0x7f80).all() if dtype=='torch.bfloat16' else np.isfinite(np.frombuffer(bits,dtype='<f4')).all()
   for name,meta in r['processor'].items():
    if 'processor_'+name in z:same('processor_'+name,meta)
   is_on=mode in ('fresh_on_reference','C-ON')
   if is_on:
    assert r['injected_mask']==[[False]*s+[True]*256+[False]*(n-s-256)]
    if mode=='fresh_on_reference':assert r['actual_injection_verified'] is True
    for j in range(3):same('ds'+str(j),r['full_deepstack'][j]);same('injected'+str(j),r['injected_deepstack'][j]);assert np.array_equal(z['injected'+str(j)],z['ds'+str(j)][spatial])
   else:assert r['injected_mask'] is None and r['injected_deepstack'] is None
  if mode=='fresh_on_reference':fresh[key]=r;actual=dict(r,mode='C-ON',ds_none=[False]*3);ref=old[key];missing=ref.get('prefill_logits') is None;missing_kv=True
  else:
   assert r['mode']==mode and r['ds_none']==[mode=='C-SKIP']*3
   actual=r;ref=fresh[key] if mode=='C-ON' else b[(*key,mode)];missing=False;missing_kv=False
  array_checks+=verify_pair(actual,ref,'C-ON' if mode=='fresh_on_reference' else mode,missing,missing_kv)
  assert comparison['metadata']['status']=='identity_passed_bitwise'
 assert len(seen)==16 and len(fresh)==4
 result={'status':'passed','independent_review':True,'generations':16,'lightweight_generations':12,'comparisons':16,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'errors':[],
  'formal_proof_sha256':sha(proof),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'run_started_sha256':sha(rs),'run_terminal_sha256':sha(rt),'launch_plan_sha256':sha(planpath),'reviewer_sha256':sha(__file__),
  'array_tensor_comparisons':array_checks,'scope':'3 DEV natural anchors plus one 32-step forced anchor; no GT or timing',
  'limitations':['Prior E4 ON reference lacks KV fingerprints; only fresh ON versus lightweight ON compares them.','KV content retained as per-layer hashes; full KV values are not archived.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','generations':16,'array_tensor_comparisons':array_checks}))
if __name__=='__main__':main()
