"""Independent no-GT review of actual B identity pair arrays and decode."""
import argparse,hashlib,json,pathlib,re,subprocess
import numpy as np
from transformers import AutoTokenizer
B=pathlib.Path(__file__).resolve().parents[1];D=B/'e4_b_identity';run=D/'b_001'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(pathlib.Path(p).read_text())
rows=lambda p:[json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x]
a=argparse.ArgumentParser();a.add_argument('--formal-proof',type=pathlib.Path,required=True);a.add_argument('--formal-started',type=pathlib.Path,required=True);a.add_argument('--formal-terminal',type=pathlib.Path,required=True);a.add_argument('--out',type=pathlib.Path,required=True);a=a.parse_args();assert not a.out.exists()
formal=load(a.formal_proof);fs=load(a.formal_started);ft=load(a.formal_terminal)
assert formal['status']=='passed' and formal['GT_opened'] is False and formal['scoring_performed'] is False and formal['generations']==8
assert ft['reason']=='completed' and ft['worker_returncode']==0 and not ft['remaining_process_group'] and ft['worker_pid']==fs['worker_pid']
assert any(pathlib.Path(x).name=='b_verify.py' for x in fs['command']) and str(a.formal_proof.resolve()) in fs['command']
for p,h in formal['files_sha256'].items():assert sha(p)==h
frozen=load(run/'frozen.json');done=load(run/'worker_complete.json');launch=load(run/'launch.json');exited=load(run/'exit.json');st=load(D/'monitor_b001/started.json');te=load(D/'monitor_b001/terminal.json')
assert done['generations']==8 and frozen['gt_read'] is False and frozen['scoring'] is False and frozen['timing_claims'] is False
assert len(frozen['anchors'])==3 and frozen['natural_pairs']==3 and frozen['forced_pairs']==1
for p,h in frozen['code_sha256'].items():assert sha(p)==h
assert te['reason']=='completed' and te['worker_returncode']==0 and not te['remaining_process_group'] and te['worker_pid']==st['worker_pid']
assert exited['returncode']==0 and exited['container_exit_code']==0 and not exited['timed_out'] and exited['container_stopped_verified'] and exited['gpu_pids_after']==[]
container=exited['container_identity']['container_id'];actual=json.loads(subprocess.check_output(['docker','inspect',container]))[0];assert not actual['State']['Running'] and actual['State']['Pid']==0 and actual['State']['ExitCode']==0 and not actual['State']['OOMKilled']
r=rows(run/'records.jsonl');journal=rows(run/'generated.jsonl');comps=rows(run/'comparisons.jsonl');assert len(r)==len(journal)==8 and len(comps)==4
refs={(x['task'],str(x['qid']),x['ds_on']):x for x in rows(B/'e4/smoke_001/records.jsonl') if x['budget']==256 and not x['forced']}
plan=load(B.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');tok=AutoTokenizer.from_pretrained(plan['recipe']['model_path'],local_files_only=True,trust_remote_code=False);token=tok.encode(' the',add_special_tokens=False)[0]
expected=[]
for i,x in enumerate(frozen['anchors']):
 for forced in ((False,True) if i==0 else (False,)):
  for mode in ('C-DROP','C-SKIP'):expected.append((x['task'],str(x['qid']),x['group_id'],mode,forced))
assert [(x['task'],str(x['qid']),x['group_id'],x['mode'],x['forced']) for x in r]==expected
size={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.int32':4,'torch.uint8':1,'torch.bool':1}
for x,j in zip(r,journal):
 assert {k:x[k] for k in ('task','qid','group_id','mode','forced','output_ids','eos_ids','generation_scalars','min_new_tokens')}==j
 assert tok.decode(x['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==x['answer']
 path=pathlib.Path(x['arrays']['path']).resolve();assert path.parent==(run/'arrays').resolve() and sha(path)==x['arrays']['sha256']==formal['array_sha256'][str(path)]
 meta={'visual':x['main_visual'],'pruned':x['pruned_visual'],'prefill':x['prefill_embeds'],'prefill_logits':x['first_logits'],**{'processor_'+k:v for k,v in x['processor'].items()}}
 if x['mode']=='C-DROP':assert len(x['full_deepstack'])==3;meta.update({'ds'+str(i):v for i,v in enumerate(x['full_deepstack'])})
 else:assert x['full_deepstack'] is None
 with np.load(path,allow_pickle=False) as z:
  assert set(z.files)==set(meta)
  for k,v in meta.items():assert z[k].dtype==np.uint8 and z[k].size==int(np.prod(v['shape']))*size[v['dtype']] and hashlib.sha256(z[k].tobytes()).hexdigest()==v['sha256']
  inds=x['spatial_ids'];start=x['image_start'];ordered=x['ordered_ids'];assert len(ordered)==len(set(ordered))==256 and inds==sorted(ordered) and all(0<=n<1024 for n in inds)
  assert x['retained_ids']==list(range(start))+[start+n for n in inds]+list(range(start+1024,x['original_length']))
  d=x['main_visual']['shape'][1]*size[x['main_visual']['dtype']];v=z['visual'].reshape(1024,d);pr=z['pruned'].reshape(256,d);pre=z['prefill'].reshape(-1,d);assert np.array_equal(v[inds],pr) and np.array_equal(pre[start:start+256],pr)
  lm=meta['prefill_logits']; raw=z['prefill_logits'].tobytes(); typ=lm['dtype']
  logits=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32) if typ=='torch.bfloat16' else np.frombuffer(raw,dtype={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}[typ])
  assert logits.size and np.isfinite(logits).all()
 delta=int(x['adjusted_delta'][0][0]);L=len(x['retained_ids']);trace=x['decode_trace'];ids=x['output_ids'];assert len(trace)==len(ids)>0
 assert trace[0]['cache_position']==list(range(L)) and trace[0]['rope_delta']==[[delta]] and trace[0]['layer_cache_lengths'] in ([],[0]*36) and trace[0]['layer_cache_lengths_after']==[L]*36
 for i,t in enumerate(trace[1:],1):
  pos=L+i-1;assert t['cache_position']==[pos] and t['position_ids']==[[[pos+delta]]]*3 and t['rope_delta']==[[delta]] and t['layer_cache_lengths']==[pos]*36 and t['layer_cache_lengths_after']==[pos+1]*36
 kv=x['cache_fingerprints'];assert len(kv)==(33 if x['forced'] else 1)
 for i,step in enumerate(kv):
  assert len(step)==36
  for layer in step:
   assert set(layer)=={'key','value'}
   for item in layer.values():assert item['dtype']=='torch.bfloat16' and item['shape']==[1,8,L+i,128] and re.fullmatch('[0-9a-f]{64}',item['sha256'])
 if x['forced']:assert ids==[token]*33 and x['min_new_tokens']==33 and x['eos_ids']==[]
 else:assert (ids[-1] in x['eos_ids']) or len(ids)==2048
common=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','first_logits','output_ids','answer','eos_ids','eos','truncated','decode_trace','cache_fingerprints','generation_scalars','min_new_tokens')
for idx in range(0,8,2):
 drop,skip=r[idx:idx+2];assert all(drop[k]==skip[k] for k in common) and drop['actual_ds_off'] and skip['actual_ds_off']
 assert sorted(map(tuple,drop['ds_results']))==[(i,False) for i in range(3)] and sorted(map(tuple,skip['ds_results']))==[(i,True) for i in range(3)]
 assert drop['ds_calls']==skip['ds_calls']==[1,1,1] and len(drop['block_calls'])==len(skip['block_calls']) and all(n==1 for n in drop['block_calls'])
 if not drop['forced']:
  off=refs[(drop['task'],str(drop['qid']),False)];on=refs[(drop['task'],str(drop['qid']),True)]
  for ref in (off,on):
   for k in ('processor','main_visual','pruned_visual','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask'):assert ref[k]==drop[k]
  for k in ('output_ids','answer','eos_ids','eos','truncated','max_new_tokens'):assert off[k]==drop[k]
for mode in ('C-DROP','C-SKIP'):
 n=next(x for x in r if x['task']==r[0]['task'] and x['qid']==r[0]['qid'] and x['mode']==mode and not x['forced']);f=next(x for x in r if x['task']==r[0]['task'] and x['qid']==r[0]['qid'] and x['mode']==mode and x['forced'])
 for k in ('processor','main_visual','pruned_visual','ordered_ids','spatial_ids','retained_ids','prefill_position_ids','adjusted_delta','prefill_embeds','first_logits'):assert n[k]==f[k]
proof={'status':'passed','independent_review':True,'scope':'E4-B three DEV natural and one 32-step forced DROP/SKIP pairs','n_generations':8,'natural_pairs':3,'forced_pairs':1,'formal_proof_sha256':sha(a.formal_proof),'formal_started_sha256':sha(a.formal_started),'formal_terminal_sha256':sha(a.formal_terminal),'reviewer_sha256':sha(__file__),'files_sha256':formal['files_sha256'],'array_sha256':formal['array_sha256'],'docker_container_id':container,'GT_opened':False,'scoring_performed':False,'timing_claims':False,'limitations':['KV content represented by actual runtime fingerprints; raw KV tensors not archived','Only three DEV anchors, including one forced prefix; no general equivalence or performance claims']}
a.out.write_text(json.dumps(proof,indent=2)+'\n');print('passed',sha(a.out))
