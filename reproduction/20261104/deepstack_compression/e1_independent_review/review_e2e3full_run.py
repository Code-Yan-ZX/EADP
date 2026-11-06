"""Independent complete E2/E3 run readback. No GT or scoring."""
import argparse, functools, hashlib, json, os, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
BASE=Path(__file__).resolve().parent.parent; FULL=BASE/'e2_e3_full'
ARMS=[('E2',x) for x in ('000','100','010','001','110','101','011','111')]+[('E3','PERM'),('E3','MEAN')]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):
 data=Path(path).read_bytes()
 if data and not data.endswith(b'\n'):raise ValueError('partial journal')
 return [json.loads(x) for x in data.splitlines()]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def rawhash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
@functools.lru_cache(maxsize=16)
def zero_hash(shape,dtype):
 size={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8}[dtype]
 return hashlib.sha256(bytes(int(np.prod(shape))*size)).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for key in ('run','formal','formal-started','formal-terminal','run-started','run-terminal','out'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();run=a.run.resolve();formal=read(a.formal)
 require(formal['status']=='passed' and formal['questions']==384 and formal['physical_generations']==3840 and formal['GT_opened'] is False and formal['scoring_performed'] is False,'formal proof')
 for path,digest in {**formal['files_sha256'],**formal['array_sha256']}.items():require(sha(path)==digest,'formal artifact changed')
 for start,end in ((a.formal_started,a.formal_terminal),(a.run_started,a.run_terminal)):
  st=read(start);t=read(end);require(t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and st['worker_pid']==t['worker_pid'],'supervisor exit')
 fs=read(a.formal_started);cmd=fs['command'];cwd=Path(fs['cwd'])
 require([(cwd/x).resolve() for x in cmd if Path(x).name=='panel_verify.py']==[(FULL/'panel_verify.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==a.formal.resolve() and formal['verifier_sha256']==sha(FULL/'panel_verify.py'),'formal command/source')
 rs=read(a.run_started);cmd=rs['command'];cwd=Path(rs['cwd']);require([(cwd/x).resolve() for x in cmd if Path(x).name=='panel_launch.py']==[(FULL/'panel_launch.py').resolve()] and cmd.count('--run-id')==1 and run.name=='run_'+cmd[cmd.index('--run-id')+1],'run command')
 ex=read(run/'exit.json');require(ex['returncode']==0 and ex['container_stopped_verified'] is True and ex['container_exit_code']==0 and ex['gpu_pids_after']==[] and ex['failure'] is None,'launcher exit')
 obj=json.loads(subprocess.check_output(['docker','inspect',ex['container_identity']['container_id']],timeout=20))[0]
 require(not obj['State']['Running'] and obj['State']['Pid']==0 and obj['State']['ExitCode']==0 and not obj['State']['OOMKilled'],'actual container exit')
 e1path=(BASE/'e1/e1full001_completion.json').resolve();require(formal['files_sha256'].get(str(e1path))==sha(e1path),'E1 formal proof binding')
 e1=read(e1path)
 for name in ('inputs','journal'):
  desc=e1['files'][name];require(sha(desc['path'])==desc['sha256'],'E1 '+name+' content changed')
 rows=lines(e1['files']['inputs']['path']);refs={}
 for r in lines(e1['files']['journal']['path']):
  if r.get('forced') is False and r.get('arm') in ('P3-D0','P3-D1'):
   key=(r['task'],str(r['qid']),r['arm']);require(key not in refs,'duplicate E1 reference');refs[key]=r
 require(len(rows)==384 and len(refs)==768,'E1 inventory')
 records=lines(run/'data/records.jsonl');journal=lines(run/'data/generated.jsonl');views=lines(run/'data/e3_views.jsonl')
 require(len(records)==len(journal)==3840 and len(views)==384,'full denominator')
 frozen=read(run/'frozen.json');require(frozen['GT_opened'] is False and frozen['scoring_performed'] is False and frozen['code_sha256'] and all(sha(path)==digest for path,digest in frozen['code_sha256'].items()),'frozen source hash');anchors={(r['task'],str(r['qid'])) for r in frozen['anchors']};require(len(anchors)==3,'anchor inventory')
 plan=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 tok=AutoTokenizer.from_pretrained(read(plan)['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 checks=0;array_paths=set()
 for qi,row in enumerate(rows):
  task=row['task'];qid=str(row['qid']);key=(task,qid);block=records[qi*10:qi*10+10];js=journal[qi*10:qi*10+10]
  require([(x['family'],x['arm']) for x in block]==ARMS,'arm order')
  for r,g,(family,arm) in zip(block,js,ARMS):
   for k in ('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced','family','arm'):
    require(r[k]==g[k],'generation event mismatch: '+k);checks+=1
   require((r['task'],str(r['qid']),r['group_id'])==(task,qid,row['group_id']) and r['image_sha256']==row['image_sha256'] and r['message_sha256']==row['message_sha256'],'question identity')
   require(r['forced'] is False and r['budget']==256 and r['ds_on'] is True and r['actual_injection_verified'] is True,'execution mode')
   ids=r['output_ids'];eos=bool(ids and ids[-1] in r['eos_ids']);require(ids and len(ids)<=2048 and r['eos']==eos and r['truncated']==(len(ids)==2048 and not eos) and (eos or r['truncated']),'incomplete output')
   require(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)==r['answer'],'token decoding')
   ref=refs[(task,qid,'P3-D1')]
   for x,y in (('processor','processor'),('main_visual','visual_sha256'),('pruned_visual','pruned_sha256'),('ordered_ids','ordered_ids'),('spatial_ids','spatial_ids'),('retained_ids','retained_ids'),('full_deepstack','deepstack_full_sha256'),('prefill_position_ids','prefill_position_ids'),('adjusted_delta','rope_delta'),('prefill_embeds','prefill_embed_sha256'),('prefill_mask','attention_mask_sha256'),('generation_scalars','generation_scalars')):
    require(r[x]==ref[y],'E1 common input '+x);checks+=1
   if family=='E2' and arm in ('000','111'):
    ep=refs[(task,qid,'P3-D0' if arm=='000' else 'P3-D1')]
    for k in ('output_ids','answer','eos_ids','eos','truncated'):require(r[k]==ep[k],'endpoint '+k);checks+=1
    require(r['prefill_logits']==ep['logits'][0],'endpoint first logits')
   start=r['image_start'];length=r['original_length'];spatial=r['spatial_ids'];ordered=r['ordered_ids']
   require(len(ordered)==256 and spatial==sorted(set(ordered)) and len(spatial)==256 and min(spatial)>=0 and max(spatial)<1024,'selection')
   kept=list(range(start))+[start+i for i in spatial]+list(range(start+1024,length));full=np.asarray(r['full_position_ids'],dtype=np.int64).reshape(3,length)
   delta=int(full.max())+1-len(kept);require(r['retained_ids']==kept and r['prefill_position_ids']==full[:,kept][:,None,:].tolist() and r['adjusted_delta']==[[delta]],'P3 positions')
   require(len(r['decode_trace'])==len(ids),'decode count')
   for i,z in enumerate(r['decode_trace']):
    require(z['layer_cache_lengths_after']==[len(kept)+i]*36 and z['rope_delta']==[[delta]],'decode KV or delta')
    cp=len(kept)+i-1
    require(z['cache_position']==(list(range(len(kept))) if i==0 else [cp]) and z['position_ids']==(r['prefill_position_ids'] if i==0 else [[[cp+delta]]]*3),'decode positions')
    require(z['layer_cache_lengths'] in (([],[0]*36) if i==0 else ([cp]*36,)),'decode KV before');checks+=36
   require(r['injected_mask']==[[False]*start+[True]*256+[False]*(length-start-1024)] and r['greedy_calls']==1 and r['visual_calls']==1 and r['main_merger_calls']==1 and all(x==1 for x in r['block_calls']) and r['ds_calls']==[1,1,1],'actual injection and vision calls')
   arc=r['arrays']
   if key in anchors:
    path=Path(arc['path']).resolve();require(arc['kind']=='full_arrays' and path.parent==(run/'data/arrays').resolve() and path not in array_paths and sha(path)==arc['sha256'],'anchor archive');array_paths.add(path)
    with np.load(path,allow_pickle=False) as arrays:
     require(rawhash(arrays['visual'])==r['main_visual']['sha256'] and rawhash(arrays['visual'][spatial])==r['pruned_visual']['sha256'] and rawhash(arrays['prefill'])==r['prefill_embeds']['sha256'] and np.array_equal(arrays['prefill'][0,start:start+256],arrays['visual'][spatial]),'visual array')
     for i in range(3):
      full_ds=arrays['ds'+str(i)];before=arrays['selected_before'+str(i)];actual=arrays['injected'+str(i)]
      require(rawhash(full_ds)==r['full_deepstack'][i]['sha256'] and np.array_equal(before,full_ds[spatial]) and rawhash(actual)==r['injected_deepstack'][i]['sha256'],'DS array')
      if family=='E2':want=before if arm[i]=='1' else np.zeros_like(before)
      elif arm=='PERM':
       seed=int.from_bytes(hashlib.sha256(('20261104|'+r['image_sha256']+'|perm').encode()).digest()[:8],'big');rng=random.Random(seed);perm=list(range(256))
       for j in range(255,0,-1):
        k=rng.randrange(j);perm[j],perm[k]=perm[k],perm[j]
       require(r['intervention_metadata']['permutation']==perm and all(j!=x for j,x in enumerate(perm)),'permutation');want=before[perm]
      else:
       dt={'torch.bfloat16':torch.bfloat16,'torch.float16':torch.float16,'torch.float32':torch.float32}[r['full_deepstack'][i]['dtype']];z=torch.from_numpy(before.copy()).view(dt).reshape(256,-1)
       want=z.mean(0,keepdim=True).expand_as(z).contiguous().view(torch.uint8).numpy().reshape(actual.shape)
      require(np.array_equal(actual,want),'injected array');checks+=1
   else:
    require(arc['kind']=='summary_only' and r['selected_deepstack']==r['injected_deepstack'],'non-anchor runtime summary')
    summary=arc['tensors'];require(summary['visual']==r['main_visual'] and summary['prefill']==r['prefill_embeds'],'non-anchor main summary')
    for i in range(3):
     require(summary['ds'+str(i)]==r['full_deepstack'][i] and summary['injected'+str(i)]==r['injected_deepstack'][i],'non-anchor DS summary')
     if family=='E2':
      if arm[i]=='1':require(summary['selected_before'+str(i)]==r['injected_deepstack'][i],'non-anchor retained slot')
      else:
       meta=r['injected_deepstack'][i];require(meta['sha256']==zero_hash(tuple(meta['shape']),meta['dtype']),'non-anchor disabled slot')
    if family=='E3' and arm=='PERM':
     seed=int.from_bytes(hashlib.sha256(('20261104|'+r['image_sha256']+'|perm').encode()).digest()[:8],'big');rng=random.Random(seed);perm=list(range(256))
     for j in range(255,0,-1):
      k=rng.randrange(j);perm[j],perm[k]=perm[k],perm[j]
     require(r['intervention_metadata']['seed']==seed and r['intervention_metadata']['permutation']==perm and all(j!=x for j,x in enumerate(perm)),'non-anchor permutation metadata')
  view=views[qi];require((view['task'],str(view['qid']),view['group_id'])==(task,qid,row['group_id']),'E3 view identity')
  for mode,idx in (('OFF',0),('MATCH',7),('PERM',8),('MEAN',9)):
   z=block[idx];digest=hashlib.sha256(json.dumps(z,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
   require(view['arms'][mode]=={'source_family':z['family'],'source_arm':z['arm'],'record_sha256':digest},'E3 alias')
 require(len(array_paths)==30 and array_paths=={Path(p).resolve() for p in formal['array_sha256']} and array_paths=={p.resolve() for p in (run/'data/arrays').glob('*.npz')},'anchor arrays inventory')
 proof={'status':'passed','independent_review':True,'questions':384,'physical_generations':3840,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(a.formal),'formal_started_sha256':sha(a.formal_started),'formal_terminal_sha256':sha(a.formal_terminal),'reviewer_sha256':sha(__file__),'n_checks':checks,'anchor_arrays_verified':len(array_paths),'finished_unix':time.time(),'limits':['Non-anchor high-dimensional arrays are not archived and cannot be independently reconstructed.','Cache contents not retained for this natural full panel.','No scoring or latency interpretation.']}
 with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'passed','questions':384,'records':3840,'checks':checks}))
if __name__=='__main__':main()
