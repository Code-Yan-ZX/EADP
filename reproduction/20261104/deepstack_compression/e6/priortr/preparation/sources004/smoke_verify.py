"""Independent CPU verification of one completed PriorTR smoke; never scores."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from admission import BASE,E1,admit,sha,jsonl
from smoke_worker import admit_review
HERE=Path(__file__).resolve().parent

def alive(identity):
 if not isinstance(identity,dict):return False
 try:
  boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();raw=Path('/proc',str(identity['pid']),'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  return identity['boot_id']==boot and int(fields[19])==identity['start_ticks'] and fields[0]!='Z'
 except (OSError,ValueError,KeyError,IndexError):return False

def verify_terminal(run,started,term,launch,exit):
 if term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or term.get('worker_pid')!=started.get('worker_pid'):raise ValueError('supervisor incomplete')
 ident=exit.get('container_identity');controller=launch.get('controller')
 if not isinstance(ident,dict) or not ident.get('container_id') or controller!=exit.get('controller') or controller.get('pid')!=term.get('worker_pid'):raise ValueError('container/controller identity')
 for proc in (controller,ident.get('host_init')):
  if not isinstance(proc,dict) or not isinstance(proc.get('pid'),int) or not isinstance(proc.get('start_ticks'),int) or not proc.get('boot_id') or alive(proc):raise ValueError('process identity incomplete or still active')
 command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 scripts=[(cwd/x).resolve() for x in command if Path(x).name=='smoke_launch.py']
 if scripts!=[(HERE/'smoke_launch.py').resolve()] or command.count('--run-id')!=1 or run.name!='smoke_'+command[command.index('--run-id')+1]:raise ValueError('wrong supervisor/launcher binding')
 if exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('timed_out') is not False or exit.get('interrupted') is not None or exit.get('container_stopped_verified') is not True or exit.get('client_exited_while_container_running') is not False or exit.get('gpu_pids_after')!=[]:raise ValueError('unclean exit')
 dock=json.loads(subprocess.run(['docker','inspect',ident['container_id']],capture_output=True,text=True,timeout=10,check=True).stdout)[0]
 if dock['State']['Running'] or dock['State']['ExitCode']!=0 or dock['Id']!=ident['container_id']:raise ValueError('independent docker inspection')
 if launch.get('worker_sha256')!=sha(HERE/'smoke_worker.py') or launch.get('launcher_sha256')!=sha(HERE/'smoke_launch.py'):raise ValueError('launch source identity')
 if launch.get('approval_bundle')!=admit_review(launch['approval_bundle']['path']):raise ValueError('approval source identity')
 cmd=launch['command']
 if cmd.count(str(HERE/'smoke_worker.py'))!=1 or cmd.count('--out')!=1 or Path(cmd[cmd.index('--out')+1]).resolve()!=run:raise ValueError('wrong worker/out')

def validate_record(r,arrays,expected_processor,nlayers,visual_tokens=1024,test_k=None):
 tensors=r['arrays']['tensors'];names=set(tensors)
 required={'full_visual','original_prefill_position','original_prefill_embeds','original_visual_mask','logits_0'}|{'processor_'+k for k in expected_processor}|{'full_ds_'+str(i) for i in range(3)}|{'vision_ds_'+str(i) for i in range(3)}|{'injected_ds_'+str(i) for i in range(3)}|{'injected_mask_'+str(i) for i in range(3)}
 if not required<=names or set(arrays)!=names:raise ValueError('missing original diagnostic arrays')
 for name,m in tensors.items():
  v=arrays[name]
  if v.dtype!=np.uint8 or hashlib.sha256(v.tobytes()).hexdigest()!=m['sha256']:raise ValueError('tensor hash '+name)
 for k,m in expected_processor.items():
  if tensors['processor_'+k]!=m:raise ValueError('processor differs '+k)
 case=r['case'];pruned=case in ('author_default','k256','k256_forced');calls=r['calls'];seq=calls[0]['tokens'];counts=[x['tokens'] for x in r['layers'] if x['step']==0]
 if len(counts)!=nlayers:raise ValueError('layer inventory')
 if len(r['injections'])!=3:raise ValueError('three DS injections missing')
 if case in ('k256','k256_forced'):k=256 if test_k is None else test_k
 elif case=='author_default':k=int(visual_tokens*0.1111)
 else:k=visual_tokens
 if counts!=[seq]*3+[seq-(visual_tokens-k)]*(nlayers-3):raise ValueError('actual per-layer tokens mismatch')
 for i,x in enumerate(r['injections']):
  wanted=visual_tokens if i<2 else k
  if x['rows']!=wanted or x['mask_count']!=wanted:raise ValueError('DS alignment')
 if not r['ids'] or r.get('eos')!=bool(r['ids'][-1] in r['eos_ids']) or r.get('truncated')!=bool(not r['forced'] and len(r['ids'])>=2048 and r['ids'][-1] not in r['eos_ids']):raise ValueError('generation termination state')
 if not r['forced'] and not (r['eos'] or r['truncated']):raise ValueError('natural output incomplete')
 if case not in ('stock','stock_forced','patched_disabled'):
  needed={'selected_ids_0','selection_scores_0','prune_before_position_0','prune_after_position_0','prune_before_hidden_0','prune_after_hidden_0','prune_after_mask_0'}
  if not needed<=names or len(r['prunes'])!=1:raise ValueError('pruning arrays missing')
  m=tensors['selected_ids_0'];selected=np.frombuffer(arrays['selected_ids_0'].tobytes(),dtype=np.int64).reshape(m['shape'])
  if len(selected)!=k or len(set(selected.tolist()))!=k or np.any(np.diff(selected)<0) or (selected<0).any() or (selected>=visual_tokens).any():raise ValueError('selected IDs invalid')
  import torch
  score_meta=tensors['selection_scores_0'];mapping={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.float16':torch.float16}
  dtype=mapping.get(score_meta['dtype'])
  if dtype is None:raise ValueError('unsupported score dtype')
  scores=torch.from_numpy(arrays['selection_scores_0'].copy()).view(dtype).reshape(score_meta['shape']).float()
  if scores.numel()!=visual_tokens or not torch.isfinite(scores).all():raise ValueError('invalid selection scores')
  dropped=torch.ones(visual_tokens,dtype=torch.bool);dropped[torch.as_tensor(selected.copy())]=False
  if dropped.any() and scores[torch.as_tensor(selected.copy())].min()<scores[dropped].max():raise ValueError('selection is not top-k')
  before=r['prunes'][0]['range_before'];after=r['prunes'][0]['range_after'];start,end=before
  if end-start!=visual_tokens or after!=[start,start+k] or r['prunes'][0]['layer']!=2:raise ValueError('prune geometry')
  retained=np.concatenate((np.arange(start),start+selected,np.arange(end,seq)))
  for name,dtype,axis in [('position',np.int64,2),('hidden',np.uint8,1)]:
   b=arrays['prune_before_'+name+'_0'];a=arrays['prune_after_'+name+'_0']
   if dtype==np.int64:
    b=np.frombuffer(b.tobytes(),dtype=dtype).reshape(tensors['prune_before_'+name+'_0']['shape']);a=np.frombuffer(a.tobytes(),dtype=dtype).reshape(tensors['prune_after_'+name+'_0']['shape'])
   if not np.array_equal(np.take(b,retained,axis=axis),a):raise ValueError('pruned '+name+' does not gather actual indices')
  for i in range(3):
   actual=arrays['injected_ds_'+str(i)];full=arrays['full_ds_'+str(i)]
   expected=full if i<2 else full[selected]
   if not np.array_equal(actual,expected):raise ValueError('injected DS rows not aligned')
 if len(calls)!=len(r['ids']):raise ValueError('forward/token count mismatch')
 expected_length=seq-(visual_tokens-k)
 if calls[0]['cache_after']!=[expected_length]*nlayers:raise ValueError('prefill KV lengths')
 if r['forced'] and (len(r['ids'])!=33 or any(x!=r['force_token'] for x in r['ids'])):raise ValueError('forced output/32 decode steps')
 logit_names={'logits_'+str(i) for i in range(len(calls))} if r['forced'] else {'logits_0'}
 if not logit_names<=names:raise ValueError('missing logits')
 import torch
 for name in logit_names:
  mm=tensors[name];mapping={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32,'torch.float16':torch.float16};dtype=mapping.get(mm['dtype'])
  if dtype is None or not torch.isfinite(torch.from_numpy(arrays[name].copy()).view(dtype)).all():raise ValueError('nonfinite/unsupported logits')
 for step,c in enumerate(calls[1:]):
  if c['cache_before']!=[expected_length+step]*nlayers or c.get('cache_after')!=[expected_length+step+1]*nlayers:raise ValueError('decode KV trajectory')
  pos=np.asarray(c['position_ids']);expect=seq+int(r['rope_delta'][0][0])+step
  if not np.all(pos[-3:]==expect):raise ValueError('decode mRoPE trajectory')
 return k

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--started',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run=a.run.resolve();proof,rows=admit();started=json.loads(a.started.read_text());term=json.loads(a.terminal.read_text());launch=json.loads((run/'launch.json').read_text());exited=json.loads((run/'exit.json').read_text());verify_terminal(run,started,term,launch,exited)
 frozen=json.loads((run/'frozen.json').read_text());done=json.loads((run/'worker_complete.json').read_text());plan=json.loads((HERE/'smoke_single_plan002.json').read_text())
 if frozen.get('sample_plan_sha256')!=sha(HERE/'smoke_single_plan002.json') or frozen.get('gt_read') is not False or frozen.get('scoring') is not False or done.get('cases')!=len(plan['cases']):raise ValueError('frozen/complete mismatch')
 records=jsonl(run/'records.jsonl');generated=jsonl(run/'generated.jsonl');names=[x['name'] for x in plan['cases']]
 if [x['case'] for x in records]!=names or [x['case'] for x in generated]!=names:raise ValueError('case inventory')
 for g,r in zip(generated,records):
  if g['ids']!=r['ids'] or g['forced']!=r['forced']:raise ValueError('generated and recorded mismatch')
 check=json.loads((HERE/'processor_check002.json').read_text()); row=plan['selected'];refs=[x['actual'] for x in check['checks'] if x['task']==row['task'] and str(x['qid'])==str(row['qid'])]
 if len(refs)!=1:raise ValueError('frozen input absent')
 from transformers import AutoTokenizer
 recipe=json.loads((BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe'];tok=AutoTokenizer.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
 expected_force=tok.encode(' the',add_special_tokens=False)[0]
 for r in records:
  expected_kwargs={'do_sample':False,'max_new_tokens':33 if r['forced'] else 2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'use_cache':True}
  expected_processor={'class':'Forced','token_id':expected_force,'min_new_tokens':33} if r['forced'] else None
  if r.get('generated_kwargs')!=expected_kwargs or r.get('forced_processor')!=expected_processor or (r['forced'] and r.get('force_token')!=expected_force):raise ValueError('actual generation recipe/forced token mismatch')
  path=Path(r['arrays']['path']).resolve()
  if path.parent!=(run/'arrays').resolve() or path.name!=r['case']+'.npz' or sha(path)!=r['arrays']['sha256']:raise ValueError('array identity')
  with np.load(path,allow_pickle=False) as a2:validate_record(r,a2,refs[0],36)
  if tok.decode(r['ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['decoded']:raise ValueError('output decode mismatch')
 by_case={r['case']:r for r in records}
 for a_name,b_name in (('stock','stock_forced'),('k256','k256_forced')):
  left,right=by_case[a_name],by_case[b_name]
  common={'full_visual','original_prefill_position','original_prefill_embeds','original_visual_mask','logits_0'}|{'full_ds_'+str(i) for i in range(3)}|{'injected_ds_'+str(i) for i in range(3)}|{'injected_mask_'+str(i) for i in range(3)}|{'processor_'+k for k in refs[0]}
  if a_name=='k256':common|={'selected_ids_0','selection_scores_0','prune_before_position_0','prune_after_position_0','prune_before_hidden_0','prune_after_hidden_0','prune_after_mask_0'}
  for key in common:
   if left['arrays']['tensors'].get(key)!=right['arrays']['tensors'].get(key):raise ValueError('natural/forced prefill mismatch: '+key)
  if left['prunes']!=right['prunes'] or [x for x in left['layers'] if x['step']==0]!=[x for x in right['layers'] if x['step']==0]:raise ValueError('natural/forced prefill path mismatch')
 native=next(x for x in records if x['case']=='stock');patched=next(x for x in records if x['case']=='patched_disabled');keep=next(x for x in records if x['case']=='priortr_keep_all')
 result={'status':'passed','scope':'single DEV diagnostic only','cases':names,'patched_disabled_same_ids_as_stock':native['ids']==patched['ids'],'patched_disabled_same_first_logits_as_stock':native['arrays']['tensors']['logits_0']==patched['arrays']['tensors']['logits_0'],'keep_all_same_ids_as_stock':native['ids']==keep['ids'],'keep_all_same_first_logits_as_stock':native['arrays']['tensors']['logits_0']==keep['arrays']['tensors']['logits_0'],'files_sha256':{str(x):sha(x) for x in (run/'frozen.json',run/'worker_complete.json',run/'records.jsonl',run/'generated.jsonl',run/'launch.json',run/'exit.json',a.started,a.terminal)},'verifier_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
