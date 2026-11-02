"""One model; all sixteen native K256 anchors before any DS-on generation."""
import argparse,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
import common as c
from worker_draft import run_arm,baseline_capture
from historical_anchor import validate as historical_anchor

def worker(note_path):
 p=c.read(c.PLAN);c.check(p,'container');note=c.read(note_path)
 assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID and note['plan_sha256']==c.sha(c.PLAN)
 assert note['GPU_device']==0 and note['single_attempt'] and not note['automatic_retry'] and note['maximum_concurrent_models_including_loading']==1
 old=c.read(c.OLD/'run_plan.json');assert Path(sys.executable).resolve()==Path(old['python']).resolve()
 out=c.HERE/'execution';out.mkdir(exist_ok=False)
 c.save(out/'identity.json',dict(**c.identity(os.getpid()),plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)))
 import torch,numpy as np
 torch.set_num_threads(4);sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner,Qwen3VLChatEADP
 import model.pruner as native
 assert {n:importlib.metadata.version(n) for n in old['expected_versions']}==old['expected_versions']
 recipe=dict(old['recipe']);assert recipe['visual_token_num']==256 and recipe['alpha']==.5 and recipe['beta']==2.
 wrapper=Qwen3VLChatEADP(**recipe)
 assert wrapper.visual_token_num==wrapper.pruner.visual_token_num==256 and wrapper.post_process is False
 assert wrapper.model.config._attn_implementation==wrapper.model.visual.config._attn_implementation=='flash_attention_2'
 assert wrapper.model.dtype==wrapper.model.visual.dtype==torch.bfloat16
 assert {str(x.device) for x in wrapper.model.parameters()}=={'cuda:0'} and torch.cuda.get_device_name(0)=='NVIDIA A40'
 modules=['vlmeval.vlm.qwen3_vl.model_fixed_res','vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl']
 sources={n:str(Path(sys.modules[n].__file__).resolve()) for n in modules}
 for path in sources.values():assert c.sha(path)==old['source_sha256'][path]
 c.save(out/'runtime.json',dict(actual_recipe=recipe,actual_sources=sources,model_dtype=str(wrapper.model.dtype),backend='flash_attention_2',post_process=False,GPU=torch.cuda.get_device_name(0),model_instances=1,plan_sha256=c.sha(c.PLAN),cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,GT_read=False,scoring_started=False))
 def metadata(x):
  x=x.detach().cpu().contiguous();a=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
  return a,dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
 def capture(i,item,arm,base):
  rawpath=out/f'{i:02d}_{arm}.raw.json'
  def preserve_raw(item,arm,response,failure):
   c.save(rawpath,dict(index=item['index'],position=item['position'],arm=arm,response=response,native_failure=failure))
  rec=run_arm(torch,wrapper,native,Qwen3VLChatCDPruner.generate_inner_transformers_fixed,item,arm,base,preserve_raw)
  record=dict(index=item['index'],position=item['position'],arm=arm,response=rec['response'],native_failure=rec['native_failure'],observation_errors=rec['errors'],selection_errors=rec['selection_errors'],intervention_attempts=rec['intervention_attempts'],intervention_applied=rec['intervention_applied'],intervention_error=rec['intervention_error'],raw_sha256=c.sha(rawpath),status='raw_preserved')
  for key in ('vision_calls','pruner_calls','processor_calls','generate_calls','llm_calls','selection_calls'):record[key]=rec[key]
  arrays={};declared={};errors=record['observation_errors']
  def save_tensor(name,value):
   if value is None:return
   try:
    if not isinstance(value,torch.Tensor):raise TypeError('not tensor')
    arrays[name],declared[name]=metadata(value)
   except Exception as error:errors.append(dict(event='serialize:'+name,error=repr(error)))
  vis=rec.get('vision')
  if isinstance(vis,(list,tuple)) and len(vis)==2:
   save_tensor('visual_features',vis[0])
   for j,x in enumerate(vis[1]):save_tensor('visual_deepstack_'+str(j),x)
  if len(rec.get('selection_results',[]))==1:
   sel=rec['selection_results'][0];save_tensor('ordered_ids',sel['ordered_ids']);save_tensor('spatial_ids',sel['ordered_ids'][0].sort().values);save_tensor('quotas',sel['quotas'])
  for key in ('pruner_input','pruner_output','original_token_embeddings','rope_deltas','output_ids'):save_tensor(key,rec.get(key))
  for k,v in rec.get('processor_inputs',{}).items():save_tensor('processor_'+k,v)
  g=rec.get('generation_kwargs',{})
  for k,v in g.items():
   if isinstance(v,torch.Tensor):save_tensor('generate_'+k,v)
  fields=('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks')
  for label in ('original_prefill','proposed_prefill','final_prefill'):
   d=rec.get(label,{})
   for key in fields:save_tensor(label+'_'+key,d.get(key))
   for j,x in enumerate(d.get('deepstack_visual_embeds') or []):save_tensor(label+'_deepstack_'+str(j),x)
  presence=[]
  for j,step in enumerate(rec['continuation']):
   for key in ('position_ids','cache_position','rope_deltas'):save_tensor('continuation_%04d_%s'%(j,key),step[key])
   presence.append({k:step[k] for k in ('visual_mask_present','deepstack_present')})
  try:
   output=rec['output_ids'];ids=output.detach().cpu().tolist();assert len(ids)==1
   record['native_redecode']=wrapper.processor.tokenizer.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  except Exception as error:errors.append(dict(event='native_redecode',error=repr(error)))
  path=out/f'{i:02d}_{arm}.npz';np.savez_compressed(path,**arrays)
  record.update(status='observations_preserved',archive_sha256=c.sha(path),arrays=declared,generate_kwargs_keys=sorted(g),generate_non_tensor_kwargs={k:v for k,v in g.items() if v is None or type(v) in (str,int,float,bool)},continuation_presence=presence,continuation_steps=len(presence),GT_read=False,scoring_started=False)
  c.save(out/f'{i:02d}_{arm}.json',record)
  assert rec['native_failure'] is None and isinstance(rec['response'],str) and not errors and not rec['selection_errors']
  assert all(rec[k]==1 for k in ('vision_calls','pruner_calls','processor_calls','generate_calls','selection_calls'))
  assert rec['llm_calls']>=1 and len(presence)==rec['llm_calls']-1 and record['native_redecode']==rec['response']
  if arm=='ds_off':
   assert rec['intervention_attempts']==0 and not rec['intervention_applied']
   baseline_capture(wrapper,rec);record['historical_anchor']=historical_anchor(torch,i,item,rec);record['status']='historical_anchor_passed';c.save(out/f'{i:02d}_{arm}.json',record)
  else:assert rec['intervention_attempts']==1 and rec['intervention_applied'] and rec['intervention_error'] is None
  return record
 def load_baseline(i):
  record=c.read(out/f'{i:02d}_ds_off.json');path=out/f'{i:02d}_ds_off.npz';assert c.sha(path)==record['archive_sha256']
  with np.load(path,allow_pickle=False) as z:
   def tensor(k):
    a=z[k].copy();t=torch.from_numpy(a);dtype=record['arrays'][k]['dtype']
    if dtype=='torch.bfloat16':t=t.view(torch.bfloat16)
    assert str(t.dtype)==dtype
    return t.to('cuda:0')
   result=dict(ordered_ids=tensor('ordered_ids')[0],spatial_ids=tensor('spatial_ids'),visual_features=tensor('visual_features'),deepstack=[tensor('visual_deepstack_'+str(j)) for j in range(3)],pruner_input=tensor('pruner_input'),pruner_output=tensor('pruner_output'),split_sizes=[256],processor_input_ids=tensor('processor_input_ids'),processor_mask=tensor('processor_attention_mask'),processor_pixel_values=tensor('processor_pixel_values'),processor_image_grid_thw=tensor('processor_image_grid_thw'),original_token_embeddings=tensor('original_token_embeddings'),rope_deltas=tensor('rope_deltas'),image_token_id=wrapper.model.config.image_token_id)
   result['generation_kwargs']={**record['generate_non_tensor_kwargs'],**{k:tensor('generate_'+k) for k in ('inputs_embeds','attention_mask')}}
   result['original_prefill']={k:tensor('original_prefill_'+k) if 'original_prefill_'+k in z else None for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks')}
   result['original_prefill']['deepstack_visual_embeds']=None
   return result
 for i,item in enumerate(p['panel']):
  capture(i,item,'ds_off',None);print(json.dumps(dict(native_anchors_completed=i+1,total=16)),flush=True)
 c.save(out/'native_anchors.json',dict(success=True,n=16,plan_sha256=c.sha(c.PLAN),records={f'{i:02d}':c.sha(out/f'{i:02d}_ds_off.json') for i in range(16)},archives={f'{i:02d}':c.sha(out/f'{i:02d}_ds_off.npz') for i in range(16)},all_native_anchors_before_any_DS_on=True,GT_read=False,scoring_started=False))
 for i,item in enumerate(p['panel']):
  base=load_baseline(i);capture(i,item,'ds_on',base);del base;print(json.dumps(dict(pairs_completed=i+1,total=16)),flush=True)
 for path,digest in p['files'].items():assert c.sha(path)==digest,path
 for group in ('source_sha256','implementation_sha256','reference_sha256'):
  for path,digest in old[group].items():assert c.sha(path)==digest,path
 from namespace_guards import check_model_stats
 from diagnose import stat
 check_model_stats(old,stat,'container')
 c.save(out/'native_complete.json',dict(success=True,n_questions=16,n_generations=32,native_anchors=16,plan_sha256=c.sha(c.PLAN),native_anchors_sha256=c.sha(out/'native_anchors.json'),records={f'{i:02d}_{a}':c.sha(out/f'{i:02d}_{a}.json') for i in range(16) for a in ('ds_off','ds_on')},archives={f'{i:02d}_{a}':c.sha(out/f'{i:02d}_{a}.npz') for i in range(16) for a in ('ds_off','ds_on')},GT_read=False,scoring_started=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--allocation-note',type=Path,required=True);args=ap.parse_args()
 try:worker(args.allocation_note.resolve())
 except BaseException as error:
  c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),raw_artifacts_preserved=True));raise
