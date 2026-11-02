"""Draft CPU-prepared worker: source/full16 anchors first, then per-item pairs."""
import argparse,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
import common as c
from worker_draft import run_arm,baseline_capture
from contracts import mask_span

def worker(note_path):
 p=c.read(c.PLAN);c.check(p,'container');note=c.read(note_path)
 assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID and note['plan_sha256']==c.sha(c.PLAN)
 assert note['GPU_device']==0 and note['single_attempt'] and not note['automatic_retry'] and note['maximum_concurrent_models_including_loading']==1
 old=c.read(c.OLD/'run_plan.json');assert Path(sys.executable).resolve()==Path(old['python']).resolve()
 out=c.HERE/'execution';out.mkdir(exist_ok=False);anchors=out/'anchors';anchors.mkdir()
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
 eos=wrapper.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
 c.save(out/'runtime.json',dict(actual_recipe=recipe,actual_sources=sources,model_dtype=str(wrapper.model.dtype),backend='flash_attention_2',post_process=False,GPU=torch.cuda.get_device_name(0),model_instances=1,plan_sha256=c.sha(c.PLAN),eos_token_ids=eos,cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,GT_read=False,scoring_started=False))
 wrapper.set_dump_image(lambda row:[row['image_path']]);items=c.rows(c.INPUT);panel=p['panel'];panel_positions={r['position']:i for i,r in enumerate(panel)}
 def get_item(pos):
  item=dict(items[pos]);msg=wrapper._build_vqa_prompt(item,'DocVQA_VAL');messages=wrapper._build_messages(msg,dataset='DocVQA_VAL')
  assert hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==item['message_sha256']
  item['message']=msg;return item
 def save_line(stream,value):stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush()
 def array_value(t):
  a=t.detach().cpu().contiguous();return a.view(torch.uint16).numpy() if a.dtype==torch.bfloat16 else a.numpy()
 def meta(t):
  if t is None:return None
  a=array_value(t);return dict(dtype=str(t.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
 def prefill(d):
  fields=('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks');deep=d.get('deepstack_visual_embeds') or []
  assert len(deep) in (0,3)
  return {**{k:meta(d.get(k)) for k in fields},**{'deepstack_'+str(j):meta(deep[j]) if deep else None for j in range(3)}}
 def collect_arrays(rec):
  values={}
  def put(k,v):
   if isinstance(v,torch.Tensor):values[k]=v
  vis=rec.get('vision')
  if isinstance(vis,(tuple,list)) and len(vis)==2:
   put('visual_features',vis[0])
   for j,x in enumerate(vis[1]):put('visual_deepstack_'+str(j),x)
  if len(rec.get('selection_results',[]))==1:
   sel=rec['selection_results'][0];put('ordered_ids',sel['ordered_ids']);put('spatial_ids',sel['ordered_ids'][0].sort().values);put('quotas',sel['quotas'])
  for k in ('pruner_input','pruner_output','original_token_embeddings','rope_deltas','output_ids'):put(k,rec.get(k))
  for k,v in rec.get('processor_inputs',{}).items():put('processor_'+k,v)
  for k,v in rec.get('generation_kwargs',{}).items():put('generate_'+k,v)
  for label in ('original_prefill','proposed_prefill','final_prefill'):
   d=rec.get(label,{})
   for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks'):put(label+'_'+k,d.get(k))
   for j,x in enumerate(d.get('deepstack_visual_embeds') or []):put(label+'_deepstack_'+str(j),x)
  for j,step in enumerate(rec['continuation']):
   for k in ('position_ids','cache_position','rope_deltas'):put('continuation_%04d_%s'%(j,k),step[k])
  return values
 streams={}
 def capture(item,arm,base,anchor_i=None):
  def preserve_raw(item,arm,response,failure):save_line(streams[arm][0],dict(position=item['position'],index=item['index'],arm=arm,response=response,native_failure=failure))
  rec=run_arm(torch,wrapper,native,Qwen3VLChatCDPruner.generate_inner_transformers_fixed,item,arm,base,preserve_raw)
  obs=dict(position=item['position'],index=item['index'],arm=arm,native_failure=rec['native_failure'],observer_errors=list(rec['errors']),selection_errors=rec['selection_errors'],intervention_attempts=rec['intervention_attempts'],intervention_applied=rec['intervention_applied'],intervention_error=rec['intervention_error'])
  for k in ('vision_calls','pruner_calls','processor_calls','generate_calls','llm_calls','selection_calls'):obs[k]=rec[k]
  arrays={};declared={}
  try:
   tensors=collect_arrays(rec)
   for name,tensor in tensors.items():
    if anchor_i is not None:
     a=array_value(tensor);arrays[name]=a;declared[name]=dict(dtype=str(tensor.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
    else:declared[name]=meta(tensor)
   ids=rec['processor_inputs']['input_ids'][0].detach().cpu().tolist();ordered=rec['selection_results'][0]['ordered_ids'][0];spatial=ordered.sort().values
   v=rec['vision'];g=rec['generation_kwargs'];image_mask=mask_span(ids,wrapper.model.config.image_token_id);start=image_mask.index(True)
   token=rec['original_token_embeddings'];pruned=rec['pruner_output'];attn=rec['processor_inputs']['attention_mask']
   rebuilt=torch.cat((token[:,:start],pruned.unsqueeze(0),token[:,start+1024:]),dim=1)
   rebuilt_mask=torch.cat((attn[:,:start],torch.ones_like(attn[:,:256]),attn[:,start+1024:]),dim=1)
   output=rec['output_ids'].detach().cpu().tolist();assert len(output)==1
   obs.update(actual_processor={k:declared['processor_'+k] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},input_ids=ids,image_mask=image_mask,
     ordered_ids=ordered.detach().cpu().tolist(),spatial_ids=spatial.detach().cpu().tolist(),quotas=rec['selection_results'][0]['quotas'].detach().cpu().tolist(),pruner_split_sizes=rec['split_sizes'],
     vision_features=declared['visual_features'],visual_deepstack=[declared['visual_deepstack_'+str(j)] for j in range(3)],pruner_input=declared['pruner_input'],pruner_output=declared['pruner_output'],original_token_embeddings=declared['original_token_embeddings'],
     selected_vision_features=meta(v[0].index_select(0,spatial)),selected_visual_deepstack=[meta(x.index_select(0,spatial)) for x in v[1]],reconstructed_inputs_embeds=meta(rebuilt),reconstructed_attention_mask=meta(rebuilt_mask),
     generate_kwargs_keys=sorted(g),generate_non_tensor_kwargs={k:v for k,v in g.items() if v is None or type(v) in (str,int,float,bool)},generate_tensor_kwargs={k:declared['generate_'+k] for k,v in g.items() if isinstance(v,torch.Tensor)},
     prefill_original={k:declared.get('original_prefill_'+k) for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_0','deepstack_1','deepstack_2')},prefill_proposed={k:declared.get('proposed_prefill_'+k) for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_0','deepstack_1','deepstack_2')},prefill_final={k:declared.get('final_prefill_'+k) for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks','deepstack_0','deepstack_1','deepstack_2')},rope_deltas=declared['rope_deltas'],
     continuation=[{**{k:meta(step[k]) for k in ('position_ids','cache_position','rope_deltas')},**{k:step[k] for k in ('visual_mask_present','deepstack_present')}} for step in rec['continuation']],
     continuation_steps=len(rec['continuation']),output_ids=output[0],native_redecode=wrapper.processor.tokenizer.batch_decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0],
     eos=dict(ids=eos,present=any(x in eos for x in output[0]),at_end=bool(output[0]) and output[0][-1] in eos,at_max_tokens=len(output[0])==2048))
  except Exception as error:obs['observer_errors'].append(dict(event='post_generation_serialization',error=repr(error)))
  # Preserve every available observation before deciding whether it is valid.
  save_line(streams[arm][1],obs)
  if anchor_i is not None:
   path=anchors/f'{anchor_i:02d}_{arm}.npz';np.savez_compressed(path,**arrays)
   c.save(anchors/f'{anchor_i:02d}_{arm}.json',dict(index=item['index'],position=item['position'],arm=arm,response=rec['response'],native_failure=rec['native_failure'],observation_errors=obs['observer_errors'],selection_errors=obs['selection_errors'],intervention_attempts=obs['intervention_attempts'],intervention_applied=obs['intervention_applied'],intervention_error=obs['intervention_error'],vision_calls=obs['vision_calls'],pruner_calls=obs['pruner_calls'],processor_calls=obs['processor_calls'],generate_calls=obs['generate_calls'],llm_calls=obs['llm_calls'],selection_calls=obs['selection_calls'],native_redecode=obs.get('native_redecode'),archive_sha256=c.sha(path),arrays=declared,generate_kwargs_keys=obs.get('generate_kwargs_keys'),generate_non_tensor_kwargs=obs.get('generate_non_tensor_kwargs'),continuation_presence=[{k:step[k] for k in ('visual_mask_present','deepstack_present')} for step in rec['continuation']],continuation_steps=len(rec['continuation']),GT_read=False,scoring_started=False,status='observations_preserved'))
  assert rec['native_failure'] is None and isinstance(rec['response'],str) and not obs['observer_errors'] and not obs['selection_errors']
  assert all(obs[k]==1 for k in ('vision_calls','pruner_calls','processor_calls','generate_calls','selection_calls'))
  assert obs['actual_processor']==item['processor_expected'] and obs['native_redecode']==rec['response']
  assert obs['pruner_split_sizes']==obs['quotas']==[256] and obs['selected_vision_features']==obs['pruner_output']
  assert obs['reconstructed_inputs_embeds']==obs['generate_tensor_kwargs']['inputs_embeds'] and obs['reconstructed_attention_mask']==obs['generate_tensor_kwargs']['attention_mask']
  assert obs['continuation_steps']==obs['llm_calls']-1 and obs['output_ids'] and len(obs['output_ids'])<=2048
  if arm=='ds_off':
   assert obs['intervention_attempts']==0 and not obs['intervention_applied'];baseline_capture(wrapper,rec)
  else:
   assert obs['intervention_attempts']==1 and obs['intervention_applied'] and obs['intervention_error'] is None
   assert all(obs['selected_visual_deepstack'][j]==obs['prefill_final']['deepstack_'+str(j)] for j in range(3))
  if anchor_i is not None:
   prior=c.read(c.FIXED/'execution'/f'{anchor_i:02d}_{arm}.json')
   assert rec['response']==prior['response'] and obs['generate_kwargs_keys']==prior['generate_kwargs_keys'] and obs['generate_non_tensor_kwargs']==prior['generate_non_tensor_kwargs']
   assert declared==prior['arrays'],(anchor_i,arm,'tensor metadata')
  return rec,obs
 def load_anchor(i):
  record=c.read(anchors/f'{i:02d}_ds_off.json');path=anchors/f'{i:02d}_ds_off.npz';assert c.sha(path)==record['archive_sha256']
  with np.load(path,allow_pickle=False) as z:
   def tensor(k):
    t=torch.from_numpy(z[k].copy());dtype=record['arrays'][k]['dtype']
    if dtype=='torch.bfloat16':t=t.view(torch.bfloat16)
    assert str(t.dtype)==dtype;return t.to('cuda:0')
   result=dict(ordered_ids=tensor('ordered_ids')[0],spatial_ids=tensor('spatial_ids'),visual_features=tensor('visual_features'),deepstack=[tensor('visual_deepstack_'+str(j)) for j in range(3)],pruner_input=tensor('pruner_input'),pruner_output=tensor('pruner_output'),split_sizes=[256],processor_input_ids=tensor('processor_input_ids'),processor_mask=tensor('processor_attention_mask'),processor_pixel_values=tensor('processor_pixel_values'),processor_image_grid_thw=tensor('processor_image_grid_thw'),original_token_embeddings=tensor('original_token_embeddings'),rope_deltas=tensor('rope_deltas'),image_token_id=wrapper.model.config.image_token_id)
   result['generation_kwargs']={**record['generate_non_tensor_kwargs'],**{k:tensor('generate_'+k) for k in ('inputs_embeds','attention_mask')}}
   result['original_prefill']={k:tensor('original_prefill_'+k) if 'original_prefill_'+k in z else None for k in ('inputs_embeds','position_ids','attention_mask','cache_position','visual_pos_masks')};result['original_prefill']['deepstack_visual_embeds']=None
   return result
 try:
  for arm in ('ds_off','ds_on'):streams[arm]=((out/f'{arm}.predictions.jsonl').open('x'),(out/f'{arm}.observations.jsonl').open('x'))
  for i,row in enumerate(panel):capture(get_item(row['position']),'ds_off',None,i)
  for i,row in enumerate(panel):
   base=load_anchor(i);capture(get_item(row['position']),'ds_on',base,i);del base
  for pair in streams.values():
   for f in pair:f.flush()
  c.save(out/'native_anchors.json',dict(success=True,n=32,all_fixed16_before_remaining=True,plan_sha256=c.sha(c.PLAN),records={f'{i:02d}_{a}':c.sha(anchors/f'{i:02d}_{a}.json') for i in range(16) for a in ('ds_off','ds_on')},archives={f'{i:02d}_{a}':c.sha(anchors/f'{i:02d}_{a}.npz') for i in range(16) for a in ('ds_off','ds_on')},GT_read=False,scoring_started=False))
  for pos in range(5349):
   if pos in panel_positions:continue
   item=get_item(pos);off,_=capture(item,'ds_off',None);base=baseline_capture(wrapper,off);capture(item,'ds_on',base);del base,off
   if (pos+1)%50==0:print(json.dumps(dict(last_position=pos,remaining_total=5333)),flush=True)
 finally:
  for pair in streams.values():
   for f in pair:f.close()
 from reorder import reorder
 for arm in ('ds_off','ds_on'):
  for kind in ('predictions','observations'):
   assert reorder(out/f'{arm}.{kind}.jsonl',out/f'{arm}.ordered.{kind}.jsonl',items,arm)==5349
 for path,digest in p['files'].items():assert c.sha(path)==digest,path
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in old[group].items():assert c.sha(path)==digest,path
 from namespace_guards import check_model_stats
 from diagnose import stat
 check_model_stats(old,stat,'container')
 for path,digest in {x['image_path']:x['image_sha256'] for x in items}.items():assert c.sha(path)==digest,path
 arms=('ds_off','ds_on')
 c.save(out/'native_complete.json',dict(success=True,n_questions=5349,n_generations=10698,native_anchors=32,plan_sha256=c.sha(c.PLAN),native_anchors_sha256=c.sha(out/'native_anchors.json'),prediction_sha256={a:c.sha(out/f'{a}.predictions.jsonl') for a in arms},observation_sha256={a:c.sha(out/f'{a}.observations.jsonl') for a in arms},ordered_prediction_sha256={a:c.sha(out/f'{a}.ordered.predictions.jsonl') for a in arms},ordered_observation_sha256={a:c.sha(out/f'{a}.ordered.observations.jsonl') for a in arms},GT_read=False,scoring_started=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--allocation-note',type=Path,required=True);args=ap.parse_args()
 try:worker(args.allocation_note.resolve())
 except BaseException as error:
  c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),partial_raw_artifacts_preserved=True));raise
