"""Two full Doc routes using one native model, no labels or scoring."""
import argparse,hashlib,importlib.metadata,json,os,sys
from pathlib import Path
import common as c

def worker(note_path):
 p=c.read(c.PLAN);c.check(p,'container');note=c.read(note_path)
 assert note['authorization']=='explicit_root_gpu_allocation' and note['run_id']==c.RUN_ID and note['plan_sha256']==c.sha(c.PLAN)
 assert note['GPU_device']==0 and note['single_attempt'] and not note['automatic_retry']
 old=c.read(c.OLD/'run_plan.json');assert Path(sys.executable).resolve()==Path(old['python']).resolve()
 out=c.HERE/'execution';out.mkdir(exist_ok=False)
 c.save(out/'identity.json',dict(**c.identity(os.getpid()),plan_sha256=c.sha(c.PLAN),allocation_sha256=c.sha(note_path)))
 import torch
 torch.set_num_threads(4)
 sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatCDPruner,Qwen3VLChatEADP
 assert {n:importlib.metadata.version(n) for n in old['expected_versions']}==old['expected_versions']
 recipe=dict(old['recipe']);recipe['visual_token_num']=1024
 wrapper=Qwen3VLChatEADP(**recipe)
 assert wrapper.visual_token_num==wrapper.pruner.visual_token_num==1024 and wrapper.post_process is False
 assert wrapper.model.config._attn_implementation==wrapper.model.visual.config._attn_implementation=='flash_attention_2'
 assert wrapper.model.dtype==wrapper.model.visual.dtype==torch.bfloat16
 assert {str(x.device) for x in wrapper.model.parameters()}=={'cuda:0'} and torch.cuda.get_device_name(0)=='NVIDIA A40'
 actual_sources={n:str(Path(sys.modules[n].__file__).resolve()) for n in ['vlmeval.vlm.qwen3_vl.model_fixed_res',
  'vlmeval.vlm.qwen3_vl.model','model.pruner','transformers.models.qwen3_vl.modeling_qwen3_vl']}
 for path in actual_sources.values():assert c.sha(path)==old['source_sha256'][path]
 c.save(out/'runtime.json',dict(actual_recipe=recipe,actual_sources=actual_sources,model_dtype=str(wrapper.model.dtype),
  backend=wrapper.model.config._attn_implementation,post_process=wrapper.post_process,GPU=torch.cuda.get_device_name(0),model_instances=1,plan_sha256=c.sha(c.PLAN)))
 wrapper.set_dump_image(lambda row:[row['image_path']])
 items=c.rows(c.HERE/'input_manifest.jsonl')
 def tensor_meta(t):
  actual=t.detach().cpu().contiguous();a=actual.view(torch.uint16).numpy() if actual.dtype==torch.bfloat16 else actual.numpy()
  return dict(dtype=str(actual.dtype),shape=list(actual.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
 streams={}
 try:
  for arm in ('plain','eadp1024'):
   streams[arm]=(out/(arm+'.predictions.jsonl')).open('x');streams[arm+'_obs']=(out/(arm+'.observations.jsonl')).open('x')
  for item in items:
   msg=wrapper._build_vqa_prompt(item,'DocVQA_VAL');messages=wrapper._build_messages(msg,dataset='DocVQA_VAL')
   assert hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==item['message_sha256']
   for arm in ('plain','eadp1024'):
    cap={};errors=[];old_generate=wrapper.model.generate;old_processor=wrapper._processor_inputs
    def processor(*args,**kwargs):
     result=old_processor(*args,**kwargs)
     try:cap['inputs']={key:value.detach().clone() for key,value in result.items()}
     except Exception as error:errors.append('processor snapshot: '+repr(error))
     return result
    def generate(*args,**kwargs):
     result=old_generate(*args,**kwargs)
     try:cap['output_ids']=result.detach().clone()
     except Exception as error:errors.append('output snapshot: '+repr(error))
     return result
    wrapper._processor_inputs=processor;wrapper.model.generate=generate
    failure=None;prediction=None
    try:
     method=Qwen3VLChatFixedRes.generate_inner_transformers_fixed if arm=='plain' else Qwen3VLChatCDPruner.generate_inner_transformers_fixed
     prediction=method(wrapper,msg,dataset='DocVQA_VAL')
    except BaseException as error:failure=repr(error)
    finally:wrapper._processor_inputs=old_processor;wrapper.model.generate=old_generate
    # Preserve the actual response, including empty strings, before any observer gate.
    from contracts import preserve_raw
    preserve_raw(streams[arm],item,prediction,failure)
    obs=dict(position=item['position'],index=item['index'],observer_errors=errors,native_failure=failure)
    try:
     inputs=cap['inputs'];output=cap['output_ids'];metadata={key:tensor_meta(value) for key,value in inputs.items()}
     suffix=output[:,inputs['input_ids'].shape[1]:] if arm=='plain' else output
     ids=suffix.detach().cpu().tolist()
     decoded=wrapper.processor.tokenizer.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
     obs.update(actual_processor=metadata,output_suffix_ids=ids[0],native_redecode=decoded,
      decode_matches=(decoded==prediction),num_image_tokens=int((inputs['input_ids']==wrapper.model.config.image_token_id).sum().item()))
    except Exception as error:obs['observer_errors'].append('post-generation inspection: '+repr(error))
    streams[arm+'_obs'].write(json.dumps(obs,ensure_ascii=False)+'\n');streams[arm+'_obs'].flush()
    from contracts import check_observation
    check_observation(item,prediction,failure,obs)
   if (item['position']+1)%50==0:print(json.dumps(dict(completed_pairs=item['position']+1,total=5349)),flush=True)
 finally:
  for stream in streams.values():stream.close()
 # Preserve all output on any later source check failure.
 for group in ('source_sha256','implementation_sha256','reference_sha256'):
  for path,digest in old[group].items():assert c.sha(path)==digest,path
 from namespace_guards import check_model_stats
 from diagnose import stat
 check_model_stats(old,stat,'container')
 for path,digest in p['file_guards'].items():assert c.sha(path)==digest,path
 for path,digest in {r['image_path']:r['image_sha256'] for r in items}.items():assert c.sha(path)==digest,path
 c.save(out/'native_complete.json',dict(success=True,counts={'plain':5349,'eadp1024':5349},
  predictions={arm:c.sha(out/(arm+'.predictions.jsonl')) for arm in ('plain','eadp1024')},
  observations={arm:c.sha(out/(arm+'.observations.jsonl')) for arm in ('plain','eadp1024')},
  plan_sha256=c.sha(c.PLAN),all_generation_before_scoring=True,GT_read=False,scoring_started=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--allocation-note',type=Path,required=True);args=ap.parse_args()
 try:worker(args.allocation_note.resolve())
 except BaseException as error:
  c.save(c.HERE/'worker_failure.json',dict(status='terminal_failure_no_retry',error=repr(error),partial_predictions_preserved=True));raise
