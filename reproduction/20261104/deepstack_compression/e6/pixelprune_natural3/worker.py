"""Author-order PixelPrune natural generation diagnostic, three frozen DEV images.

This is a diagnostic with observation hooks and no score or timing claim. The
published class patch is applied before model construction, unlike the previous
same-model identity smoke. All prediction exposures are journalled promptly.
"""
import argparse,hashlib,inspect,json,os,sys,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/media/disk2/YZX/research/EADP_v2');BASE=ROOT/'reproduction/20261104/deepstack_compression';HERE=BASE/'e6/pixelprune_natural3';PIXEL=BASE/'e6/external_sources/pixelprune_8d347eb';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
sys.path[:0]=[str(AUTHOR),str(PIXEL)]
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 with open(path,'x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,obj):
 with open(path,'a') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def verify_sources(plan):
 for path,h in plan['sha256'].items():
  if sha(path)!=h:raise ValueError('source hash mismatch: '+path)
def preflight(plan):
 verify_sources(plan)
 rows=[json.loads(x) for x in (BASE/'e1/input_preparation_002/inputs.jsonl').read_text().splitlines()];index={(r['task'],str(r['qid'])):r for r in rows}
 if len(index)!=len(rows):raise ValueError('duplicate DEV input')
 chosen=[]
 for anchor in plan['anchors']:
  r=index[(anchor['task'],anchor['qid'])]
  digest=hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
  if r['message_sha256']!=anchor['message_sha256'] or digest!=anchor['message_sha256'] or sha(r['image_path'])!=anchor['image_sha256']:raise ValueError('frozen input mismatch')
  chosen.append(r)
 if len(chosen)!=3 or len({r['task'] for r in chosen})!=3:raise ValueError('three task anchors required')
 return chosen
def weight_stats(weights):
 result=[]
 for w in weights['files']:
  p=Path(w['path']);s=p.stat()
  if s.st_size!=w['bytes'] or w['sha256']!=w['expected_lfs_sha256']:raise ValueError('weight identity mismatch')
  result.append({'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'inode':s.st_ino,'device':s.st_dev})
 if len(result)!=4:raise ValueError('four weights required')
 return result
def selected_indices(keep):
 if keep is None:return list(range(1024))
 if len(keep)!=1:raise ValueError('single image expected')
 idx=keep[0].detach().cpu().tolist()
 if len(idx)%4:raise ValueError('incomplete patch group')
 ids=[idx[i]//4 for i in range(0,len(idx),4)]
 if idx!=[4*j+k for j in ids for k in range(4)] or ids!=sorted(set(ids)) or not ids or ids[-1]>=1024:raise ValueError('invalid patch ordering')
 return ids
def validate_record(r):
 n=r.get('selected_count');seq=r.get('sequence_length');trace=r.get('trace',[]);ids=r.get('selected_indices',[])
 checks={'unique_selection':type(n) is int and 1<=n<=1024 and len(ids)==n and ids==sorted(set(ids)) and all(0<=i<1024 for i in ids),
 'vision_and_three_ds':r.get('visual_rows')==n and r.get('ds_rows')==[n]*3 and r.get('prefill_ds_rows')==[n]*3 and r.get('prefill_visual_count')==n,
 'same_main_text_positions':r.get('prefill_main_match') is True and r.get('prefill_text_match') is True and r.get('prefill_positions_match') is True and r.get('prefill_ds_match')==[True]*3,
 'natural_complete':r.get('status')=='completed' and 0<r.get('generated_count',0)<=2048 and (r.get('ends_eos') is True or r.get('generated_count')==2048),
 'calls_match_tokens':len(trace)==r.get('generated_count') and len(trace)>0,
 'hooks_removed':r.get('hook_cleanup_valid') is True}
 if type(n) is int and type(seq) is int and trace:
  initial=seq-1024+n;delta=r['original_rope_delta'];maxpos=r['original_max_position']
  checks['positions_and_kv']=all(t.get('rope_delta')==delta==t.get('model_rope_delta') and len(t.get('layers',[]))==36 and all(z['key'][-2]==initial+i and z['value'][-2]==initial+i for z in t['layers']) and (i==0 or (t['position_ids']==[[[maxpos+i]]]*3 and t['cache_position']==[seq+i-1])) for i,t in enumerate(trace))
 else:checks['positions_and_kv']=False
 return checks
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',required=True,type=Path);a=p.parse_args();plan=json.loads((HERE/'plan001.json').read_text());chosen=preflight(plan)
 import torch,transformers
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 import pixelprune.patches.qwen3_vl_hf as patch
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 if Path(inspect.getfile(patch)).resolve()!=PIXEL/'pixelprune/patches/qwen3_vl_hf.py':raise ValueError('foreign PixelPrune source')
 weights=json.loads((ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text());stats=weight_stats(weights)
 # Explicitly set the author's actual patch before any model instantiation.
 if getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('preexisting patch')
 apply_pixelprune(model='qwen3_vl')
 if (hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLModel.get_image_features is not patch._model_get_image_features or hf.Qwen3VLModel.forward is not patch._model_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward):raise ValueError('patch dispatch mismatch')
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
  for w in weights['files']:
   if sha(w['path'])!=w['expected_lfs_sha256']:raise ValueError('weight payload mismatch')
  if weight_stats(weights)!=stats:raise ValueError('weights changed')
  write(a.out,{'status':'passed','source_plan_sha256':sha(HERE/'plan001.json'),'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,'weight_sha256_verified':4,'weight_stats':stats,'patch_before_model':True,'torch':torch.__version__,'transformers':transformers.__version__});return
 if stats!=json.loads((HERE/'preflight001.json').read_text())['weight_stats'] or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('weight/GPU identity')
 a.out.mkdir(exist_ok=False);journal=a.out/'progress.jsonl';append(journal,{'event':'worker_started','patch_before_model':True})
 os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD='0.0',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false');os.environ.pop('PIXELPRUNE_LOG_FILE',None)
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
 wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cuda',dtype=torch.bfloat16)
 eos=model.generation_config.eos_token_id;eos=[eos] if type(eos) is int else list(eos or [])
 if not eos:raise ValueError('no EOS token configured')
 kwargs={'do_sample':False,'max_new_tokens':2048,'top_p':0.8,'top_k':20,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 outcomes=[]
 for r in chosen:
  key=r['task']+'_'+str(r['qid']);append(journal,{'event':'arm_started','task':r['task'],'qid':str(r['qid'])})
  inputs=wrapper._processor_inputs(wrapper._build_messages(r['message'],dataset=r['task']));ids=inputs['input_ids'];seq=ids.shape[1];orig,delta=model.model.get_rope_index(ids,inputs['image_grid_thw'],None,inputs['attention_mask']);visual=torch.where(ids[0]==model.config.image_token_id)[0]
  if len(visual)!=1024:raise ValueError('unexpected processor budget')
  text=model.get_input_embeddings()(ids).detach();selected=[];visual_out=[];prefill={};trace=[];model.model.rope_deltas=None
  def pre_visual(_m,_a,kw):selected.append(selected_indices(kw.get('keep_indices')))
  def post_visual(_m,_a,_kw,out):visual_out.append((out[0].detach(),[x.detach() for x in out[1]]))
  def pre_lm(_m,_a,kw):
   pos=kw.get('position_ids');emb=kw.get('inputs_embeds');mask=kw.get('visual_pos_masks');ds=kw.get('deepstack_visual_embeds')
   if not trace:
    if len(selected)!=1 or len(visual_out)!=1 or emb is None or mask is None or ds is None:raise ValueError('missing actual prefill')
    idx=torch.tensor(selected[0],device=ids.device);keep=torch.ones(seq,device=ids.device,dtype=torch.bool);keep[visual]=False;keep[visual[idx]]=True;v,origds=visual_out[0];vm=mask[0].bool()
    prefill.update(main=bool(torch.equal(emb[0,vm],v.to(emb.dtype))),text=bool(torch.equal(emb[0,~vm],text[0,ids[0]!=model.config.image_token_id])),positions=bool(torch.equal(pos,orig[:,:,keep])),ds=[bool(torch.equal(x,y.to(x.dtype))) for x,y in zip(ds,origds)],ds_rows=[int(x.shape[0]) for x in ds],visual_count=int(mask.sum()))
   trace.append({'position_ids':None if pos is None else pos.detach().cpu().tolist(),'cache_position':None if kw.get('cache_position') is None else kw['cache_position'].detach().cpu().tolist()})
  def post_model(_m,_a,out):
   if len(trace)!=len(records)+1:raise ValueError('unexpected forward sequence')
   records.append({'position_ids':trace[-1]['position_ids'],'cache_position':trace[-1]['cache_position'],'rope_delta':None if out.rope_deltas is None else out.rope_deltas.detach().cpu().tolist(),'model_rope_delta':None if model.model.rope_deltas is None else model.model.rope_deltas.detach().cpu().tolist(),'layers':[{'key':list(c.keys.shape),'value':list(c.values.shape)} for c in out.past_key_values.layers]})
   append(journal,{'event':'forward_complete','task':r['task'],'qid':str(r['qid']),'index':len(records)-1})
  records=[];watched=[model.model.visual,model.model.language_model,model];old=[(tuple(x._forward_pre_hooks),tuple(x._forward_hooks)) for x in watched];hooks=[model.model.visual.register_forward_pre_hook(pre_visual,with_kwargs=True),model.model.visual.register_forward_hook(post_visual,with_kwargs=True),model.model.language_model.register_forward_pre_hook(pre_lm,with_kwargs=True),model.register_forward_hook(post_model)];result={'task':r['task'],'qid':str(r['qid']),'message_sha256':r['message_sha256'],'image_sha256':r['image_sha256'],'sequence_length':int(seq),'original_rope_delta':delta.detach().cpu().tolist(),'original_max_position':int(orig.max()),'generation_kwargs':kwargs,'eos_token_ids':eos}
  try:
   with torch.inference_mode():generated=model.generate(**inputs,**kwargs)
   torch.cuda.synchronize();tokens=generated[0,seq:].tolist();write(a.out/(key+'_generated.json'),{'task':r['task'],'qid':str(r['qid']),'tokens':tokens})
   result.update(status='completed',generated_count=len(tokens),ends_eos=bool(tokens and tokens[-1] in eos),decoded=processor.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False))
  except Exception as exc:
   result.update(status='failed',error_type=type(exc).__name__,error=str(exc));write(a.out/(key+'_exception.json'),{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
  finally:
   for h in hooks:h.remove()
  result.update(selected_indices=selected[0] if selected else [],selected_count=len(selected[0]) if selected else None,visual_rows=int(visual_out[0][0].shape[0]) if visual_out else None,ds_rows=[int(x.shape[0]) for x in visual_out[0][1]] if visual_out else [],prefill_ds_rows=prefill.get('ds_rows'),prefill_visual_count=prefill.get('visual_count'),prefill_main_match=prefill.get('main'),prefill_text_match=prefill.get('text'),prefill_positions_match=prefill.get('positions'),prefill_ds_match=prefill.get('ds'),trace=records,hook_cleanup_valid=old==[(tuple(x._forward_pre_hooks),tuple(x._forward_hooks)) for x in watched])
  result['diagnostic_checks']=validate_record(result);write(a.out/(key+'.json'),result);append(journal,{'event':'arm_recorded','task':r['task'],'qid':str(r['qid']),'status':result['status']});outcomes.append(result)
 if weight_stats(weights)!=stats:raise ValueError('weights changed during run')
 verify_sources(plan);write(a.out/'worker_complete.json',{'status':'diagnostic_completed','arm_statuses':{x['task']+'/'+x['qid']:x['status'] for x in outcomes},'GT_opened':False,'scoring_performed':False,'warning':'completion means attempts recorded; inspect each arm and diagnostic checks'})
if __name__=='__main__':main()
