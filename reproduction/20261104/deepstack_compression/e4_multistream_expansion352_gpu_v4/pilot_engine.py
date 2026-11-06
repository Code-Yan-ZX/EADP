"""Nine-image diagnostic engine; caller must enforce reviewed GPU admission.

The caller supplies already source-verified real-model functions. This module
is not a launcher and does not itself authorize or initialize CUDA.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def save_json(path,value):
 with Path(path).open('x') as f:json.dump(value,f,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def run(selected,rows,refs,e4_records,*,out,torch,wrapper,native,author_generate,Adapter,Capture,clone,capture_one,preflight_one,validate_e1,tensor_meta,identity_check,norm_mod,feature_mod,journal,clock=time.monotonic):
 if len(selected)!=9 or len({(r['task'],str(r['qid'])) for r in selected})!=9:raise ValueError('pilot cohort')
 out=Path(out);out.mkdir(exist_ok=False);index={(r['task'],str(r['qid'])):r for r in rows}
 if len(index)!=384:raise ValueError('E1 input inventory')
 statuses=[]
 for i,target in enumerate(selected):
  key=(target['task'],str(target['qid']));row=index.get(key);ref=refs.get(key);e4=e4_records.get(key)
  journal({'status':'image_started','index':i,'task':key[0],'qid':key[1]})
  try:
   if row is None or ref is None or e4 is None or row.get('split')!='DEV' or any(str(row.get(k))!=str(target.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('sample identity')
   start=clock();preflight_one(torch,wrapper,row,ref);preflight_seconds=clock()-start
   torch.cuda.synchronize();start=clock()
   ops,base,actual,clean=capture_one(torch,wrapper,native,row,author_generate,Adapter,Capture,clone,journal)
   torch.cuda.synchronize();capture_seconds=clock()-start
   if not clean():raise ValueError('cleanup failed')
   evidence=validate_e1(actual,ref,tensor_meta)
   meta=dict(target,processor={k:tensor_meta(actual['processor'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},main_visual=tensor_meta(actual['visual'][0]),full_deepstack=[tensor_meta(x) for x in actual['visual'][1]])
   identity_check(target,meta,e4)
   visual=actual['visual'];prefill=actual['prefill'];ordered=actual['ordered'].detach().cpu().reshape(-1).tolist()
   # Store actual metadata/coordinates, never full high-dimensional tensors.
   actual_identity={'processor':meta['processor'],'visual_sha256':meta['main_visual'],'deepstack_full_sha256':meta['full_deepstack'],'ordered_ids':ordered,'spatial_ids':sorted(ordered),'prefill_embed_sha256':tensor_meta(prefill['inputs_embeds']),'attention_mask_sha256':tensor_meta(prefill['attention_mask']),'prefill_position_ids':prefill['position_ids'].detach().cpu().tolist(),'rope_delta':actual['rope_delta'].detach().cpu().tolist(),'injected_mask':prefill['visual_pos_masks'].detach().cpu().tolist(),'injected_deepstack':[tensor_meta(x) for x in prefill['deepstack_visual_embeds']]}
   if any(actual_identity[k]!=ref[k] for k in actual_identity):raise ValueError('actual full E1 identity mismatch')
   start=clock();streams=[x.detach().to(dtype=torch.float64,device='cpu').numpy() for x in (actual['visual'][0],*actual['visual'][1])];values=norm_mod.norms(streams);features=norm_mod.features_from_norms(values,feature_mod);feature_seconds=clock()-start
   path=out/('image_%03d_norms.npy'%i);start=clock()
   with path.open('xb') as f:np.save(f,values,allow_pickle=False);f.flush();os.fsync(f.fileno())
   serialization_seconds=clock()-start
   item={'status':'passed','index':i,**{k:target[k] for k in ('task','qid','group_id','image_sha256','message_sha256')},'features':features,'norms_path':str(path),'norms_sha256':sha(path),'e1_identity':evidence,'actual_identity':actual_identity,'e4_meta':meta,'timing_seconds':{'processor_preflight':preflight_seconds,'capture_including_processor_visual_selector_and_one_token_generation':capture_seconds,'features_and_cpu_transfer':feature_seconds,'norms_write_and_fsync':serialization_seconds},'hooks_clean':True,'scope':'diagnostic cost, not TTFT'}
   save_json(out/('image_%03d.json'%i),item);journal({'status':'image_completed','index':i,'task':key[0],'qid':key[1],'norms_sha256':item['norms_sha256']});statuses.append({'task':key[0],'qid':key[1],'status':'passed'})
  except BaseException as exc:
   journal({'status':'image_failed','index':i,'task':key[0],'qid':key[1],'error_type':type(exc).__name__,'message':str(exc)});statuses.append({'task':key[0],'qid':key[1],'status':'failed'})
   for j in range(i+1,len(selected)):
    future=selected[j];journal({'status':'image_not_evaluated','index':j,'task':future['task'],'qid':str(future['qid']),'reason':'earlier_failure'});statuses.append({'task':future['task'],'qid':str(future['qid']),'status':'not_evaluated'})
   save_json(out/'status.json',{'status':'failed','images_planned':9,'images_passed':i,'rows':statuses,'GT_opened':False,'scoring_performed':False});raise
 save_json(out/'status.json',{'status':'passed','images_planned':9,'images_passed':9,'rows':statuses,'GT_opened':False,'scoring_performed':False})
 return statuses
