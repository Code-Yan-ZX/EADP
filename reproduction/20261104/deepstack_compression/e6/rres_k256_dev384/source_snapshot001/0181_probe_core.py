"""Low-resolution processor contract on each frozen input; no model/GPU."""
import json,os,traceback
from pathlib import Path
from design import panel
from panel_core import sha,key,write,event
from processor_contract import check
def run(rows,anchors,refs,torch,wrapper,module,config,fixed_resolution,validate_original,rope_index,out,verify_image=sha):
 anchors=panel(anchors);out=Path(out);index={key(x):x for x in rows};expected=[key(x) for x in anchors]
 if len(rows)!=384 or len(index)!=384 or set(index)!=set(expected) or len(refs)!=384 or set(refs)!=set(expected) or not out.is_dir():raise ValueError('incomplete input identity inventory')
 results=[];started=[];journal=out/'processor_progress.jsonl'
 try:
  for a in anchors:
   k=key(a);r=index[k];ref=refs[k]
   if any(str(r.get(x))!=str(a[x]) for x in ('task','qid','group_id','image_sha256','message_sha256')) or ref['processor']!=a['processor'] or ref['image_start']!=a['image_start']:raise ValueError('E1 input mismatch')
   event(journal,{'event':'started','task':k[0],'qid':k[1],'group_id':a['group_id']});started.append(k)
   if verify_image(r['image_path'])!=a['image_sha256']:raise ValueError('image hash')
   messages=wrapper._build_messages(r['message'],dataset=r['task']);original=wrapper._processor_inputs(messages)
   with fixed_resolution(module,512):low=wrapper._processor_inputs(messages)
   if module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('resolution restoration')
   value=check(torch,original,low,config,ref,validate_original,rope_index)
   record={**{k:a[k] for k in ('task','qid','group_id','image_sha256','message_sha256')},**value};results.append(record)
   event(journal,{'event':'verified','task':k[0],'qid':k[1],'group_id':a['group_id'],'low_visual_tokens':256})
  if len(results)!=384 or torch.cuda.is_initialized():raise ValueError('incomplete processor probe or CUDA initialized')
  write(out/'processor_probe.json',{'status':'passed','images':384,'anchors':results,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'scoring_performed':False,'errors':[]})
  return results
 except BaseException as exc:
  write(out/'processor_failure.json',{'status':'failed','expected':384,'started':len(started),'verified':len(results),'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'GT_opened':False});raise
