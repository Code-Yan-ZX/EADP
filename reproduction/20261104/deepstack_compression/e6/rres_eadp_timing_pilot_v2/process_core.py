"""Measured-request process loop; requires separately admitted dependencies.

No command-line entry and no model construction or GPU authority are provided.
"""
import hashlib,json,os,traceback
from pathlib import Path
import qwen_request,runner

def append(path,value):
 with Path(path).open('a') as f:
  json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def processor_evidence(torch,inputs):
 result={}
 for name in ('input_ids','attention_mask','pixel_values','image_grid_thw'):
  x=inputs[name].detach().contiguous().cpu()
  result[name]={'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
 return result
def run_process(job,rows,references,*,torch,wrapper,module,pruner,adapter_type,criteria_type,fixed_resolution,author_generate,out):
 out=Path(out)
 if not out.is_dir() or job['arm'] not in ('eadp','rres'):raise ValueError('process identity')
 events=out/'journal.jsonl';records=[]
 for index,item in enumerate(job['requests']):
  key=(item['task'],str(item['qid']))
  destination=out/('raw_%04d.json'%index);captured={}
  append(events,{'event':'started','index':index,'request':item})
  def persist(raw):
   ids=raw.detach().cpu().tolist()
   if not isinstance(ids,list) or len(ids)!=1 or not isinstance(ids[0],list):raise ValueError('unsupported raw output')
   payload={'request':item,'arm':job['arm'],'full_raw_sequence':ids}
   with destination.open('x') as f:json.dump(payload,f);f.write('\n');f.flush();os.fsync(f.fileno())
   captured['ids']=ids[0]
  try:
   row=rows[key];ref=references[(job['arm'],*key)]
   if row['group_id']!=item['group_id'] or row['task']!=job['task']:raise ValueError('request identity')
   wrapper.model.model.rope_deltas=None
   result=qwen_request.request(arm=job['arm'],torch=torch,wrapper=wrapper,module=module,pruner=pruner,row=row,accepted=ref,adapter_type=adapter_type,criteria_type=criteria_type,fixed_resolution=fixed_resolution,author_generate=author_generate,persist=persist)
   inputs=result.pop('processor_inputs');prompt=inputs['input_ids'].detach().cpu().tolist()[0]
   output=runner.normalize(job['arm'],captured['ids'],prompt)
   same_tokens=output==ref['output_ids'];same_text=result['response']==ref['answer']
   actual_processor=processor_evidence(torch,inputs);image_hash=file_sha(row['image_path']);first_expected=len(prompt)+1 if job['arm']=='rres' else 1
   checks={'tokens_match':same_tokens,'response_match':same_text,'processor_match':actual_processor==ref['processor'],'image_match':image_hash==row['image_sha256'],'first_criteria_length_match':result['times']['first_criteria_length']==first_expected,'criteria_count_match':result['times']['criteria_calls']==len(output)}
   result.update(event='completed',index=index,request=item,arm=job['arm'],raw_path=str(destination),output_ids=output,processor=actual_processor,image_sha256=image_hash,checks=checks)
   append(events,result);records.append(result)
  except BaseException as exc:
   append(events,{'event':'failed','index':index,'request':item,'arm':job['arm'],'raw_persisted':destination.is_file(),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
   raise
 return {'requests':len(records),'all_completed':len(records)==len(job['requests']),'identity_mismatches':sum(not all(r['checks'].values()) for r in records),'results_retained':True}
