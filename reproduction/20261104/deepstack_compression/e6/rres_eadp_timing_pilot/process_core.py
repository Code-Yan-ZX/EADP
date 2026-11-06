"""Measured-request process loop; requires separately admitted dependencies.

No command-line entry and no model construction or GPU authority are provided.
"""
import json,os,traceback
from pathlib import Path
import qwen_request,runner

def append(path,value):
 with Path(path).open('a') as f:
  json.dump(value,f,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def run_process(job,rows,references,*,torch,wrapper,module,pruner,adapter_type,criteria_type,fixed_resolution,author_generate,out):
 out=Path(out)
 if not out.is_dir() or job['arm'] not in ('eadp','rres'):raise ValueError('process identity')
 events=out/'journal.jsonl';records=[]
 for index,item in enumerate(job['requests']):
  key=(item['task'],str(item['qid']));row=rows[key];ref=references[(job['arm'],*key)]
  if row['group_id']!=item['group_id'] or row['task']!=job['task']:raise ValueError('request identity')
  wrapper.model.model.rope_deltas=None
  destination=out/('raw_%04d.json'%index);captured={}
  append(events,{'event':'started','index':index,'request':item})
  def persist(raw):
   ids=raw.detach().cpu().tolist()
   if not isinstance(ids,list) or len(ids)!=1 or not isinstance(ids[0],list):raise ValueError('unsupported raw output')
   payload={'request':item,'arm':job['arm'],'full_raw_sequence':ids}
   with destination.open('x') as f:json.dump(payload,f);f.write('\n');f.flush();os.fsync(f.fileno())
   captured['ids']=ids[0]
  try:
   result=qwen_request.request(arm=job['arm'],torch=torch,wrapper=wrapper,module=module,pruner=pruner,row=row,accepted=ref,adapter_type=adapter_type,criteria_type=criteria_type,fixed_resolution=fixed_resolution,author_generate=author_generate,persist=persist)
   prompt=result.pop('rres_prompt_tensor')
   prompt=prompt.detach().cpu().tolist()[0] if prompt is not None else None
   output=runner.normalize(job['arm'],captured['ids'],prompt)
   same_tokens=output==ref['output_ids'];same_text=result['response']==ref['answer']
   result.update(event='completed',index=index,request=item,arm=job['arm'],raw_path=str(destination),output_ids=output,expected_tokens_match=same_tokens,expected_response_match=same_text)
   append(events,result);records.append(result)
  except BaseException as exc:
   append(events,{'event':'failed','index':index,'request':item,'arm':job['arm'],'raw_persisted':destination.is_file(),'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
   raise
 return {'requests':len(records),'all_completed':len(records)==len(job['requests']),'identity_mismatches':sum(not (r['expected_tokens_match'] and r['expected_response_match']) for r in records),'results_retained':True}
