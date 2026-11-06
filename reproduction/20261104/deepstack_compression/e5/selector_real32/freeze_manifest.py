"""Freeze the pilot32 plus a deterministic 128-image DEV extension; no predictions or GT."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def group_order(g):return hashlib.sha256(('20261104|'+g).encode()).hexdigest()
def build(rows,pilot):
 index={(r['task'],str(r['qid'])):r for r in rows};groups={(r['task'],r['group_id']) for r in rows};tasks=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');quota=dict(zip(tasks,(43,43,42)))
 if len(rows)!=384 or len(index)!=384 or len(groups)!=384 or len(pilot)!=32:raise ValueError('DEV or pilot inventory')
 chosen=[]
 for task in tasks:
  frozen=[r for r in pilot if r['task']==task];pool=[r for r in rows if r['task']==task and (r['task'],r['group_id']) not in {(x['task'],x['group_id']) for x in frozen}]
  for x in frozen:
   y=index.get((x['task'],str(x['qid'])))
   if y is None or any(y.get(k)!=x.get(k) for k in ('group_id','message_sha256','image_sha256')):raise ValueError('pilot identity mismatch')
  if len(frozen)>quota[task] or len(pool)<quota[task]-len(frozen):raise ValueError('quota unavailable')
  chosen.extend(sorted(frozen+sorted(pool,key=lambda r:(group_order(r['group_id']),str(r['qid'])))[:quota[task]-len(frozen)],key=lambda r:(group_order(r['group_id']),str(r['qid']))))
 if len(chosen)!=128 or len({r['group_id'] for r in chosen})!=128:raise ValueError('cross-task group overlap')
 return [{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in chosen]

def main():
 inputs=BASE/'e1/input_preparation_002/inputs.jsonl';schedule=BASE/'e4_c_microbench/schedule001.json';rows=[json.loads(x) for x in inputs.read_text().splitlines() if x.strip()];p=json.loads(schedule.read_text())['input_identities'];items=build(rows,p)
 out={'status':'frozen','scope':'32 image pilot; 128 DEV images x 3 budgets for eventual 384 cases','seed':'20261104','pilot_after_observed_timing':True,'full_sample_order':'task DocVQA, TextVQA, AI2D; within task SHA256(20261104|group_id), retain every pilot then fill earliest other DEV groups','task_quotas':{'DocVQA_VAL':43,'TextVQA_VAL':43,'AI2D_TEST':42},'budgets':[128,256,512],'pilot':[ {k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in p], 'full_128':items,'inputs_sha256':sha(inputs),'schedule_sha256':sha(schedule),'freeze_script_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'gpu_used':False}
 with (HERE/'sample_manifest_001.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
