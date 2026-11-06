"""Freeze a second disjoint DEV oracle batch without predictions or labels."""
import collections,hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
TASKS=(('DocVQA_VAL',64),('TextVQA_VAL',32),('AI2D_TEST',32))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def order(g):return (hashlib.sha256(('20261104|'+g).encode()).hexdigest(),g)
def main():
 panel=ROOT/'data_manifest/panel_manifest.jsonl';summary=ROOT/'data_manifest/manifest_summary.json';verify=ROOT/'data_manifest/verification.json';e1p=ROOT/'e1/e1full001_completion.json';first=ROOT/'m3_oracle/sample_001.json';first_review=ROOT/'m3_oracle/sample_001_independent_review.json';protocol=ROOT/'m3_oracle/protocol_001.json';amend=ROOT/'m3_oracle_exec/protocol_amendment_001.json';verified=ROOT/'m3_oracle_verify_v2/runtime_001_verification_author.json';independent=ROOT/'e1_independent_review/m3oracle_panel_review.json';score=ROOT/'m3_oracle_analysis_v2/score_001/analysis.json';score_review=ROOT/'e1_independent_review/m3oracle_scores_review.json'
 ds=read(summary);validation=read(verify);e1=read(e1p);prior=read(first);pr=read(first_review);formal=read(verified);ir=read(independent);sr=read(score_review)
 assert validation['status']=='pass' and ds['outputs_sha256']['panel_manifest.jsonl']==sha(panel) and e1['status']=='passed' and e1['natural_questions']==384 and e1['GT_opened'] is False
 assert pr['status']=='passed' and pr['sample_sha256']==sha(first) and prior['GT_opened'] is False and prior['predictions_opened'] is False
 assert formal['status']=='passed' and formal['n_questions']==128 and ir['status']=='passed' and ir['formal_proof_sha256']==sha(verified)
 assert sr['status']=='passed' and sr['questions']==128 and sr['scores_verified']==896 and sr['analysis_sha256']==sha(score)
 for path in (ROOT/'manifest_verify_supervision/terminal.json',ROOT/'e1/monitorverifye1full001/terminal.json',ROOT/'e1_independent_review/monitor_m3oracle_panel001/terminal.json',ROOT/'e1_independent_review/monitor_m3oracle_scores001/terminal.json'):
  t=read(path);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 inputs=Path(e1['files']['inputs']['path']);assert sha(inputs)==e1['files']['inputs']['sha256']
 rows=[r for r in lines(panel) if r['split']=='DEV'];inp=lines(inputs);lookup={(r['task'],str(r['qid'])):r for r in inp};assert len(rows)==len(inp)==len(lookup)==384 and len({r['group_id'] for r in rows})==384
 old=prior['examples'];assert len(old)==128 and len({x['group_id'] for x in old})==128;old_groups={x['group_id'] for x in old}
 result=[]
 for task,count in TASKS:
  candidates=sorted((r for r in rows if r['task']==task),key=lambda r:order(r['group_id']));previous=[x for x in old if x['task']==task]
  assert len(candidates)==128 and len(previous)==count and [x['group_id'] for x in previous]==[x['group_id'] for x in candidates[:count]]
  available=[x for x in candidates if x['group_id'] not in old_groups];assert len(available)>=count
  for row in available[:count]:
   x=lookup[(task,str(row['qid']))];assert x['group_id']==row['group_id']
   result.append({k:x[k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')})
 assert len(result)==128 and len({x['group_id'] for x in result})==128 and not ({x['group_id'] for x in result}&old_groups)
 deps=[panel,summary,verify,e1p,inputs,first,first_review,protocol,amend,verified,independent,score,score_review,Path(__file__)]
 output={'stage':'sample_only','batch':2,'created_unix':time.time(),'selection':'next unexposed DEV groups in frozen SHA256(20261104|group_id) order within each task',
         'selection_after_first_batch_scores':True,'selection_uses_scores':False,'GT_opened':False,'predictions_opened':False,
         'counts':dict(TASKS),'first_batch_group_count':len(old_groups),'source_sha256':{str(x):sha(x) for x in deps},'examples':result,
         'limitations':['Second exploratory look after observing batch 1; report batches separately.','No new labels, predictions, GPU, teacher or inference have been used.','Not a deployment score or student confirmation set.']}
 path=HERE/'sample_002.json'
 with path.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'sample_frozen','questions':128,'sha256':sha(path)}))
if __name__=='__main__':main()
