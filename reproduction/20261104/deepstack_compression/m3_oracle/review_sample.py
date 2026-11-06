"""Independent sample-only M3 audit; no predictions, GT or model execution."""
import datetime,hashlib,json,os,time
from pathlib import Path
from collections import Counter
BASE=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('incomplete input file')
 return [json.loads(x) for x in b.splitlines()]
def main():
 started=time.time();errors=[];checks=0;seen=0
 def req(x,msg):
  nonlocal checks
  checks+=1
  if not bool(x):errors.append(msg)
 try:
  sample=read(HERE/'sample_001.json');e1=read(BASE/'e1/e1full001_completion.json');verify=read(BASE/'data_manifest/verification.json')
  req(sample.get('stage')=='sample_only' and sample.get('GT_opened') is False and sample.get('predictions_opened') is False and sample.get('selection_uses_scores') is False,'sample scope')
  for p,d in sample['source_sha256'].items():req(sha(p)==d,'frozen source '+p)
  for name in ('manifest_verify_supervision','e1/monitorverifye1full001','m3_oracle/monitor_freeze002'):
   term=read(BASE/name/'terminal.json');begin=read(BASE/name/'started.json')
   req(term['reason']=='completed' and term['worker_returncode']==0 and term['remaining_process_group']==[] and term['worker_pid']==begin['worker_pid'],'supervisor '+name)
  req(verify['status']=='pass' and e1['status']=='passed' and e1['natural_questions']==384 and e1['GT_opened'] is False,'upstream proof')
  panel=lines(BASE/'data_manifest/panel_manifest.jsonl');inputs=lines(e1['files']['inputs']['path']);req(sha(e1['files']['inputs']['path'])==e1['files']['inputs']['sha256'],'E1 input SHA')
  dev=[x for x in panel if x.get('split')=='DEV'];lookup={(x['task'],str(x['qid'])):x for x in inputs}
  req(len(dev)==len(inputs)==len(lookup)==384 and len({x['group_id'] for x in dev})==384,'frozen DEV denominator')
  expected=[]
  for task,n in (('DocVQA_VAL',64),('TextVQA_VAL',32),('AI2D_TEST',32)):
   rows=[x for x in dev if x['task']==task];req(len(rows)==128,task+' panel count')
   rows.sort(key=lambda x:(hashlib.sha256(('20261104|'+x['group_id']).encode()).hexdigest(),x['group_id']))
   expected += [{k:lookup[(x['task'],str(x['qid']))][k] for k in ('task','qid','group_id','image_path','image_sha256','message_sha256')} for x in rows[:n]]
  req(sample['examples']==expected and sample['counts']=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32},'predeclared selection and E1 identity')
  req(len({r['group_id'] for r in expected})==128 and Counter(x['task'] for x in expected)=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32},'unique stratified groups')
  for x in expected:
   r=lookup[(x['task'],str(x['qid']))];req(sha(r['image_path'])==r['image_sha256'] and hashlib.sha256(json.dumps(r['message'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()==r['message_sha256'],'image/message '+x['task']+':'+str(x['qid']));seen+=1
  score_term=read(BASE/'e4_analysis/monitor_score_panel003/terminal.json');score_time=datetime.datetime.fromisoformat(score_term['utc']).timestamp()
  req(sample.get('selected_after_e4_dev_scoring') is True and sample['created_unix']>score_time,'post-E4 freeze disclosure')
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('exception '+repr(exc))
 out={'status':'passed' if not errors and seen==128 else 'failed','questions_verified':seen,'checks':checks,'errors':errors,'sample_sha256':sha(HERE/'sample_001.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'predictions_opened':False,'scoring_performed':False,'scope':'independent frozen DEV sample reconstruction only; not teacher or GPU approval','selected_after_e4_dev_scoring':True,'start_unix':started,'end_unix':time.time()}
 with (HERE/'sample_001_independent_review.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(out));return 0 if out['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
