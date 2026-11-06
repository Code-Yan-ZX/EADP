"""Bounded label-free, independent-review-gated document split execution."""
import hashlib,json,os,re,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=ROOT/'e1_independent_review'
if 'core' in sys.modules or hashlib.sha256((HERE/'core.py').read_bytes()).hexdigest()!='39e71a1242244da2f6efbc16ac81f9a1e190d44de534efb3c2844e8e29a81a25':raise ValueError('preloaded or changed core')
import core
if Path(core.__file__).resolve()!=(HERE/'core.py').resolve():raise ValueError('wrong core module')
PLAN_SHA='26b1a37db5480b0ef8b799e83866c438c5a03b88577b7b73fc8383f2f24f6a89'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def supervisor(mon,script,args):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if not (t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and cwd==ROOT and len(cmd)==2+len(args) and (cwd/cmd[1]).resolve()==script.resolve() and cmd[2:]==args):raise ValueError('supervisor evidence')
def check_reviewers(pre,ind,doc):
 if pre.get('status')!='passed' or pre.get('independent_review') is not True or pre.get('reviewer_sha256')!=sha(R/'review_train_audit_prelaunch.py') or len(pre.get('source_sha256',{}))!=6 or any(sha(f)!=h for f,h in pre['source_sha256'].items()):raise ValueError('audit prelaunch proof')
 if ind.get('reviewer_sha256')!=sha(R/'review_train_audit_full.py') or doc.get('reviewer_sha256')!=sha(R/'document_overlap_review001/independent_recalc.py'):raise ValueError('upstream reviewer changed')
def gate():
 plan=HERE/'plan_001.json'
 if sha(plan)!=PLAN_SHA:raise ValueError('plan changed')
 p=read(plan)
 initial=read(ROOT/'initial_plan.json')
 if time.time()+600>initial['budget_end_unix']:raise ValueError('insufficient budget')
 if any(sha(f)!=h for f,h in p['evidence_sha256'].items()):raise ValueError('accepted proof changed')
 checks=[('e1_independent_review/monitor_train_audit_prelaunch','e1_independent_review/review_train_audit_prelaunch.py',['--out','e1_independent_review/train_audit_prelaunch_review.json']),('m3_train_data/monitor_train_audit001','m3_train_data/train_audit.py',['001']),('e1_independent_review/monitor_train_audit001','e1_independent_review/review_train_audit_full.py',['--out','e1_independent_review/train_audit001_review.json']),('m3_train_data/monitor_document_overlap002','m3_train_data/document_overlap_analysis_v2.py',[]),('e1_independent_review/document_overlap_review001/monitor_recalc001','e1_independent_review/document_overlap_review001/independent_recalc.py',['--out','e1_independent_review/document_overlap002_review.json']),('m3_train_docsplit/monitor_cpu003','m3_train_docsplit/cpu_validation.py',['--out','m3_train_docsplit/cpu_check_003']),('e1_independent_review/monitor_m3_train_docsplit_prelaunch','e1_independent_review/review_m3_train_docsplit.py',['--out','m3_train_docsplit/independent_review.json'])]
 for m,s,a in checks:supervisor(ROOT/m,ROOT/s,a)
 audit=read(ROOT/'m3_train_data/train_audit_001/result.json');ind=read(R/'train_audit001_review.json');doc=read(R/'document_overlap002_review.json');cpu=read(HERE/'cpu_check_003/verification.json');review=read(HERE/'independent_review.json');pre=read(R/'train_audit_prelaunch_review.json')
 check_reviewers(pre,ind,doc)
 if not (audit['status']=='provisional_completed' and audit['counts']['rows']==39463 and ind['status']=='passed' and ind['independent_review'] is True and ind['formal_result_sha256']==sha(ROOT/'m3_train_data/train_audit_001/result.json') and ind['rows_verified']==39463 and ind['outputs_sha256']==audit['outputs'] and doc['status']=='passed' and doc['independent_review'] is True and doc['GT_opened'] is False and cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after']==review['source_sha256'] and all(sha(f)==h for f,h in review['source_sha256'].items()) and review['status']=='passed' and review['independent_review'] is True and review['reviewer_sha256']==sha(R/'review_m3_train_docsplit.py') and review['cpu_sha256']==sha(HERE/'cpu_check_003/verification.json') and review['cpu_started_sha256']==sha(HERE/'monitor_cpu003/started.json') and review['cpu_terminal_sha256']==sha(HERE/'monitor_cpu003/terminal.json') and review['GT_opened'] is False and review['predictions_opened'] is False):raise ValueError('upstream or code review')
 source=ROOT/'m3_train_data/train_audit_001/rows.jsonl';manifest=ROOT/'data_manifest/full_manifest.jsonl'
 if sha(source)!=p['source_rows_sha256'] or sha(source)!=audit['outputs']['rows.jsonl'] or sha(manifest)!=p['eval_full_manifest_sha256'] or sha(manifest)!=doc['inputs']['eval_manifest_sha256']:raise ValueError('metadata input hash')
 return p,source,manifest
def main():
 if sys.argv!=[str(HERE/'run.py')]:raise ValueError('absolute worker invocation required')
 for _ in range(100):
  try:started=read(HERE/'monitor_run001/started.json');break
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
 else:raise ValueError('run supervisor not recorded')
 if started['worker_pid']!=os.getpid() or started['process_group']!=os.getpid() or started['cwd']!=str(ROOT) or started['timeout_seconds']!=600 or started['command']!=['/home/dell/miniconda3/bin/python',str(HERE/'run.py')]:raise ValueError('run supervision')
 p,source,manifest=gate();evaldocs=set();n=0
 for r in rows(manifest):
  if r['task']=='DocVQA_VAL':
   n+=1;m=re.fullmatch(r'(.+)_p(\d+)',Path(r['source_image_path']).stem)
   if not m:raise ValueError('invalid evaluation image name')
   evaldocs.add(m[1])
 if n!=p['docvqa_eval_rows']:raise ValueError('eval denominator')
 result,summary=core.compute(rows(source),evaldocs)
 if len(result)!=p['row_count']:raise ValueError('row denominator')
 out=HERE/'run_001';out.mkdir(exist_ok=False)
 with (out/'assignments.jsonl').open('x') as f:
  for x in result:f.write(json.dumps(x,sort_keys=True)+'\n')
  f.flush();os.fsync(f.fileno())
 proof={'status':'completed','stage':p['stage'],'summary':summary,'plan_sha256':sha(HERE/'plan_001.json'),'source_rows_sha256':sha(source),'eval_manifest_sha256':sha(manifest),'assignments_sha256':sha(out/'assignments.jsonl'),'review_sha256':sha(HERE/'independent_review.json'),'source_sha256':sha(__file__),'scope':p['scope'],'limitations':p['limitations'],'labels_opened':False,'predictions_opened':False,'gpu_used':False,'training_performed':False}
 with (out/'result.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
