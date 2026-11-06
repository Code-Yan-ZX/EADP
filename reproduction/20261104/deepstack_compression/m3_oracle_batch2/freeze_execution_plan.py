"""Freeze CPU-only resource and protocol contract; grants no execution."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 sample=HERE/'sample_002.json';review=HERE/'sample_002_independent_review.json';started=HERE/'monitor_review002/started.json';terminal=HERE/'monitor_review002/terminal.json';p=read(sample);r=read(review);s=read(started);t=read(terminal)
 assert p['counts']=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32} and p['selection_after_first_batch_scores'] is True and p['GT_opened'] is False and p['predictions_opened'] is False
 assert r['status']=='passed' and r['independent_review'] is True and r['sample_sha256']==sha(sample) and r['questions']==128 and r['overlap_first_batch']==0 and r['GT_opened'] is False and r['predictions_opened'] is False
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']
 old=ROOT/'m3_oracle_exec/runtime_plan.json';first=read(old);firstproof=ROOT/'m3_oracle_verify_v2/runtime_001_verification_author.json';f=read(firstproof);runterminal=ROOT/'m3_oracle_exec/monitor_runtime001/terminal.json';tr=read(runterminal);budget=ROOT/'initial_plan.json';initial=read(budget)
 assert f['status']=='passed' and f['n_questions']==128 and f['new_generations']==768 and tr['reason']=='completed' and tr['worker_returncode']==0 and tr['remaining_process_group']==[]
 assert first['budget_end_unix']==initial['budget_end_unix']==1794065414
 deps=[sample,review,started,terminal,old,firstproof,runterminal,budget,ROOT/'m3_oracle/protocol_001.json',ROOT/'m3_oracle/proposals.py',ROOT/'m3_oracle/teacher_contract.py',ROOT/'m3_oracle/oracle_teacher.py',ROOT/'m3_oracle/saliency.py',ROOT/'m3_oracle/target_contract.py',ROOT/'m3_oracle_exec/runtime_001_sources/manifest.json',ROOT/'m3_oracle_exec/panel.py',ROOT/'m3_oracle_exec/one_question.py',ROOT/'m3_oracle_exec/search.py',ROOT/'m3_oracle_exec/teacher_backend.py',ROOT/'m3_oracle_exec/generate_backend.py',ROOT/'m3_oracle_exec/baseline.py',Path(__file__)]
 plan={'stage':'prospective_locked','batch':2,'created_unix':time.time(),'sample_path':str(sample),'sample_sha256':sha(sample),'sample_review_sha256':sha(review),
       'questions':128,'questions_by_task':p['counts'],'new_generations_per_question':6,'maximum_new_generations':768,'maximum_candidate_evaluations_per_question':49,'maximum_candidate_evaluations':6272,
       'gradient_passes':128,'reference_policy':'first reference only, same frozen teacher and tokenization','candidate_policy':'unchanged r8/r16 MAIN/MULTI/RANDOM equal budgets and Python 3.10 RNG',
       'baseline_policy':'reuse E1 P3-D1 only after per-question input, selection, DS, P3 and actual first-logits identity',
       'first_batch_observed':{'supervisor_elapsed_seconds':tr['elapsed_seconds'],'candidate_evaluations':sum(x['evaluations'] for x in f['questions']),'new_generations':768,'note':'one run; not a second-batch runtime guarantee'},
       'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix'],'inner_timeout_seconds':7200,'outer_timeout_seconds':7380,'question_checkpoint_seconds':1800,'progress_stale_seconds':2400,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':120,'minimum_free_bytes':32212254720,'gpu_name':'NVIDIA A40',
       'source_sha256':{str(x):sha(x) for x in deps},'GT_opened':False,'predictions_opened':False,'gpu_authorized':False,
       'limitations':['Second exploratory look after first results; report batches separately.','Requires new code review, real no-GPU worker preflight, GPU scheduling and full acceptance.','No CONFIRM/FINAL or student confirmation use.']}
 out=HERE/'execution_plan_002.json'
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'status':'plan_frozen_no_execution','questions':128,'sha256':sha(out)}))
if __name__=='__main__':main()
