"""Independent batch-2 preflight-only review; never reads answer/prediction files."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'m3_oracle_batch2';HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def req(condition,message):
 if not condition:raise ValueError(message)
def supervisor(start,terminal,script,out):
 s,t=read(start),read(terminal);cmd=s['command'];cwd=Path(s['cwd'])
 req(t.get('reason')=='completed' and t.get('worker_returncode')==0 and t.get('remaining_process_group')==[] and
     type(s.get('worker_pid')) is int and s['worker_pid']>0 and s['worker_pid']==t.get('worker_pid') and
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and
     cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve(),'supervisor or command mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 req(out==(B/'independent_preflight_review.json').resolve(),'unexpected proof path')
 cpu=B/'cpu_check_003/verification.json';cs=B/'monitor_cpu003/started.json';ct=B/'monitor_cpu003/terminal.json'
 supervisor(cs,ct,B/'cpu_validation.py',cpu.parent);v=read(cpu);src=v['source_sha256_after']
 req(v.get('status')=='passed' and v.get('returncode')==0 and v.get('timed_out') is False and v.get('source_sha256_before')==src and
     v.get('GT_opened') is False and v.get('predictions_opened') is False and v.get('real_model_loaded') is False and v.get('gpu_used') is False and len(src)==147,
     'CPU proof invalid')
 req(all(sha(path)==digest for path,digest in src.items()),'source changed after CPU proof')
 req(all(src.get(str(path.resolve()))==sha(path) for path in list(B.glob('*.py'))+[B/'execution_plan_002.json']),
     'batch-2 source inventory missing')
 independently=HERE/'monitor_m3batch2_indcpu001';ist=read(independently/'started.json');ite=read(independently/'terminal.json')
 req(ite.get('reason')=='completed' and ite.get('worker_returncode')==0 and ite.get('remaining_process_group')==[] and
     ist.get('worker_pid')==ite.get('worker_pid') and 'unittest' in ist.get('command',[]),'independent CPU tests failed')
 req('Ran 6 tests' in (independently/'stderr.log').read_text() and (independently/'stderr.log').read_text().rstrip().endswith('OK'),
     'independent CPU test inventory')
 planpath=B/'execution_plan_002.json';plan=read(planpath);sample=read(B/'sample_002.json');first=read(ROOT/'m3_oracle/sample_001.json')
 review=B/'sample_002_independent_review.json';sr=read(review);ss=B/'monitor_review002/started.json';st=B/'monitor_review002/terminal.json'
 # The selection reviewer has a fixed output path rather than an --out argument.
 started,terminal=read(ss),read(st);cmd=started['command'];cwd=Path(started['cwd'])
 req(terminal.get('reason')=='completed' and terminal.get('worker_returncode')==0 and terminal.get('remaining_process_group')==[] and
     terminal.get('worker_pid')==started.get('worker_pid') and [(cwd/x).resolve() for x in cmd if Path(x).name=='review_sample.py']==[(B/'review_sample.py').resolve()],
     'sample reviewer did not exit cleanly')
 req(sr.get('status')=='passed' and sr.get('independent_review') is True and sr.get('questions')==128 and sr.get('overlap_first_batch')==0 and
     sr.get('GT_opened') is False and sr.get('predictions_opened') is False and sr.get('reviewer_sha256')==sha(B/'review_sample.py') and
     sr.get('sample_sha256')==sha(B/'sample_002.json'),'sample review mismatch')
 req(sample.get('selection_after_first_batch_scores') is True and sample.get('selection_uses_scores') is False and
     sample.get('GT_opened') is False and sample.get('predictions_opened') is False and
     collections.Counter(x['task'] for x in sample['examples'])=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32} and
     len({x['group_id'] for x in sample['examples']})==128 and
     not ({x['group_id'] for x in sample['examples']}&{x['group_id'] for x in first['examples']}),'second sample inventory or overlap')
 req(all(set(x)=={'task','qid','group_id','image_path','image_sha256','message_sha256'} for x in sample['examples']),
     'unexpected sample fields')
 req(all(sha(x['image_path'])==x['image_sha256'] for x in sample['examples']),'sample image changed')
 req(all(sha(path)==digest for path,digest in plan['source_sha256'].items()) and
     plan['sample_sha256']==sha(B/'sample_002.json') and plan['sample_review_sha256']==sha(review) and
     (plan['questions'],plan['new_generations_per_question'],plan['maximum_new_generations'],plan['maximum_candidate_evaluations_per_question'])==(128,6,768,49) and
     (plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['snapshot_seconds'],plan['progress_stale_seconds'],plan['question_checkpoint_seconds'])==(7200,7380,3600,2400,1800) and
     plan['budget_end_unix']==1794065414 and time.time()+plan['outer_timeout_seconds']<=plan['budget_end_unix'] and
     plan['GT_opened'] is False and plan['predictions_opened'] is False and plan['gpu_authorized'] is False,'plan or deadline mismatch')
 req(not (B/'runtime_approval.json').exists(),'batch-2 approval was already set during review')
 out_data={'status':'passed','independent_review':True,'preflight_only':True,'GT_opened':False,'predictions_opened':False,'gpu_used':False,
           'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(cs),'cpu_terminal_sha256':sha(ct),
           'independent_cpu_started_sha256':sha(independently/'started.json'),'independent_cpu_terminal_sha256':sha(independently/'terminal.json'),
           'reviewer_sha256':sha(__file__),'plan_sha256':sha(planpath),'source_sha256':src,'sample_sha256':sha(B/'sample_002.json'),
           'sample_review_sha256':sha(review),'questions':128,'overlap_first_batch':0,
           'scope':'only allows actual no-GPU/no-model/no-prediction/no-new-GT worker preflight; no GPU execution approval'}
 with out.open('x') as f:json.dump(out_data,f,indent=2);f.write('\n');f.flush();__import__('os').fsync(f.fileno())
 return 0
if __name__=='__main__':sys.exit(main())
