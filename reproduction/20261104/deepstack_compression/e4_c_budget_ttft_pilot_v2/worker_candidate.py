"""Single-model 48-job continuation; no executable GPU entry or authority."""
import json,os,time
from pathlib import Path
def append(path,item):
 with Path(path).open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def execute_schedule(rows,records,jobs,contract,runner,torch,wrapper,native,adapter,measure,criteria,author_generate,journal):
 contract.validate(jobs,rows)
 if time.get_clock_info('perf_counter').implementation!='clock_gettime(CLOCK_MONOTONIC)':raise ValueError('timing/health clock mismatch')
 with Path(journal).open('x') as f:f.flush();os.fsync(f.fileno())
 lookup={(r['task'],str(r['qid'])):r for r in rows}
 for job in jobs:
  key=(job['task'],job['qid']);row=lookup[key];ref=records[(*key,job['budget'])]
  def emit(kind,payload=None):append(journal,{'event':kind,'job':job,'payload':payload})
  emit('started')
  try:
   result=runner.execute(torch,wrapper,native,row,job['budget'],ref,adapter,measure,criteria,author_generate,lambda raw:emit('raw_generation',raw))
   emit('completed',result)
  except BaseException as exc:
   emit('failed',{'type':type(exc).__name__,'message':str(exc)});raise
 return {'jobs':len(jobs),'warmups':12,'measured':36,'stage':'exploratory 3-anchor natural TTFT','GT_opened':False,'scoring_performed':False}
if __name__=='__main__':raise SystemExit('GPU locked: reviewed worker, container and identity admission missing')
