"""Independent synthetic-only review of the future M3 DEV scoring chain."""
import argparse,hashlib,json,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent;HERE=BASE/'m3_oracle_analysis'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==HERE/'independent_review.json'
 cpu=HERE/'cpu_check_006/verification.json';start=HERE/'monitor_cpu006/started.json';end=HERE/'monitor_cpu006/terminal.json';s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and t['worker_pid']==s['worker_pid']==s['process_group']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_validation.py']==[(HERE/'cpu_validation.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==cpu.parent
 c=read(cpu);assert c['status']=='passed' and c['GT_opened'] is False and c['predictions_opened'] is False and c['scoring_performed'] is False and c['gpu_used'] is False
 sources=c['source_sha256_after'];assert sources==c['source_sha256_before'] and all(sha(p)==h for p,h in sources.items())
 assert all(sources.get(str(p.resolve()))==sha(p) for p in list(HERE.glob('*.py'))+[HERE/'plan.json'])
 tests=c['tests'];assert len(tests)==5 and len({x['test'] for x in tests})==5
 for test in tests:
  assert test['returncode']==0 and test['timeout'] is False
  for stream in ('stdout','stderr'):assert sha(cpu.parent/(test['test']+'.'+stream))==test[stream+'_sha256']
 plan=read(HERE/'plan.json');assert plan['questions']==128 and plan['new_generations']==768 and plan['bootstrap']['unit']=='image_group'
 assert plan['oracle_protocol_sha256']==sha(BASE/'m3_oracle/protocol_001.json') and plan['oracle_amendment_sha256']==sha(BASE/'m3_oracle_exec/protocol_amendment_001.json') and plan['sample_sha256']==sha(BASE/'m3_oracle/sample_001.json')
 sample=read(BASE/'m3_oracle/sample_001.json');rows=sample['examples'];assert len(rows)==128 and len({x['group_id'] for x in rows})==128
 result={'status':'passed','independent_review':True,'GT_opened':False,'predictions_opened':False,'scoring_performed':False,'gpu_used':False,
         'reviewer_sha256':sha(__file__),'cpu_proof_sha256':sha(cpu),'cpu_started_sha256':sha(start),'cpu_terminal_sha256':sha(end),'source_sha256':sources,
         'limits':['Synthetic-only scoring source review; complete M3 panel and independent acceptance are required before any prediction or reference is read.',
                   'Seven-arm answer-dependent oracle is exploratory and not a deployable comparison.',
                   'The 128-image sample is not a representative population and bootstrap intervals are unadjusted across comparisons.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','tests':len(tests),'sources':len(sources)}))
if __name__=='__main__':main()
