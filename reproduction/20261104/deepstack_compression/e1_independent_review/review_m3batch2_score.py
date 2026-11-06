"""Independent source/CPU review of second-batch scorer, no prediction or GT."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'m3_oracle_batch2_analysis';R=ROOT/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sup(dir,script,out):
 s=read(dir/'started.json');t=read(dir/'terminal.json');c=s['command'];wd=Path(s['cwd'])
 if not (t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and wd==ROOT and len(c)==4 and (wd/c[1]).resolve()==script.resolve() and c[2]=='--out' and (wd/c[3]).resolve()==out.resolve()):raise ValueError('CPU supervisor')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(A/'independent_prelaunch_review.json').resolve():raise ValueError('wrong output')
 cpu=A/'cpu_check_007/verification.json';mon=A/'monitor_cpu007';ind=R/'m3batch2_score_cpu007/verification.json';im=R/'monitor_m3batch2_score_cpu007'
 sup(mon,A/'cpu_validation.py',cpu.parent);sup(im,A/'cpu_validation.py',ind.parent);c=read(cpu);i=read(ind);src=c['source_sha256_after']
 if not (c['status']==i['status']=='passed' and c['source_sha256_before']==i['source_sha256_before']==i['source_sha256_after']==src and len(src)==41 and all(sha(p)==v for p,v in src.items()) and all(x[k] is False for x in (c,i) for k in ('GT_opened','predictions_opened','gpu_used'))):raise ValueError('CPU source or scope')
 for f in (cpu.parent/'stderr.log',ind.parent/'stderr.log'):
  log=f.read_text()
  if log.count(' ... ok')!=9 or log.count('\nOK\n')!=5 or not log.rstrip().endswith('OK'):raise ValueError('incomplete synthetic tests')
 if c['stderr_sha256']!=sha(cpu.parent/'stderr.log') or i['stderr_sha256']!=sha(ind.parent/'stderr.log'):raise ValueError('CPU test log changed')
 plan=read(A/'plan.json');score=(A/'score_oracle.py').read_text()
 if not (plan['questions']==128 and plan['new_generations']==768 and plan['bootstrap']=={'replicates':10000,'seed':20261104,'unit':'image_group','interval':'percentile 95%'} and plan['minimum_relevant_gain_pp']==1.0 and plan['GT_opened'] is False and plan['predictions_opened'] is False and plan['scoring_performed'] is False and sha(A/'plan.json') in score and sha(ROOT/'m3_oracle_analysis/plan.json')==plan['first_batch_plan_sha256'] and sha(A/'plan_001_before_timing_clarification.json')==plan['prior_plan_sha256']):raise ValueError('analysis plan mismatch')
 proof={'status':'passed','independent_review':True,'reviewer_sha256':sha(__file__),'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(mon/'started.json'),'cpu_terminal_sha256':sha(mon/'terminal.json'),'source_sha256':src,'independent_cpu_sha256':sha(ind),'independent_cpu_started_sha256':sha(im/'started.json'),'independent_cpu_terminal_sha256':sha(im/'terminal.json'),'GT_opened':False,'predictions_opened':False,'scope':'source and synthetic scoring review only; real DEV scoring requires full formal and independent panel acceptance and real exit'}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
