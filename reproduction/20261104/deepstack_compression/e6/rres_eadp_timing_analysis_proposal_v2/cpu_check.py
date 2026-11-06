"""Pure synthetic complete analysis-chain proof; never reads real timings."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(d,script,out=None):
 a=read(d/'started.json');b=read(d/'terminal.json');cmd=a['command'];cwd=Path(a['cwd'])
 if (a.get('worker_pid')!=b.get('worker_pid') or b.get('reason')!='completed' or b.get('worker_returncode')!=0 or b.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('prior supervisor')
def main():
 if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU environment')
 if (D/'analysis_cpu_proof001.json').exists():raise ValueError('existing proof')
 stat=D/'statistics_plan001.json';freeze=D/'monitor_statistics_freeze001';proof=I/'rres_eadp_timing_statistics_v2_prelaunch_review.json';review=I/'review_rres_eadp_timing_statistics_v2.py';monitor=I/'monitor_rres_eadp_timing_statistics_v2'
 supplement=I/'rres_eadp_timing_statistics_v2_supplement.json';suppreview=I/'review_rres_eadp_timing_statistics_v2_supplement.py';suppmonitor=I/'monitor_rres_eadp_timing_statistics_v2_supplement'
 term(freeze,D/'freeze_statistics.py');term(monitor,review,proof);term(suppmonitor,suppreview)
 p=read(stat);q=read(proof);u=read(supplement)
 if (p.get('status')!='frozen' or p.get('numpy_version')!=np.__version__ or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('plan_sha256')!=sha(stat) or q.get('reviewer_sha256')!=sha(review) or q.get('errors')!=[] or q.get('GT_opened') is not False or q.get('timing_results_opened') is not False or u.get('status')!='passed' or u.get('reviewer_sha256')!=sha(suppreview) or u.get('errors')!=[] or u.get('GT_opened') is not False or u.get('timing_results_opened') is not False or any(sha(k)!=v for k,v in p['source_sha256'].items()) or any(sha(k)!=v for k,v in u['source_sha256'].items())):raise ValueError('statistics proof/source')
 files=[D/n for n in ('PLAN.md','statistics_plan001.json','gate.py','metrics.py','windows.py','normalize.py','report_core.py','report.py','test_gate.py','test_metrics.py','test_windows.py','test_normalize.py','test_report_core.py','test_report_gate.py','cpu_check.py')]+[proof,review,supplement,suppreview,freeze/'started.json',freeze/'terminal.json',monitor/'started.json',monitor/'terminal.json',suppmonitor/'started.json',suppmonitor/'terminal.json']
 hashes={str(x):sha(x) for x in files}
 tests=[('test_gate.py',3),('test_metrics.py',6),('test_windows.py',6),('test_normalize.py',3),('test_report_core.py',2),('test_report_gate.py',2)]
 for pattern,count in tests:
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p',pattern,'-v'],capture_output=True,text=True,timeout=60)
  if r.returncode or f'Ran {count} tests' not in r.stderr or '\nOK\n' not in r.stderr:raise ValueError('test failed '+pattern+': '+r.stderr[-2000:])
 if hashes!={str(x):sha(x) for x in files}:raise ValueError('source changed during tests')
 data={'status':'passed','scope':'prelaunch CPU-only complete analysis candidate; no actual timing or GT opened','tests':sum(x[1] for x in tests),'GT_opened':False,'timing_results_opened':False,'GPU_used':False,'numpy_version':np.__version__,'statistics_plan_sha256':sha(stat),'statistics_review_sha256':sha(proof),'statistics_supplement_sha256':sha(supplement),'source_sha256':hashes,'errors':[]}
 with (D/'analysis_cpu_proof001.json').open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
