"""Freeze independent CPU-only statistical and window-contract review evidence."""
import hashlib,json,os
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;M=B/'e4_c_multibudget_analysis192_v2';W=B/'e4_c_multibudget_analysis192_independent';X=B/'e4_c_multibudget_expansion192'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(d,module):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid') or cmd.count('-m')!=1 or cmd[cmd.index('-m')+1]!='unittest' or module not in cmd:raise ValueError('independent CPU test supervisor')
 log=(d/'stderr.log').read_text()
 if 'Ran 3 tests' not in log or '\nOK\n' not in log:raise ValueError('independent test count')
 return {'started_sha256':sha(d/'started.json'),'terminal_sha256':sha(d/'terminal.json'),'test_log_sha256':sha(d/'stderr.log')}
def main():
 out=I/'e4c_analysis192_v2_independent_cpu_review.json'
 if out.exists():raise ValueError('output exists')
 paths=[M/'metrics.py',M/'test_metrics.py',I/'test_e4c_analysis192_v2_independent.py',W/'window_detail.py',W/'test_window_detail.py',W/'GATE_AND_WINDOWS_DESIGN.md',X/'ANALYSIS_PLAN.md',X/'analysis_plan_freeze001.json',Path(__file__).resolve()];before={str(p):sha(p) for p in paths}
 stats=check(I/'monitor_e4c_analysis192_v2_independent_cpu001','test_e4c_analysis192_v2_independent');windows=check(W/'monitor_cpu001','test_window_detail');f=read(X/'analysis_plan_freeze001.json')
 if f.get('analysis_plan_sha256')!=sha(X/'ANALYSIS_PLAN.md') or f.get('GT_opened') is not False or f.get('expansion_predictions_opened') is not False or f.get('GPU_authorized') is not False or before!={str(p):sha(p) for p in paths}:raise ValueError('analysis plan/source changed')
 proof={'status':'passed','independent_review':True,'scope':'192 image exploratory pure statistics and bounded-window CPU design only','source_sha256':before,'statistics_cpu':stats,'windows_cpu':windows,'GT_opened':False,'target_predictions_opened':False,'GPU_authorized':False,'errors':[],'limits':['Only synthetic tests; no full runtime, formal proof, independently verified result or actual environment observer exists.','Future independently audited source gates and completed no-GT formal and independent acceptance mandatory before target journal reading.','All 7920 samples retained; unobserved activity and logger/manual fsync remain unknown.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
