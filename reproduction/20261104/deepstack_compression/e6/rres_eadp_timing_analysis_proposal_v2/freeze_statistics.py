"""Freeze pure statistics/synthetic evidence only; no timing records or GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1];END=1794065414

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 if any(os.environ.get(x)!='1' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('single-thread CPU environment')
 if (D/'statistics_plan001.json').exists():raise ValueError('do not overwrite')
 paths=[D/'PLAN.md',D/'metrics.py',D/'windows.py',D/'test_metrics.py',D/'test_windows.py',Path(__file__),B/'e6/rres_eadp_timing_pilot_v4/schedule001.json',B/'e6/rres_eadp_timing_pilot_v4/schedule.py',B/'initial_plan.json']
 before={str(p):sha(p) for p in paths}
 commands=[]
 for pattern,number in [('test_metrics.py',6),('test_windows.py',6)]:
  result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(D),'-p',pattern,'-v'],capture_output=True,text=True,timeout=60)
  if result.returncode or f'Ran {number} tests' not in result.stderr or '\nOK\n' not in result.stderr:raise ValueError('test failure '+result.stderr[-3000:])
  commands.append({'pattern':pattern,'tests':number,'result':'passed'})
 if before!={str(p):sha(p) for p in paths}:raise ValueError('source changed')
 schedule=json.loads((B/'e6/rres_eadp_timing_pilot_v4/schedule001.json').read_text());budget=json.loads((B/'initial_plan.json').read_text())
 if budget['budget_end_unix']!=END or schedule['status']!='frozen' or len(schedule['measured_inputs'])!=32 or len(schedule['warmup_inputs'])!=30 or len(schedule['processes'])!=12 or schedule['warmup_requests']!=60 or schedule['measured_requests']!=320:raise ValueError('schedule/budget')
 p={'status':'frozen','scope':'statistics and normalized-window CPU candidate; not GPU or result-read admission','images':32,'processes':12,'warmups':60,'measured':320,'bootstrap_draws':10000,'seed':20261106,'bootstrap_rng':'numpy.random.default_rng','percentiles':[2.5,97.5],'percentile_method':'linear','numpy_version':np.__version__,'python_version':sys.version,'budget_end_unix':END,'GT_opened':False,'timing_results_opened':False,'GPU_used':False,'source_sha256':before,'tests':commands,'limitations':['Result/proof/observer schemas and executable read gate are pending integration and separate freeze/review.', 'Unknown interference windows remain unknown and all samples are retained.'],'errors':[]}
 with (D/'statistics_plan001.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
