"""Post-acceptance report; no GT, no GPU, no timing reads before gate."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent

def load_gate():
 p=D/'gate.py'
 if hashlib.sha256(p.read_bytes()).hexdigest()!='5926b536b2fdb1f30a42a7d1d2d299f8c55a330ff242ca9c8a3f643b035eab97' or '_paired_timing_analysis_gate' in sys.modules:raise ValueError('gate source/preload')
 spec=importlib.util.spec_from_file_location('_paired_timing_analysis_gate',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[spec.name];raise
 return m

def read(path):return json.loads(Path(path).read_text())
def lines(path):
 with Path(path).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]

def execute(plan,gate,read_json=read,read_lines=lines):
 receipt=gate.admit(plan)  # no journal, observer or health file opened before complete dual acceptance
 gate.verify_files(receipt)
 import numpy as np
 if np.__version__!=plan['numpy_version']:raise ValueError('frozen NumPy version')
 for name in ('metrics','windows','normalize','report_core'):
  if name in sys.modules:raise ValueError('preloaded analysis module '+name)
 sys.path.insert(0,str(D))
 import report_core
 for name in ('metrics','windows','normalize','report_core'):
  if Path(sys.modules[name].__file__).resolve()!=(D/(name+'.py')).resolve():raise ValueError('analysis module origin')
 schedule=read_json(plan['schedule_path']);journals=[read_lines(p) for p in plan['journal_paths']];clocks=[read_json(p) for p in plan['worker_completion_paths']]
 observer=read_lines(plan['observer_path']);run_health=read_lines(plan['run_health_path']);observer_health=read_lines(plan['observer_health_path']);automatic=read_lines(plan['automatic_health_windows_snapshot'])
 observer_impls={x.get('monotonic_clock') for x in observer}
 observer_clock=next(iter(observer_impls)) if len(observer_impls)==1 and isinstance(next(iter(observer_impls)),str) else 'unverified observer clock'
 result=report_core.analyze(schedule,journals,clocks,observer,observer_clock,run_health,observer_health,automatic,
                            plan.get('run_supervisor_monotonic_clock'),plan.get('observer_supervisor_monotonic_clock'),plan.get('automatic_health_monotonic_clock'))
 gate.verify_files(receipt)
 return {'status':'passed','scope':'posthoc same-DEV paired timing; not equal compute','GT_opened':False,'scoring_performed':False,
         'analysis_plan_sha256':gate.sha(D/'analysis_plan001.json'),'statistics_plan_sha256':gate.sha(D/'statistics_plan001.json'),
         'accepted_formal_sha256':receipt['formal_sha256'],'accepted_independent_sha256':receipt['independent_sha256'],
         'accepted_files_sha256':receipt['files_sha256'],'result':result,'errors':[]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=(D/'analysis001.json').resolve() or a.out.exists():raise ValueError('fixed new output required')
 if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU environment')
 gate=load_gate();plan=read(D/'analysis_plan001.json');result=execute(plan,gate)
 with a.out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
