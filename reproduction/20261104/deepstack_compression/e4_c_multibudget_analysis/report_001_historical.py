"""Run fixed, paired pilot analysis only after both complete acceptances."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
EXPECTED_GATE_SHA='0101b92eaf446f713442f5c16e44ffa0802c801bab701ee1e3f43860d56646d6'
def load(name,path):
 if name in sys.modules:raise ValueError('preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 if a.out.resolve()!=D/'analysis001.json' or a.out.exists():raise ValueError('unexpected output')
 gate_path=D/'analysis_gate.py'
 if hashlib.sha256(gate_path.read_bytes()).hexdigest()!=EXPECTED_GATE_SHA:raise ValueError('analysis gate changed')
 gate=load('_multibudget_analysis_gate',gate_path);admission=gate.admit()
 # No prediction-bearing journal is parsed before the complete admissions above.
 plan=gate.read(B/'e4_c_multibudget_expansion/pilot_plan001.json');formal=gate.read(B/'e4_c_multibudget_worker_accept_v3/verification001.json');journal=B/'e4_c_multibudget_worker_v3/run_001/journal.jsonl'
 with journal.open() as f:events=[json.loads(x) for x in f if x.strip()]
 metrics=load('_multibudget_analysis_metrics',D/'metrics.py')
 if metrics.np.__version__!='2.2.6':raise ValueError('NumPy version differs from frozen bootstrap')
 result=metrics.summarize(plan['jobs'],events);annotations=formal['annotations'];windows=formal['health_windows']
 if len(annotations)!=1520:raise ValueError('missing health annotations')
 health={}
 for i,(job,ann) in enumerate(zip(plan['jobs'],annotations)):
  if ann.get('order')!=i or any(ann.get(k)!=job.get(k) for k in ('phase','task','budget','protocol')):raise ValueError('health annotation/job mismatch')
  for key in ('ttft_health_overlap_indices','full_job_health_overlap_indices'):
   v=ann.get(key)
   if not isinstance(v,list) or len(set(v))!=len(v) or any(type(x) is not int or not 0<=x<len(windows) for x in v):raise ValueError('health annotation indices')
  if job['phase']=='measure':
   key='/'.join((job['task'],job['protocol'],str(job['budget'])));h=health.setdefault(key,{'measured_jobs':0,'ttft_direct_overlap':0,'generation_direct_overlap':0});h['measured_jobs']+=1;h['ttft_direct_overlap']+=bool(ann['ttft_health_overlap_indices']);h['generation_direct_overlap']+=bool(ann['full_job_health_overlap_indices'])
 result.update(admission=admission,analysis_plan_sha256=gate.sha(B/'e4_c_multibudget_expansion/ANALYSIS_PLAN.md'),pilot_plan_sha256=gate.sha(B/'e4_c_multibudget_expansion/pilot_plan001.json'),predictions_opened=True,prediction_scope='accepted completed timing journal parsed for timings and output terminal counts',health_overlap=health,health_limit='Only logged direct health syscall windows; background activity and unbounded ps subprocesses cannot be localized, all measurements retained',author_generation_completion_limit='synchronized generate return before raw token copy/fsync and author text decode; not full end-to-end latency')
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('32-image exploratory timing analysis completed')
if __name__=='__main__':main()
