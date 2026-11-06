"""Actual single-job bridge, without a GPU-authorizing entry point.
Admission must be implemented and independently reviewed before use.
"""
import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent; B=D.parents[1]; C=B/'e6/pixel_wavefront_t020_model_timing_v2';J=B/'e6/pixel_wavefront_t020_timing_controller_draft'
def load(name,path):
 p=Path(path).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign cached module '+name)
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def prepare_cpu():
 # These imports are exercised before a future real no-GPU preflight exits.
 worker=load('_pixel_t020_runtime_worker',D/'worker.py');controller=load('_pixel_t020_runtime_controller',D/'controller.py')
 host=load('_pixel_t020_runtime_host',D/'host_runner.py');stages=load('_pixel_t020_runtime_stages',D/'nvml_stages.py')
 check=load('_pixel_t020_runtime_job_contract',J/'job_contract.py');model=load('_pixel_t020_runtime_model',C/'model_candidate.py')
 return {'worker':worker,'controller':controller,'host_runner':host,'nvml_stages':stages,'check':check,'model':model}

def unsafe_run(job,jobdir,prepared):
 """Unsafe: requires checked GPU gate, supervisor, container release first."""
 model=prepared['model'];runtime=model.prepare_cpu();worker=prepared['worker']
 # input_preflight is imported by the previous accepted model module through
 # the reviewed source chain, and its actual source path must be verified by
 # the final entry before this continuation can ever be called.
 import input_preflight
 if not getattr(input_preflight,'__file__',None) or Path(input_preflight.__file__).resolve()!=(B/'e4_c_latency/input_preflight.py').resolve():raise ValueError('foreign input_preflight')
 rows,e1=input_preflight.load_accepted_e1_references(B/'e1/e1full001_completion.json',B/'e1/monitorverifye1full001/started.json',B/'e1/monitorverifye1full001/terminal.json')
 selected=[r for r in rows if (r['task'],str(r['qid']))==(job['task'],str(job['qid']))]
 if len(selected)!=1:raise ValueError('missing or duplicate input row')
 row=selected[0];ref=e1[(row['task'],str(row['qid']))]
 if any(str(row.get(k))!=str(job[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('accepted row identity')
 row['image_start']=ref['image_start']
 return worker.run_one(job,jobdir,runtime,row,ref,model.unsafe_continuation,prepared['check'].check)

if __name__=='__main__':raise SystemExit('GPU locked; unsafe continuation has no executable launcher')
