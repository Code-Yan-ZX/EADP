"""CPU-only postprocessing of already captured, sanitized NVML query windows.
Does not start a sampler, a model, Docker, CUDA or nvidia-smi.
"""
import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent; C=D.parent/'pixel_wavefront_t020_model_timing_v2'/'nvml_contract.py'
def _load():
 name='_pixel_t020_nvml_contract_stage';old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=C.resolve():raise ValueError('foreign nvml contract')
  return old
 spec=importlib.util.spec_from_file_location(name,C);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 return mod

def summarize(metadata,target_pid,uuid,bounds,max_gap_ns):
 names=('model_load_start','model_load_end','request_start','request_end','teardown_end')
 if any(type(bounds.get(k)) is not int for k in names):raise ValueError('missing stage boundaries')
 t=[bounds[k] for k in names]
 if not 0<t[0]<t[1]<=t[2]<t[3]<=t[4]:raise ValueError('invalid stage order')
 # Covers model-load through teardown; query windows can contain boundaries.
 checked=_load().validate(metadata,target_pid,uuid,(t[0],t[4]),max_gap_ns)
 result={}
 for name,start,end in [('model_load',t[0],t[1]),('request',t[2],t[3]),('teardown',t[3],t[4])]:
  indices=[i for i,m in enumerate(metadata) if m['query_start_ns']<=end and m['query_end_ns']>=start]
  inside=[i for i in indices if metadata[i]['query_start_ns']>=start and metadata[i]['query_end_ns']<=end]
  stage={}
  for key in ('target_used_bytes','total','used','free','reserved'):
   vals=[checked[i]['target_used_bytes'] if key=='target_used_bytes' else checked[i]['device_memory_bytes'][key] for i in indices]
   known=[v for v in vals if v is not None]
   stage[key]={'known_samples':len(known),'unknown_samples':len(vals)-len(known),'known_sample_max_bytes':max(known) if known else None,'coverage_complete':bool(vals) and len(known)==len(vals)}
  result[name]={'overlap_indices':indices,'fully_contained_indices':inside,'metrics':stage,'target_absent_samples':sum(not checked[i]['target_present'] for i in indices)}
 return {'status':'passed','stage_bounds_monotonic_ns':bounds,'stages':result,'sample_count':len(metadata),'limits':['observed samples only; transient peaks may be missed','MiB-granularity source','NVML target process and device totals are distinct','overlapping query windows do not pinpoint the sample time within the query']}

if __name__=='__main__':raise SystemExit('CPU postprocessing only; no sampler or GPU launcher')
