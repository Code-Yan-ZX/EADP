"""Reviewed host controller for single-model NVML; no standalone approval."""
import argparse,importlib.util,json,os,secrets,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python';AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';sys.path[:0]=[str(D),str(B/'e4_c_memory_real'),str(B/'e4_c_memory_runtime'),str(B/'e4_c_latency')]
_imports=sys.modules.get('imports')
if _imports is not None and (not getattr(_imports,'__file__',None) or Path(_imports.__file__).resolve()!=D/'imports.py'):raise ValueError('foreign imports module')
from imports import checked_import,check_existing,latency_bindings
check_existing(latency_bindings())
gate=checked_import('gate',D/'gate.py');timing_launch=checked_import('timing_launch',B/'e4_c_latency/timing_launch.py');guard=checked_import('guard',B/'e4_c_memory_real/guard.py')
check_existing(latency_bindings())
if Path(gate.__file__).resolve()!=D/'gate.py' or Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py' or Path(guard.__file__).resolve()!=B/'e4_c_memory_real/guard.py':raise ValueError('controller source collision')
summarize=checked_import('summary',D/'summary.py').summarize

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path.resolve():raise ValueError('module source mismatch')
 return m
def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if a.run_id!='001':raise ValueError('unexpected run id')
 approval=gate.review(preflight_only=False,host=True);s=gate.read(D/'monitor_run001/started.json');cmd=s['command'];cwd=Path(s['cwd']);plan=gate.verify_plan()
 if s['worker_pid']!=os.getpid() or s['timeout_seconds']!=1680 or s['grace_seconds']!=90 or s['snapshot_seconds']!=3600 or [(cwd/x).resolve() for x in cmd if Path(x).name=='controller.py']!=[Path(__file__).resolve()] or cmd[-2:]!=['--run-id','001']:raise ValueError('controller supervisor mismatch')
 init=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':init['budget_start_unix'],'budget_end_unix':init['budget_end_unix']};timing_launch.validate_budget_window(budget,time.time(),1560,1680,90)
 check_existing({'diagnose':AP.parent/'diagnose.py',**{n:B/'e4_c_memory_runtime'/(n+'.py') for n in ('controller_adapter','controller_io','docker_guard','interruptible','sampler_supervisor')}})
 runtime=load(AP.parent/'run_panel.py','eadp_modelmemory_container_runtime');ctrl=load(B/'e4_c_memory_real/controller.py','eadp_modelmemory_real_controller');verify=load(B/'e4_c_memory_nvml/verify_metadata.py','eadp_modelmemory_verify_metadata')
 check_existing({'diagnose':AP.parent/'diagnose.py',**{n:B/'e4_c_memory_runtime'/(n+'.py') for n in ('controller_adapter','controller_io','docker_guard','interruptible','sampler_supervisor')}})
 inv,compute=ctrl.query_gpu();uuid=guard.parse_inventory(inv,compute);out=D/'run_001';out.mkdir(exist_ok=False);nonce=secrets.token_hex(16);name='eadp-memory-model-'+nonce;worker=[str(D/'worker.py'),'--out',str(out),'--nonce',nonce];author=gate.read(AP);argv=runtime.container_command(author,worker,True,name);create=guard.create_command(argv,name,nonce)
 sampler=[PY,str(D/'sampler.py'),'--directory',str(out)];sm=out/'sampler_monitor';supervisor=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(sm),'--timeout','1500','--snapshot-seconds','3600','--poll-seconds','1','--grace-seconds','15','--']+sampler
 save(out/'launch.json',{'scope':'single EADP K256 C-ON process memory; no GT or timing','approval':approval,'nonce':nonce,'gpu_uuid':uuid,'docker_create':create,'sampler_worker':sampler,'sampler_supervisor':supervisor,'model_loaded':True,'GT_opened':False})
 result={'status':'failed','failure':None,'scope':'one model process, sampled NVML and allocator maxima'}
 try:
  ops=ctrl.construct(directory=out,mode='C-ON',nonce=nonce,gpu_uuid=uuid,create_argv=create,name=name,image=author['container_image'],worker_argv=worker,sampler_command=supervisor,sampler_worker_command=sampler,sampler_monitor=sm,worker_timeout=1200,ready_timeout=300,max_gap_ns=5_000_000_000)
  lifecycle=ctrl.guarded_lifecycle(ops,ctrl.query_gpu);result['lifecycle']=lifecycle
  if lifecycle.get('status')!='passed':raise ValueError('controller lifecycle failed')
  expected=gate.read(out/'expected.json');events=lines(out/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample'];metadata=lines(out/'device_metadata.jsonl');checked=verify.verify(expected,metadata,samples);worker_events=lines(out/'worker_events.jsonl');indices={}
  for event in worker_events:
   tag=event.get('event')
   if tag in ('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'):
    if tag in indices:raise ValueError('duplicate stage event')
    indices[tag]=event
  if set(indices)!={'barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'} or len([e for e in worker_events if e.get('event')=='generate_returned'])!=1:raise ValueError('incomplete memory stage events')
  boundary={'release_ns':indices['barrier_released']['monotonic_ns'],**{name+'_ns':indices[name]['monotonic_ns'] for name in ('model_load_start','model_load_end','request_start','request_end','teardown_end')}}
  if not samples or samples[0]['query_end_ns']>boundary['release_ns'] or gate.read(out/'identity.json').get('status')!='passed' or gate.read(out/'worker_complete.json').get('status')!='completed':raise ValueError('pre-CUDA sample or model identity missing')
  stages=summarize(samples,metadata,boundary,5_000_000_000)
  if not stages['stages']['model_load']['fully_contained_samples']:raise ValueError('model load had no fully-contained samples')
  if gate.review(preflight_only=False,host=True)!=approval:raise ValueError('approval changed during run')
  memory={'status':'passed','stage_bounds_monotonic_ns':boundary,'nvml_stages':stages,'nvml_metadata':checked,'allocator_model_load':{k:indices['model_load_end'][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')},'allocator_request':{k:indices['request_end'][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')},'limitations':['single image and single process, not request-only memory','sample maxima may miss transients and are MiB-rounded','NVML device and process totals are different quantities','actual P3/DS identity depends on accepted identical C-ON adapter','processor input CPU cloning and synchronization occur during request; archive and hashing occur after request end']};save(out/'memory_summary.json',memory);result.update(status='passed',memory_summary_sha256=gate.sha(out/'memory_summary.json'),sample_count=len(samples))
 except BaseException as e:result['failure']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:save(out/'exit.json',result)
 if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
