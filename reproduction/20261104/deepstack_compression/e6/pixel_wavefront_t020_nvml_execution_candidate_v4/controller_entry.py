"""Candidate single-job Pixel NVML lifecycle; GPU gate remains hard locked."""
import argparse,hashlib,importlib.util,json,os,secrets,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate';H=B/'e6/pixel_wavefront_t020_nvml_host_candidate';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def read(p):return json.loads(Path(p).read_text())
def append(p,v):
 with Path(p).open('a') as f:f.write(json.dumps(v,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def checked(name,path,source):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=source[str(path)]:raise ValueError('preloaded or changed dependency')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def run_one(job,approval,ctrl,guard,runtime,stages,verify_metadata):
 """Unsafe continuation only after the future independently reviewed GPU gate."""
 if type(job.get('order')) is not int or job['order'] not in range(9) or approval['jobs'][job['order']]!=job:raise ValueError('job schedule')
 out=D/'run_001'/('job_%03d'%job['order']);out.mkdir(parents=True,exist_ok=False);nonce=secrets.token_hex(16);name='eadp-memory-pixel-'+nonce;author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author_plan=read(author)
 worker=[str(D/'worker.py'),'--job-index',str(job['order']),'--out',str(out),'--nonce',nonce];argv=runtime.container_command(author_plan,worker,True,name);create=guard.create_command(argv,name,nonce)
 inv,compute=ctrl.query_gpu();uuid=guard.parse_inventory(inv,compute)
 sampler=[PY,str(D/'sampler_entry.py'),'--directory',str(out)];sm=out/'sampler_monitor';supervisor=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(sm),'--timeout','1500','--snapshot-seconds','3600','--poll-seconds','1','--grace-seconds','15','--',*sampler]
 ctrl.publish(out/'launch.json',{'job':job,'plan_sha256':approval['plan_sha256'],'nonce':nonce,'gpu_uuid':uuid,'docker_create':create,'sampler_worker':sampler,'sampler_supervisor':supervisor,'GT_opened':False})
 ops=ctrl.construct(directory=out,mode=job['arm'],nonce=nonce,gpu_uuid=uuid,create_argv=create,name=name,image=author_plan['container_image'],worker_argv=worker,sampler_command=supervisor,sampler_worker_command=sampler,sampler_monitor=sm,worker_timeout=1200,ready_timeout=300,max_gap_ns=5_000_000_000)
 lifecycle=ctrl.guarded_lifecycle(ops,ctrl.query_gpu)
 if lifecycle.get('status')!='passed':raise ValueError('real lifecycle failed; evidence retained')
 expected=read(out/'expected.json');events=lines(out/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample'];metadata=lines(out/'device_metadata.jsonl');verify_metadata.verify(expected,metadata,samples)
 marks={}
 for event in lines(out/'worker_events.jsonl'):
  name=event.get('event')
  if name in ('barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'):
   if name in marks:raise ValueError('duplicate worker stage')
   marks[name]=event
 if set(marks)!={'barrier_released','model_load_start','model_load_end','request_start','request_end','teardown_end'}:raise ValueError('worker stage inventory')
 bounds={'release':marks['barrier_released']['monotonic_ns'],**{name:marks[name]['monotonic_ns'] for name in ('model_load_start','model_load_end','request_start','request_end','teardown_end')}}
 summary=stages.summarize(metadata,expected['host_pid'],uuid,bounds,5_000_000_000)
 if read(out/'worker_complete.json').get('status')!='completed' or read(out/'identity.json').get('status')!='passed':raise ValueError('worker identity incomplete')
 value={'status':'passed','job':job,'container_id':expected['container_id'],'host_pid':expected['host_pid'],'gpu_uuid':uuid,'stage_summary':summary,'allocator_model_load':{k:marks['model_load_end'][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')},'allocator_request':{k:marks['request_end'][k] for k in ('allocated_peak_bytes','reserved_peak_bytes')},'limitations':['MiB sampled maxima may miss transient peaks','overlap windows can contain samples outside the stage','GPU process, device and allocator are distinct quantities','graphics may coexist; unseen background unknown','CPU copies and synchronization can affect memory; no TTFT claim']}
 ctrl.publish(out/'memory_summary.json',value);return value
def prepare_dependencies(src):
 for file in ('contract.py','barrier.py','host_evidence.py','sampler.py','sampler_io.py','controller.py','controller_adapter.py','controller_io.py','docker_guard.py','guard.py','interruptible.py','lifecycle.py','sampler_supervisor.py','worker_barrier.py','real_sampler.py','nvml_source.py','verify_metadata.py'):
  path=(R/file).resolve()
  if hashlib.sha256(path.read_bytes()).hexdigest()!=src[str(path)]:raise ValueError('runtime dependency source mismatch')
 for name in ('contract','barrier','host_evidence','sampler','sampler_io','controller','controller_adapter','controller_io','docker_guard','guard','interruptible','lifecycle','sampler_supervisor','worker_barrier','real_sampler','nvml_source','verify_metadata'):
  if name in sys.modules:raise ValueError('preloaded runtime dependency '+name)
 sys.path.insert(0,str(R));import controller,guard,verify_metadata
 if any(Path(mod.__file__).resolve()!=R/name for mod,name in ((controller,'controller.py'),(guard,'guard.py'),(verify_metadata,'verify_metadata.py'))):raise ValueError('runtime module paths')
 stages=checked('_pixel_nvml_stages',H/'stages.py',src);prior=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';author=checked('_pixel_nvml_author_loader',prior,src);runtime=author.load_author(src)
 return controller,guard,runtime,stages,verify_metadata

def main():
 p=argparse.ArgumentParser();p.add_argument('--job-index',type=int);p.add_argument('--no-gpu-dependency-smoke',action='store_true');a=p.parse_args()
 if a.no_gpu_dependency_smoke:
  if a.job_index is not None or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned noGPU smoke')
  src=read(D/'cpu_preflight_plan001.json')['source_sha256'];mods=prepare_dependencies(src)
  print(json.dumps({'status':'passed','module_paths':[str(Path(m.__file__).resolve()) for m in mods],'gpu_queried':False,'model_loaded':False}));return 0
 if a.job_index is None:raise ValueError('job index')
 s=importlib.util.spec_from_file_location('_pixel_nvml_controller_gate',D/'gate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);approval=g.admit_gpu()
 if a.job_index not in range(9):raise ValueError('job index')
 supervision=checked('_pixel_nvml_job_supervision',D/'supervision.py',approval['source_sha256']);plan=approval['plan'];supervision.current(D/'run_001'/('monitor_job_%03d'%a.job_index),D/'controller_entry.py',['--job-index',str(a.job_index)],plan['job_timeout_seconds'],plan['job_grace_seconds'],5)
 controller,guard,runtime,stages,verify_metadata=prepare_dependencies(approval['source_sha256']);job=approval['jobs'][a.job_index]
 run_one(job,approval,controller,guard,runtime,stages,verify_metadata)
 if g.admit_gpu()!=approval:raise ValueError('controller approval changed')
 return 0
if __name__=='__main__':main()
