"""One 49-request warm worker; GPU entry requires all reviewed gates."""
import argparse,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];GATE_SHA='80cde09a2fd7426e2f793eaea57fcde9f988be95f8e9d7165c0a240dc80867f3'
def load(name,path,digest):
 p=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('cached or changed source')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def dependencies(source):
 get=lambda n,p:load(n,p,source[str(Path(p).resolve())])
 model=get('_warm192_model',D/'model.py');request=get('_warm192_request',D/'request.py');process=get('_warm192_process',D/'process.py');timer=get('_warm192_timer',B/'e6/pixel_wavefront_t020_model_timing_v2/t020_timing_boundaries.py');barrier=get('_warm192_barrier',B/'e6/pixel_wavefront_t020_timing_runtime_v4/worker.py');candidate,prepared,inputmod,rows,refs=model.prepare_cpu(source)
 return model,request,process,timer,barrier,candidate,prepared,inputmod,rows,refs

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--job-index',type=int);ap.add_argument('--no-gpu-dependency-smoke',action='store_true');a=ap.parse_args()
 if a.no_gpu_dependency_smoke:
  if a.job_index is not None or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU smoke arguments')
  p=json.loads((D/'preflight_plan001.json').read_text());parts=dependencies(p['source_sha256']);design=json.loads((D/'design001.json').read_text())
  for x in [q for b in design['blocks'] for q in b]+[design['processes'][0]['warmup']]:
   k=(x['task'],str(x['qid']));row=parts[-2][k];ref=parts[-1][k]
   if any(str(row.get(z))!=str(x.get(z)) for z in ('task','qid','group_id','image_sha256','message_sha256')) or ref['processor']!=x['processor']:raise ValueError('accepted input identity')
  print(json.dumps({'status':'passed','actual_inputs':193,'cuda_initialized':parts[6]['torch'].cuda.is_initialized(),'model_loaded':False,'GPU_authorized':False}));return 0
 if a.job_index is None:raise ValueError('missing job index')
 gate=load('_warm192_gpu_gate',D/'gate.py',GATE_SHA);approval=gate.admit_gpu();jobs=approval['jobs']
 if a.job_index not in range(8):raise ValueError('job index')
 job=jobs[a.job_index];out=D/'run_001'/('job_%03d'%a.job_index);launch=gate.read(out/'launch.json');cmd=launch.get('command',[])
 if cmd.count(str(D/'worker.py'))!=1 or cmd.count('--job-index')!=1 or cmd[cmd.index('--job-index')+1:cmd.index('--job-index')+2]!=[str(a.job_index)]:raise ValueError('worker argv')
 model,request,process,timer,barrier,candidate,prepared,inputmod,rows,refs=dependencies(approval['source_sha256']);release=barrier.wait_release(job,out,launch,socket.gethostname(),approval['plan_sha256'])
 if os.getpid()!=1 or release['container_id'] not in Path('/proc/self/cgroup').read_text() or prepared['torch'].cuda.is_initialized():raise ValueError('container or pre-CUDA identity')
 try:
  wrapper,load_record=model.initialize(candidate,prepared,request.RECIPE);process.save(out/'model_load.json',dict(load_record,job=job,container_id=release['container_id']))
  def execute(row,ref,arm,raw):return request.execute(wrapper,prepared['pixel'],prepared['torch'],row,ref,prepared['selector'],prepared['candidate'].select,arm,prepared['base'],timer,inputmod.validate_processor,raw)
  result=process.run(job,out,rows,refs,execute)
  if gate.admit_gpu()!=approval:raise ValueError('post-run source/weight drift')
  process.save(out/'worker_complete.json',dict(result,post_source_and_weight_check=True))
 except BaseException as exc:
  process.save(out/'worker_failure.json',{'job_order':job['order'],'error_type':type(exc).__name__,'error':str(exc)});raise
 return 0
if __name__=='__main__':raise SystemExit(main())
