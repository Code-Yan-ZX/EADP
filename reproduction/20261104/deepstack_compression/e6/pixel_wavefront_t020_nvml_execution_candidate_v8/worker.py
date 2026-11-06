"""Future isolated Pixel memory worker; gate unconditionally forbids GPU."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_nvml_runtime_candidate_v2';C=B/'e6/pixel_wavefront_t020_nvml_model_candidate';REF=B/'e6/pixel_wavefront_t020_nvml_reference'
def load(name,path):
 p=Path(path).resolve()
 if name in sys.modules:raise ValueError('cached worker module')
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def run_admitted(job,out,nonce,prepared,row,e1ref,reference,barrier,model_run,clock=time.monotonic_ns):
 """Unsafe: caller must pass future independent source/worker/GPU gate first."""
 out=Path(out);torch=prepared['torch']
 if os.getpid()!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or torch.cuda.is_initialized():raise ValueError('worker must be pre-CUDA PID1')
 expected,release=barrier.handshake(out,nonce,torch.cuda.is_initialized,timeout=300)
 if expected['mode']!=job['arm'] or expected['container_id'] not in Path('/proc/self/cgroup').read_text():raise ValueError('Pixel arm/container identity')
 append(out/'worker_events.jsonl',{'event':'barrier_released','monotonic_ns':clock(),'first_sample_ns':release['first_sample_ns'],'job':job})
 def event(v):append(out/'worker_events.jsonl',v)
 def emit(full):
  if not isinstance(full,list) or len(full)!=1 or not isinstance(full[0],list):raise ValueError('raw shape')
  save(out/'raw.json',{'job':job,'full_sequence_ids':full[0],'accepted_input_length':reference['processor']['input_ids']['shape'][1]})
 try:
  model,wrapper,result=model_run.execute(prepared,job,row,e1ref,reference,event,emit)
  save(out/'identity.json',result)
  del wrapper,model;torch.cuda.empty_cache();torch.cuda.synchronize();event({'event':'teardown_end','monotonic_ns':clock()})
  return result
 except BaseException as e:
  save(out/'worker_failure.json',{'job':job,'error_type':type(e).__name__,'message':str(e),'raw_exists':(out/'raw.json').exists()});raise
def prepare_dependencies(src):
 def checked(name,path):
  path=Path(path).resolve()
  if hashlib.sha256(path.read_bytes()).hexdigest()!=src[str(path)]:raise ValueError('worker source changed')
  return load(name,path)
 refs=checked('_pixel_nvml_refs',REF/'reference.py');all_refs,_=refs.load_refs()
 candidate_path=B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py';candidate=checked('_pixel_nvml_model',candidate_path);prepared=candidate.prepare_cpu();prepared['model_path']=candidate.MODEL
 import input_preflight
 if Path(input_preflight.__file__).resolve()!=(B/'e4_c_latency/input_preflight.py').resolve():raise ValueError('foreign input preflight')
 barrier=checked('_pixel_nvml_barrier',R/'worker_barrier.py');model_run=checked('_pixel_nvml_run',C/'model_run.py')
 if prepared['torch'].cuda.is_initialized():raise ValueError('unexpected CUDA initialization')
 return all_refs,prepared,input_preflight,barrier,model_run

def main():
 p=argparse.ArgumentParser();p.add_argument('--job-index',type=int);p.add_argument('--out',type=Path);p.add_argument('--nonce');p.add_argument('--no-gpu-dependency-smoke',action='store_true');args=p.parse_args()
 if args.no_gpu_dependency_smoke:
  if args.job_index is not None or args.out is not None or args.nonce is not None or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned noGPU smoke')
  src=json.loads((D/'cpu_preflight_plan001.json').read_text())['source_sha256'];refs,prepared,_,barrier,model_run=prepare_dependencies(src)
  if len(refs)!=9 or len(prepared['references'])!=6:raise ValueError('missing references')
  print(json.dumps({'status':'passed','references':len(refs),'cuda_initialized':False,'model_loaded':False,'barrier':str(Path(barrier.__file__).resolve()),'model_run':str(Path(model_run.__file__).resolve())}));return 0
 if args.job_index is None or args.out is None or args.nonce is None:raise ValueError('missing job arguments')
 if hashlib.sha256((D/'gate.py').read_bytes()).hexdigest()!='d31380dc45424639b2a95a1eb651fc7e040c10b724702cc691336d2431351e7b':raise ValueError('gate source')
 gate=load('_pixel_nvml_gpu_gate',D/'gate.py');approval=gate.admit_gpu()
 # A future gate must verify the source/weights/plan and independent review.
 jobs=approval['jobs']
 if args.job_index not in range(9) or args.out.resolve()!=(D/'run_001'/('job_%03d'%args.job_index)).resolve():raise ValueError('worker job/output')
 job=jobs[args.job_index];all_refs,prepared,input_preflight,barrier,model_run=prepare_dependencies(approval['source_sha256']);ref=all_refs[(job['task'],str(job['qid']),job['arm'])]
 e1=B/'e1';rows,e1refs=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 matches=[x for x in rows if (x['task'],str(x['qid']))==(job['task'],str(job['qid']))]
 if len(matches)!=1:raise ValueError('worker input identity')
 run_admitted(job,args.out,args.nonce,prepared,matches[0],e1refs[(job['task'],str(job['qid']))],ref,barrier,model_run)
 if gate.admit_gpu()!=approval:raise ValueError('approval or weights changed during run')
 save(args.out/'worker_complete.json',{'status':'completed','job':job,'GT_opened':False,'scoring_performed':False,'post_source_and_weight_check':True,'limits':['online processor/selection capture; full original intermediates not archived','capture and synchronization affect allocation','no TTFT claim']})
 return 0
if __name__=='__main__':main()
