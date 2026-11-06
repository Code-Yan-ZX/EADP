"""One separately reviewed no-model CUDA allocation + real host sampler."""
import importlib.util,json,os,secrets,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;ROOT=B.parents[2];PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python';AP=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path[:0]=[str(D),str(B/'e4_c_memory_real'),str(B/'e4_c_memory_runtime'),str(B/'e4_c_latency')]
import gate
import timing_launch,guard
if Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py' or Path(guard.__file__).resolve()!=B/'e4_c_memory_real/guard.py':raise ValueError('host dependency collision')
validate_budget_window=timing_launch.validate_budget_window
create_command,parse_inventory=guard.create_command,guard.parse_inventory

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
 if Path(x.__file__).resolve()!=path.resolve():raise ValueError('source mismatch')
 return x
def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if a.run_id!='001':raise ValueError('one frozen run only')
 approval=gate.verify_review();start=gate.read(D/'monitor_run001/started.json')
 if start.get('worker_pid')!=os.getpid() or start.get('timeout_seconds')!=300 or start.get('grace_seconds')!=90:raise ValueError('supervision mismatch')
 cmd=start['command'];cwd=Path(start['cwd'])
 if [(cwd/x).resolve() for x in cmd if Path(x).name=='controller.py']!=[Path(__file__).resolve()] or cmd[-2:]!=['--run-id','001']:raise ValueError('supervisor command mismatch')
 plan=gate.verify_plan();init=gate.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':init['budget_start_unix'],'budget_end_unix':init['budget_end_unix']};validate_budget_window(budget,time.time(),180,300,90)
 runtime=load(AP.parent/'run_panel.py','eadp_real_memory_container_runtime');ctrl=load(B/'e4_c_memory_real/controller.py','eadp_real_memory_controller');verify=load(B/'e4_c_memory_nvml/verify_metadata.py','eadp_real_memory_metadata')
 inv,compute=ctrl.query_gpu();uuid=parse_inventory(inv,compute)
 out=D/'run_001';out.mkdir(exist_ok=False);nonce=secrets.token_hex(16);name='eadp-memory-realhandshake-'+nonce
 worker=[str(D/'worker.py'),'--out',str(out),'--nonce',nonce];author=gate.read(AP);argv=runtime.container_command(author,worker,True,name);create=create_command(argv,name,nonce)
 sampler=[PY,str(D/'sampler.py'),'--directory',str(out)];sm=out/'sampler_monitor';supervisor=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(sm),'--timeout','150','--snapshot-seconds','3600','--poll-seconds','1','--grace-seconds','15','--']+sampler
 save(out/'launch.json',{'scope':'real no-model allocation and NVML sampling','approval':approval,'nonce':nonce,'gpu_uuid':uuid,'docker_create':create,'sampler_worker':sampler,'sampler_supervisor':supervisor,'model_loaded':False,'GT_opened':False})
 result={'status':'failed','failure':None,'scope':'real no-model allocation and NVML sampling'}
 try:
  ops=ctrl.construct(directory=out,mode='C-ON',nonce=nonce,gpu_uuid=uuid,create_argv=create,name=name,image=author['container_image'],worker_argv=worker,sampler_command=supervisor,sampler_worker_command=sampler,sampler_monitor=sm,worker_timeout=60,ready_timeout=60,max_gap_ns=5_000_000_000)
  lifecycle=ctrl.guarded_lifecycle(ops,ctrl.query_gpu);result['lifecycle']=lifecycle
  if lifecycle.get('status')!='passed':raise ValueError('lifecycle failed')
  expected=gate.read(out/'expected.json');events=lines(out/'samples.jsonl');samples=[x for x in events if x.get('type')=='sample'];metadata=lines(out/'device_metadata.jsonl');meta=verify.verify(expected,metadata,samples)
  alloc=gate.read(out/'allocation.json');done=gate.read(out/'worker_complete.json');start_ns=alloc['allocation_monotonic_ns'];end_ns=done['free_start_monotonic_ns']
  inside=[s for s in samples if s['query_start_ns']>=start_ns and s['query_end_ns']<=end_ns and type(s.get('used_bytes')) is int and s['used_bytes']>0]
  if alloc['allocation_bytes']!=16*1024*1024 or done['allocation_bytes']!=alloc['allocation_bytes'] or done['status']!='completed' or not inside:raise ValueError('allocated process not observed in held window')
  if gate.verify_review()!=approval:raise ValueError('proof/source changed')
  result.update(status='passed',metadata=meta,held_window_sample_count=len(inside),allocation=alloc,worker_complete=done,limitations=['MiB-rounded samples can miss transient peaks','CUDA context and allocator overhead differ from 16 MiB allocation','device usage is not per-process usage','no model, performance, or full-workload conclusion'])
 except BaseException as e:result['failure']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 finally:save(out/'exit.json',result)
 if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
