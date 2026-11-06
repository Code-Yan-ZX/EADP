"""Prospective 24-process Docker controller. Unfrozen and GPU locked."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];R=B/'e6/pixel_wavefront_t020_timing_runtime_v4';C=B/'e6/pixel_wavefront_t020_model_timing_v2';V4=B/'e6/pixel_natural_timing_pilot_v4'
GATE_SHA='0287d21d559c622902531de74c87b80cd976368ec93698aea41409342dfbcd6d'
def load(name,path,expected):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('source mismatch before import')
 m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign module '+name)
  return m
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def weight_stats(paths):
 if not isinstance(paths,list) or len(paths)!=4 or len(set(paths))!=4:raise ValueError('four weight paths required')
 result=[]
 for name in paths:
  st=Path(name).stat();result.append({'path':name,'stat':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]})
 return result

def write_once(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def validate_parent_argv(argv,cwd,self_cmd,plan):
 def one(flag):
  if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):raise ValueError('supervisor flag')
  return argv[argv.index(flag)+1]
 if (sum((cwd/x).resolve()==B/'e4_c_latency/timing_supervise.py' for x in argv)!=1
     or Path(one('--run-dir')).resolve()!=(D/'monitor_run001').resolve()
     or float(one('--timeout'))!=plan['outer_timeout_seconds'] or float(one('--grace-seconds'))!=plan['grace_seconds']
     or float(one('--snapshot-seconds'))!=plan['snapshot_seconds'] or float(one('--poll-seconds'))!=plan['poll_seconds']
     or argv.count('--')!=1 or argv[argv.index('--')+1:]!=self_cmd):raise ValueError('actual parent supervisor argv')
 return True

def validate_supervisor(start, plan):
 cmd=start['command'];cwd=Path(start['cwd']);parent=os.getppid();parent_cmd=[x.decode() for x in (Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0') if x];self_cmd=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
 if (cwd.resolve()!=Path.cwd().resolve() or start.get('worker_pid')!=os.getpid() or start.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or cmd!=self_cmd or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--run-id','001'] or start.get('timeout_seconds')!=plan['outer_timeout_seconds'] or start.get('grace_seconds')!=plan['grace_seconds'] or start.get('snapshot_seconds')!=plan['snapshot_seconds'] or start.get('poll_seconds')!=plan['poll_seconds']):raise ValueError('wrong actual supervisor')
 validate_parent_argv(parent_cmd,cwd,self_cmd,plan)

def load_dependencies(source):
 get=lambda n,p:load(n,p,source[str(Path(p).resolve())])
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 paths=[('_pixel_t020_final_controller',R/'controller.py'),('_pixel_t020_final_host',R/'host_runner.py'),('_pixel_t020_final_check',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py'),('_pixel_t020_final_refs',C/'reference.py'),('_pixel_t020_final_rawprefix',D/'raw_prefix.py'),('_pixel_t020_final_bounded',D/'gpu_bounded.py'),('_pixel_t020_final_author_runtime',author.parent/'run_panel.py')]
 return tuple(get(name,path) for name,path in paths)

def smoke_dependencies():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('smoke requires no visible GPU')
 p=json.loads((D/'preflight_plan001.json').read_text())
 modules=load_dependencies(p['source_sha256'])
 return {'status':'passed','paths':[str(Path(m.__file__).resolve()) for m in modules],'gpu_queried':False,'model_loaded':False}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id');ap.add_argument('--no-gpu-dependency-smoke',action='store_true');a=ap.parse_args()
 if a.no_gpu_dependency_smoke:
  if a.run_id is not None:raise ValueError('incompatible arguments')
  print(json.dumps(smoke_dependencies()));return 0
 if a.run_id!='001':raise ValueError('unplanned run')
 gate=load('_pixel_t020_final_gpu_gate',D/'gpu_gate.py',GATE_SHA);approval=gate.admit();plan=gate.read(D/'gpu_plan001.json');validate_supervisor(gate.read(D/'monitor_run001/started.json'),plan)
 initial=gate.read(B/'initial_plan.json');now=time.time()
 if (now<initial['budget_start_unix'] or now+plan['outer_timeout_seconds']+plan['grace_seconds']>=initial['budget_end_unix'] or gate.sha(B/'initial_plan.json')!=plan['budget_source_sha256'] or shutil.disk_usage(D).free<plan['minimum_disk_bytes']):raise ValueError('absolute budget or disk')
 source=plan['source_sha256'];controller,host,check,refsmod,rawprefix,bounded,runtime=load_dependencies(source)
 author_path=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';author=gate.read(author_path);refs,_=refsmod.load()
 weights_before=weight_stats(plan['weight_files'])
 if runtime.gpu_pids()!=[]:raise ValueError('GPU busy')
 def supervise():validate_supervisor(gate.read(D/'monitor_run001/started.json'),plan)
 def run_job(job,jobdir):
  if job['order']==0:write_once(D/'run_001/weight_stats_before.json',weights_before)
  name='eadp-pixel-t020-ttft-%03d-%d'%(job['order'],os.getpid());command=runtime.container_command(author,[str(D/'worker.py'),'--job-index',str(job['order'])],True,name)
  if command.count('--rm')!=1:raise ValueError('unexpected Docker command')
  command.remove('--rm')
  return host.bounded_one(job,jobdir,command,name,runtime.identity(os.getpid()),approval['plan_sha256'],D/'worker.py',approval['worker_sha256'],bounded.run_bounded,runtime.gpu_pids,runtime.identity,supervise,plan['inner_timeout_seconds'])
 def verify_job(job,jobdir,exit):
  item=gate.read(jobdir/'record.json');ref=refs[(job['task'],str(job['qid']),job['arm'])]
  if item.get('job')!=job or check.check(job,item['record'],ref,jobdir/'raw.json',jobdir)!=item.get('metrics'):raise ValueError('worker identity')
  rawprefix.check(jobdir/'raw.json',ref)
 result=controller.run_all(approval['jobs'],D/'run_001',run_job,verify_job,supervise,initial['budget_end_unix'],plan['inner_timeout_seconds'],plan['cleanup_seconds'],time.time)
 weights_after=weight_stats(plan['weight_files']);write_once(D/'run_001/weight_stats_after.json',weights_after)
 if weights_after!=weights_before:raise ValueError('weight file identity changed during run')
 if gate.admit()!=approval:raise ValueError('sources or approval changed during run')
 with (D/'run_001/final.json').open('x') as f:
  json.dump({'status':'completed','jobs':24,'warmups':6,'measured':18,'post_run_source_and_approval_checked':True},f);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':raise SystemExit(main())
