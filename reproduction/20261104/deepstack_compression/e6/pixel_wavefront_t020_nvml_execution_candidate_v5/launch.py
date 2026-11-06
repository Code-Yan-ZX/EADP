"""Prospective sequential nine-job scheduler; GPU gate currently hard locked."""
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def monitored(cmd,cwd,timeout,cleanup_wait):
 """On interruption ask the nested supervisor to stop its child and wait."""
 proc=subprocess.Popen(cmd,cwd=cwd,start_new_session=True)
 try:return proc.wait(timeout=timeout)
 except BaseException:
  if proc.poll() is None:
   proc.terminate()
   try:proc.wait(timeout=cleanup_wait)
   except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
  raise
def run_all(jobs,run_one,record):
 if len(jobs)!=9 or [j['order'] for j in jobs]!=list(range(9)):raise ValueError('frozen nine-job order')
 for job in jobs:
  record({'event':'started','job':job})
  try:
   value=run_one(job)
   if value.get('status')!='passed' or value.get('job')!=job:raise ValueError('job result invalid')
   record({'event':'completed','job':job,'result':value})
  except BaseException as exc:
   record({'event':'failed','job':job,'error_type':type(exc).__name__,'message':str(exc)})
   raise
 return {'status':'passed','jobs':9}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if a.run_id!='001':raise ValueError('unplanned run')
 if hashlib.sha256((D/'gate.py').read_bytes()).hexdigest()!='51779490594059820db26b2625ec35e32528ef059923b6d87e7b9d073b7530c8':raise ValueError('gate source')
 s=importlib.util.spec_from_file_location('_pixel_nvml_scheduler_gate',D/'gate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);approval=g.admit_gpu()
 # Future gate must verify outer supervisor, input/weight identities, budget,
 # noGPU actual container and independent review before this continuation.
 plan=approval['plan'];jobs=approval['jobs'];src=approval['source_sha256'];sp=(D/'supervision.py').resolve()
 if hashlib.sha256(sp.read_bytes()).hexdigest()!=src[str(sp)]:raise ValueError('supervision source')
 spec=importlib.util.spec_from_file_location('_pixel_nvml_scheduler_supervision',sp);supervision=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervision)
 supervision.current(D/'monitor_run001',D/'launch.py',['--run-id','001'],plan['outer_timeout_seconds'],plan['grace_seconds'],5)
 if time.time()+plan['outer_timeout_seconds']+plan['grace_seconds']>=plan['budget_end_unix']:raise ValueError('remaining absolute budget')
 out=D/'run_001';out.mkdir(exist_ok=False)
 rp=(D/'recovery.py').resolve();gp=(B/'e6/pixel_wavefront_t020_nvml_runtime_candidate/docker_guard.py').resolve()
 for path in (rp,gp):
  if hashlib.sha256(path.read_bytes()).hexdigest()!=src[str(path)]:raise ValueError('recovery source')
 def load(name,path):
  if name in sys.modules:raise ValueError('cached recovery module')
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 recovery=load('_pixel_nvml_recovery',rp);guard=load('_pixel_nvml_owned_guard',gp)
 signals=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP);previous={sig:signal.getsignal(sig) for sig in signals}
 def interrupt(sig,frame):
  for x in signals:signal.signal(x,signal.SIG_IGN)
  raise InterruptedError('outer scheduler signal '+str(sig))
 for sig in signals:signal.signal(sig,interrupt)
 def run_one(job):
  i=job['order'];monitor=out/('monitor_job_%03d'%i)
  cmd=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(monitor),'--timeout',str(plan['job_timeout_seconds']),'--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds',str(plan['job_grace_seconds']),'--',PY,str(D/'controller_entry.py'),'--job-index',str(i)]
  try:
   rc=monitored(cmd,B,plan['job_timeout_seconds']+plan['job_grace_seconds']+30,plan['job_grace_seconds']+30)
   terminal=json.loads((monitor/'terminal.json').read_text())
   if rc!=0 or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]:raise ValueError('job supervisor failed')
   return json.loads((out/('job_%03d'%i)/'memory_summary.json').read_text())
  except BaseException:
   directory=out/('job_%03d'%i)
   if directory.exists():
    note={}
    try:note['container']=recovery.recover(job,directory,approval['plan_sha256'],guard.DockerGuard)
    except BaseException as exc:note['container']={'status':'unknown','error_type':type(exc).__name__,'message':str(exc)}
    if (directory/'launch.json').exists():
     try:note['sampler']=recovery.stop_sampler(directory)
     except BaseException as exc:note['sampler']={'status':'unknown','error_type':type(exc).__name__,'message':str(exc)}
    with (directory/'recovery.json').open('x') as f:json.dump(note,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
   raise
 try:result=run_all(jobs,run_one,lambda event:append(out/'journal.jsonl',event))
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
 if g.admit_gpu()!=approval:raise ValueError('source/weight admission changed')
 with (out/'final.json').open('x') as f:json.dump(result,f);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':raise SystemExit(main())
