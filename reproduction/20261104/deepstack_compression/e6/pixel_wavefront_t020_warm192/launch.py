"""Eight warm-model containers; independently reviewed GPU gate mandatory."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];GATE_SHA='80cde09a2fd7426e2f793eaea57fcde9f988be95f8e9d7165c0a240dc80867f3'
def load(name,path,digest):
 p=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('cached or changed source')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def validate_supervisor(start,plan):
 cmd=start['command'];cwd=Path(start['cwd']);parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x];actual=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
 def flag(name):
  if parent.count(name)!=1:raise ValueError('supervisor argv')
  return parent[parent.index(name)+1]
 if (cwd.resolve()!=B or start.get('worker_pid')!=os.getpid() or start.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or cmd!=actual or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--run-id','001'] or any(start.get(k)!=plan[v] for k,v in [('timeout_seconds','outer_timeout_seconds'),('grace_seconds','grace_seconds'),('snapshot_seconds','snapshot_seconds'),('poll_seconds','poll_seconds')]) or sum((cwd/x).resolve()==B/'e4_c_latency/timing_supervise.py' for x in parent)!=1 or Path(flag('--run-dir')).resolve()!=(D/'monitor_run001').resolve() or float(flag('--timeout'))!=plan['outer_timeout_seconds'] or float(flag('--grace-seconds'))!=plan['grace_seconds'] or float(flag('--snapshot-seconds'))!=plan['snapshot_seconds'] or float(flag('--poll-seconds'))!=plan['poll_seconds'] or parent.count('--')!=1 or parent[parent.index('--')+1:]!=actual):raise ValueError('unapproved actual supervisor')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
 if a.run_id!='001':raise ValueError('unplanned run')
 gate=load('_warm192_gpu_gate',D/'gate.py',GATE_SHA);approval=gate.admit_gpu();plan=approval['plan'];source=approval['source_sha256'];validate_supervisor(gate.read(D/'monitor_run001/started.json'),plan);now=time.time()
 if now+plan['outer_timeout_seconds']+plan['grace_seconds']>=plan['budget_end_unix'] or shutil.disk_usage(D).free<plan['minimum_disk_bytes']:raise ValueError('deadline or disk')
 def get(n,p):return load(n,p,source[str(Path(p).resolve())])
 observer=get('_warm192_observer_check',D/'observer_check.py');observer.live();manual=D/'manual_health_windows001.jsonl'
 if manual.exists():raise ValueError('manual monitoring file reused')
 with manual.open('x') as f:f.flush();os.fsync(f.fileno())
 controller=get('_warm192_controller',D/'controller.py');host=get('_warm192_host',B/'e6/pixel_wavefront_t020_timing_runtime_v4/host_runner.py');bounded=get('_warm192_bounded',B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_bounded.py');author_loader=get('_warm192_author_loader',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py');runtime=author_loader.load_author(source)
 author=gate.read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 if runtime.gpu_pids()!=[]:raise ValueError('GPU busy')
 def check_supervisor():validate_supervisor(gate.read(D/'monitor_run001/started.json'),plan)
 def run_job(job,jobdir):
  name='eadp-pixel-warm192-%03d-%d'%(job['order'],os.getpid());command=runtime.container_command(author,[str(D/'worker.py'),'--job-index',str(job['order'])],True,name)
  if command.count('--rm')!=1:raise ValueError('unexpected Docker command')
  command.remove('--rm');return host.bounded_one(job,jobdir,command,name,runtime.identity(os.getpid()),approval['plan_sha256'],D/'worker.py',source[str((D/'worker.py').resolve())],bounded.run_bounded,runtime.gpu_pids,runtime.identity,check_supervisor,plan['inner_timeout_seconds'])
 def check_job(job,jobdir,exit):
  result=gate.read(jobdir/'worker_complete.json');records=[json.loads(x) for x in (jobdir/'journal.jsonl').read_text().splitlines() if x.strip()]
  if result.get('status')!='completed' or result.get('order')!=job['order'] or result.get('arm')!=job['arm'] or result.get('warmup')!=1 or result.get('measured')!=48 or result.get('post_source_and_weight_check') is not True or len(records)!=98 or any(records[2*i].get('event')!='started' or records[2*i+1].get('event')!='completed' or records[2*i].get('index')!=i or records[2*i+1].get('index')!=i for i in range(49)):raise ValueError('worker inventory')
 result=controller.run(approval['jobs'],D/'run_001',run_job,check_job,check_supervisor,plan['budget_end_unix'],plan['inner_timeout_seconds'],plan['cleanup_seconds'])
 if gate.admit_gpu()!=approval:raise ValueError('post-run approval changed')
 with (D/'run_001/final.json').open('x') as f:json.dump(dict(result,post_source_and_weight_check=True),f);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':raise SystemExit(main())
