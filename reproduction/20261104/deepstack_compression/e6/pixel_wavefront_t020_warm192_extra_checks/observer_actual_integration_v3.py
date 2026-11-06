"""Real timing supervisor and observer, with GPU and Docker commands stubbed."""
import json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
import observer_guard_v3 as guard
E=Path(__file__).resolve().parent;B=E.parents[1];W=E.parent/'pixel_wavefront_t020_warm192'
supervisor=B/'e4_c_latency/timing_supervise.py'
with tempfile.TemporaryDirectory() as t:
 d=Path(t);bin=d/'bin';bin.mkdir()
 for n in ('nvidia-smi','docker'):
  p=bin/n;p.write_text('#!/bin/sh\nexit 0\n');p.chmod(0o755)
 observer=d/'observer.py';shutil.copy2(W/'observer.py',observer)
 run=d/'monitor';out=d/'observations001.jsonl';stop=d/'stop001'
 cmd=[sys.executable,str(supervisor),'--run-dir',str(run),'--timeout','23100','--snapshot-seconds','3600','--poll-seconds','1','--grace-seconds','10','--',sys.executable,str(observer),'--output',str(out),'--stop',str(stop),'--interval','60','--duration','23000']
 env=dict(os.environ,PATH=str(bin)+os.pathsep+os.environ['PATH'],CUDA_VISIBLE_DEVICES='')
 proc=subprocess.Popen(cmd,cwd=B,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  deadline=time.monotonic()+25
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise ValueError('supervisor exited early')
   if (run/'started.json').exists() and out.exists() and 'logging_complete' in out.read_text():break
   time.sleep(.1)
  else:raise TimeoutError('observer did not start')
  s=json.loads((run/'started.json').read_text());g=guard.verify(s['worker_pid'],proc.pid,supervisor,run,21600,300)
  stop.write_text('stop\n');rc=proc.wait(timeout=15)
  term=json.loads((run/'terminal.json').read_text());health=[json.loads(x) for x in (run/'health.jsonl').read_text().splitlines() if x.strip()]
  if rc!=0 or term['reason']!='completed' or term['worker_returncode']!=0 or not health or any('health_snapshot_window_monotonic_ns' not in x or 'time_namespace' not in x for x in health):raise ValueError('real supervisor health format or exit')
  proof={'status':'passed','real_supervisor':str(supervisor),'actual_guard':g,'actual_health_first':health[0],'actual_supervisor_terminal':term,'GPU_and_docker_queries_stubbed':True,'model_loaded':False,'GT_opened':False,'timing_results_opened':False}
  with (E/'observer_actual_integration_v3_001.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
