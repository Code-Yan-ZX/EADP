"""Independently admitted full latency-worker CPU preflight; no GPU devices."""
import argparse,json,os,sys,time
from pathlib import Path
from real_worker import PLAN,HERE,sha
from latency_worker import sources
from latency_admission import admit
from timing_launch import run_bounded,validate_budget_window

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid preflight identity')
 approval=admit('smoke',preflight_only=True);budget=json.loads((HERE/'budget_plan.json').read_text());budget['budget_source']=str((HERE/budget['budget_source']).resolve())
 validate_budget_window(budget,time.time(),1200,1500,90)
 monitor=HERE/('monitor_worker_preflight'+args.run_id);started=json.loads((monitor/'started.json').read_text())
 if started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=1500 or started.get('grace_seconds')!=90:raise ValueError('CPU preflight supervisor mismatch')
 out=HERE/('worker_preflight'+args.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-latency-cpu-'+args.run_id+'-'+str(os.getpid());worker=HERE/'latency_worker.py'
 cmd=container_command(plan,[str(worker),'--stage','smoke','--out',str(out),'--preflight-only'],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only container command invalid')
 cmd.remove('--rm');source={str(x):sha(x) for x in sources()}
 if source!=approval['source_sha256']:raise ValueError('reviewed source changed')
 launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'source_sha256':source,'gpu_requested':False,
         'worker_sha256':sha(worker),'approval_bundle':approval,'timeout_seconds':1200,'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,1200,None,identity,out,require_gpu_idle=False)
 if any(sha(path)!=digest for path,digest in source.items()):raise ValueError('source changed during CPU preflight')
 return result
if __name__=='__main__':sys.exit(main())
