"""Bounded retained no-GPU memory worker preflight; reviewed admission required."""
import argparse,json,os,sys,time
from pathlib import Path
from admission import HERE,LATENCY,admit,sources,sha
from timing_launch import run_bounded,validate_budget_window
from real_worker import PLAN

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run ID')
 approval=admit();budget=json.loads((LATENCY/'budget_plan.json').read_text());budget['budget_source']=str((LATENCY/budget['budget_source']).resolve());validate_budget_window(budget,time.time(),1200,1500,90)
 monitor=HERE/('monitor_container_preflight'+a.run_id);started=json.loads((monitor/'started.json').read_text())
 if started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=1500 or started.get('grace_seconds')!=90:raise ValueError('memory CPU preflight supervisor mismatch')
 frozen={str(p):sha(p) for p in sources()}
 if frozen!=approval['source_sha256']:raise ValueError('reviewed source changed')
 out=HERE/('container_preflight'+a.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-memory-cpu-'+a.run_id+'-'+str(os.getpid());worker=HERE/'preflight_entry.py';cmd=container_command(plan,[str(worker),'--out',str(out)],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only container command invalid')
 cmd.remove('--rm');launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'source_sha256':frozen,'gpu_requested':False,'worker_sha256':sha(worker),'approval_bundle':approval,'timeout_seconds':1200}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,1200,None,identity,out,require_gpu_idle=False)
 if any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('source changed during preflight')
 return result
if __name__=='__main__':sys.exit(main())
