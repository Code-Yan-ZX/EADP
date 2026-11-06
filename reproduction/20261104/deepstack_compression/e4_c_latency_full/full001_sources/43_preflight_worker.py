"""Independently admitted retained no-GPU full-worker preflight."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PILOT=HERE.parent/'e4_c_latency';sys.path.insert(0,str(PILOT))
from real_worker import PLAN,sha
from timing_launch import run_bounded,validate_budget_window
from full_admission import admit,sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('bad run ID')
 approval=admit(preflight_only=True,host=True);budget=json.loads((HERE/'budget_plan.json').read_text());budget['budget_source']=str((HERE/budget['budget_source']).resolve());validate_budget_window(budget,time.time(),1200,1500,90)
 monitor=HERE/('monitor_worker_preflight'+a.run_id);started=json.loads((monitor/'started.json').read_text())
 if started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=1500 or started.get('grace_seconds')!=90:raise ValueError('CPU preflight supervisor mismatch')
 source={str(p):sha(p) for p in sources()}
 if source!=approval['source_sha256']:raise ValueError('reviewed full source changed')
 out=HERE/('worker_preflight'+a.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-latency-full-cpu-'+a.run_id+'-'+str(os.getpid());worker=HERE/'full_worker.py';cmd=container_command(plan,[str(worker),'--out',str(out),'--preflight-only'],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd:raise ValueError('CPU-only worker command invalid')
 cmd.remove('--rm');launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'source_sha256':source,'worker_sha256':sha(worker),'gpu_requested':False,'approval_bundle':approval,'timeout_seconds':1200}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,1200,None,identity,out,require_gpu_idle=False)
 if any(sha(p)!=d for p,d in source.items()):raise ValueError('source changed during worker preflight')
 return result
if __name__=='__main__':sys.exit(main())
