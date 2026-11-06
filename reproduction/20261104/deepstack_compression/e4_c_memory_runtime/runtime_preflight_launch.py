"""Bounded retained no-GPU container preflight for memory runtime draft."""
import argparse,json,os,sys,time
from pathlib import Path
from runtime_admission import HERE,B,admit,source_hashes,sha
sys.path.insert(0,str(B/'e4_c_latency'))
from timing_launch import run_bounded,validate_budget_window
from real_worker import PLAN

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('invalid run ID')
 approval=admit();limits=json.loads((HERE/'preflight_plan.json').read_text());budget=json.loads((B/'e4_c_latency/budget_plan.json').read_text());budget['budget_source']=str((B/'e4_c_latency'/budget['budget_source']).resolve());validate_budget_window(budget,time.time(),limits['inner_timeout_seconds'],limits['outer_timeout_seconds'],limits['grace_seconds'])
 monitor=HERE/('monitor_container_preflight'+a.run_id);start=json.loads((monitor/'started.json').read_text())
 if start.get('worker_pid')!=os.getpid() or start.get('timeout_seconds')!=limits['outer_timeout_seconds'] or start.get('grace_seconds')!=limits['grace_seconds']:raise ValueError('runtime supervisor identity')
 frozen=source_hashes()
 if frozen!=approval['source_sha256']:raise ValueError('runtime reviewed source changed')
 out=HERE/('container_preflight'+a.run_id);out.mkdir(exist_ok=False);sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
 plan=json.loads(PLAN.read_text());name='eadp-memory-runtime-cpu-'+a.run_id+'-'+str(os.getpid());worker=HERE/'runtime_preflight_entry.py';cmd=container_command(plan,[str(worker),'--out',str(out)],False,name)
 if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('CPU-only container command invalid')
 cmd.remove('--rm');launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'source_sha256':frozen,'gpu_requested':False,'worker_sha256':sha(worker),'approval_bundle':approval,'timeout_seconds':limits['inner_timeout_seconds']}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 result=run_bounded(cmd,name,limits['inner_timeout_seconds'],None,identity,out,require_gpu_idle=False)
 if source_hashes()!=frozen:raise ValueError('runtime source changed during preflight')
 return result
if __name__=='__main__':sys.exit(main())
