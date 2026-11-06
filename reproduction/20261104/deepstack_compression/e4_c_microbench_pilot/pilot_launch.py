"""Supervised and independently admitted Docker launcher, CPU preflight or GPU pilot."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from pilot_admission import admit,sources
from real_worker import PLAN,sha
from timing_launch import run_bounded

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.run_id!=('002' if a.preflight_only else '001'):raise ValueError('unexpected predefined run ID')
 gpu=not a.preflight_only;approval=admit(preflight_only=a.preflight_only,host=True);inner=3600 if gpu else 1200;outer=3720 if gpu else 1500;stale=1200 if gpu else None;prefix='pilot_' if gpu else 'worker_preflight';out=HERE/(prefix+a.run_id);monitor=HERE/('monitor_'+('pilot' if gpu else 'worker_preflight')+a.run_id);s=json.loads((monitor/'started.json').read_text())
 if s.get('worker_pid')!=os.getpid() or s.get('timeout_seconds')!=outer or s.get('grace_seconds')!=90 or (gpu and (s.get('stale_seconds')!=stale or s.get('progress_paths')!=[str((out/'journal.jsonl').resolve())] or s.get('snapshot_seconds')!=3600)):raise ValueError('supervisor mismatch')
 budget=json.loads((BASE/'initial_plan.json').read_text())
 if time.time()+outer>budget['budget_end_unix']:raise ValueError('three-day deadline insufficient')
 if shutil.disk_usage(HERE).free<10*1024**3:raise ValueError('insufficient disk')
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,gpu_pids,identity
 if gpu and gpu_pids()!=[]:raise ValueError('GPU occupied')
 source={str(p):sha(p) for p in sources()}
 if source!=approval['source_sha256']:raise ValueError('source changed')
 out.mkdir(exist_ok=False);name='eadp-component-'+('gpu-' if gpu else 'cpu-')+a.run_id+'-'+str(os.getpid());worker=HERE/'pilot_worker.py';cmd=container_command(json.loads(PLAN.read_text()),[str(worker),'--out',str(out)]+([] if gpu else ['--preflight-only']),gpu,name)
 if cmd.count('--rm')!=1 or (not gpu and ('--gpus' in cmd or '--device' in cmd)):raise ValueError('invalid container flags')
 cmd.remove('--rm');launch={'command':cmd,'name':name,'out':str(out),'controller':identity(os.getpid()),'approval_bundle':approval,'source_sha256':source,'worker_sha256':sha(worker),'launcher_sha256':sha(__file__),'gpu_requested':gpu,'timeout_seconds':inner}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(cmd,name,inner,gpu_pids if gpu else None,identity,out,require_gpu_idle=gpu)
if __name__=='__main__':sys.exit(main())
