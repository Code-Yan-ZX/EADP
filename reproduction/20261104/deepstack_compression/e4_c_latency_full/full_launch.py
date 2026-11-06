"""Bounded, exclusively scheduled full-latency launch after final review."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PILOT=HERE.parent/'e4_c_latency';sys.path.insert(0,str(PILOT))
from real_worker import PLAN,sha
from timing_launch import run_bounded,validate_budget_window
from full_admission import admit,sources

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if not a.run_id.isalnum():raise ValueError('bad run ID')
 approval=admit(preflight_only=False,host=True);budget=json.loads((HERE/'budget_plan.json').read_text());budget['budget_source']=str((HERE/budget['budget_source']).resolve());validate_budget_window(budget,time.time(),budget['inner_timeout_seconds'],budget['outer_timeout_seconds'],budget['grace_seconds'])
 out=HERE/('full_'+a.run_id);monitor=HERE/('monitor_full'+a.run_id);started=json.loads((monitor/'started.json').read_text())
 if (started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=budget['outer_timeout_seconds'] or started.get('stale_seconds')!=budget['journal_stale_seconds'] or started.get('progress_paths')!=[str((out/'journal.jsonl').resolve())] or
     started.get('snapshot_seconds')!=budget['snapshot_seconds'] or started.get('poll_seconds')!=budget['poll_seconds'] or started.get('grace_seconds')!=budget['grace_seconds']):raise ValueError('full supervisor mismatch')
 if shutil.disk_usage(HERE).free<budget['minimum_free_bytes']:raise ValueError('insufficient disk')
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,gpu_pids,identity
 if gpu_pids()!=[]:raise ValueError('GPU occupied')
 source={str(p):sha(p) for p in sources()}
 if source!=approval['source_sha256']:raise ValueError('reviewed full source changed')
 out.mkdir(exist_ok=False);name='eadp-latency-full-'+a.run_id+'-'+str(os.getpid());worker=HERE/'full_worker.py';plan=json.loads(PLAN.read_text());cmd=container_command(plan,[str(worker),'--out',str(out)],True,name)
 if cmd.count('--rm')!=1:raise ValueError('container retention mismatch')
 cmd.remove('--rm');validate_budget_window(budget,time.time(),budget['inner_timeout_seconds'],budget['outer_timeout_seconds'],budget['grace_seconds'])
 launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'approval_bundle':approval,'source_sha256':source,'worker_sha256':sha(worker),'launcher_sha256':sha(__file__),'budget_sha256':sha(HERE/'budget_plan.json'),'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(cmd,name,budget['inner_timeout_seconds'],gpu_pids,identity,out)
if __name__=='__main__':sys.exit(main())
