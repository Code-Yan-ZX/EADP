"""Reviewed, bounded latency smoke launch. Full stage remains explicitly locked."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
from real_worker import PLAN,HERE,sha
from latency_worker import sources
from latency_admission import admit
from timing_launch import run_bounded,validate_budget_window

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('smoke','full'),required=True);args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid run ID')
 if args.stage!='smoke':raise ValueError('full timing locked pending smoke proof and bounded budget')
 approval=admit(args.stage,preflight_only=False);plan=json.loads((HERE/'budget_plan.json').read_text());stage=plan['smoke'];plan['budget_source']=str((HERE/plan['budget_source']).resolve())
 validate_budget_window(plan,time.time(),stage['inner_timeout_seconds'],stage['outer_timeout_seconds'],stage['grace_seconds'])
 out=HERE/('latency_'+args.run_id);monitor=HERE/('monitor_latency'+args.run_id);started=json.loads((monitor/'started.json').read_text())
 if (started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=stage['outer_timeout_seconds'] or started.get('stale_seconds')!=stage['journal_stale_seconds'] or
     started.get('progress_paths')!=[str((out/'journal.jsonl').resolve())] or started.get('snapshot_seconds')!=stage['snapshot_seconds'] or
     started.get('poll_seconds')!=stage['poll_seconds'] or started.get('grace_seconds')!=stage['grace_seconds']):raise ValueError('timing supervisor mismatch')
 if shutil.disk_usage(HERE).free<stage['minimum_free_bytes']:raise ValueError('insufficient free disk')
 sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,gpu_pids,identity
 if gpu_pids()!=[]:raise ValueError('GPU occupied')
 source={str(p):sha(p) for p in sources()}
 if source!=approval['source_sha256']:raise ValueError('reviewed source changed')
 out.mkdir(exist_ok=False);name='eadp-latency-'+args.run_id+'-'+str(os.getpid());worker=HERE/'latency_worker.py';author=json.loads(PLAN.read_text())
 cmd=container_command(author,[str(worker),'--stage',args.stage,'--out',str(out)],True,name)
 if cmd.count('--rm')!=1:raise ValueError('container retention policy mismatch')
 cmd.remove('--rm');validate_budget_window(plan,time.time(),stage['inner_timeout_seconds'],stage['outer_timeout_seconds'],stage['grace_seconds'])
 launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'approval_bundle':approval,'source_sha256':source,
         'worker_sha256':sha(worker),'launcher_sha256':sha(__file__),'budget_sha256':sha(HERE/'budget_plan.json'),'started_unix':time.time()}
 with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(cmd,name,stage['inner_timeout_seconds'],gpu_pids,identity,out)
if __name__=='__main__':sys.exit(main())
