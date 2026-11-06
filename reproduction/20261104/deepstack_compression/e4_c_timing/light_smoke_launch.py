"""Independent-review-gated, bounded 16-arm diagnostic Docker launch."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
from real_worker import HERE,PLAN,sha
from timing_launch import validate_budget_window,run_bounded

def check_supervisor(started_path,progress_path,plan,now,pid):
    started=json.loads(Path(started_path).read_text())
    if (started.get('worker_pid')!=pid or started.get('timeout_seconds')!=plan['outer_timeout_seconds'] or
        started.get('stale_seconds')!=plan['journal_stale_seconds'] or started.get('progress_paths')!=[str(Path(progress_path).resolve())] or
        started.get('snapshot_seconds')!=plan['snapshot_seconds'] or started.get('poll_seconds')!=plan['poll_seconds'] or
        started.get('grace_seconds')!=plan['grace_seconds']):raise ValueError('identity supervisor configuration mismatch')
    return validate_budget_window(plan,now,plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['grace_seconds'])

def load_budget():
    budget=json.loads((HERE/'light_smoke_budget.json').read_text());source=(HERE/budget['budget_source']).resolve()
    if sha(source)!=budget['budget_source_sha256']:raise ValueError('initial research budget changed')
    budget['budget_source']=str(source)
    return budget

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--references',type=Path,required=True)
    args=p.parse_args()
    if not args.run_id.isalnum():raise ValueError('invalid run ID')
    from light_admission import admit
    approval=admit()
    if args.references.resolve()!=Path(approval['reference_path']).resolve() or sha(args.references)!=approval['reference_sha256']:raise ValueError('reference mismatch')
    budget=load_budget();out=HERE/('light_smoke_'+args.run_id);monitor=HERE/('monitor_light_smoke'+args.run_id)
    check_supervisor(monitor/'started.json',out/'generated.jsonl',budget,time.time(),os.getpid())
    if shutil.disk_usage(HERE).free<budget['minimum_free_bytes']:raise ValueError('insufficient free disk')
    sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,gpu_pids,identity
    if gpu_pids()!=[]:raise ValueError('GPU occupied')
    plan=json.loads(PLAN.read_text());out.mkdir(exist_ok=False);name='eadp-light-'+args.run_id+'-'+str(os.getpid())
    worker=HERE/'light_smoke_worker.py';command=container_command(plan,[str(worker),'--references',str(args.references.resolve()),'--out',str(out)],True,name)
    if command.count('--rm')!=1:raise ValueError('container retention policy mismatch')
    command.remove('--rm');check_supervisor(monitor/'started.json',out/'generated.jsonl',budget,time.time(),os.getpid())
    launch={'command':command,'name':name,'controller':identity(os.getpid()),'approval_bundle':approval,'worker_sha256':sha(worker),
            'launcher_sha256':sha(__file__),'budget_sha256':sha(HERE/'light_smoke_budget.json'),'timeout_seconds':budget['inner_timeout_seconds'],'started_unix':time.time()}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return run_bounded(command,name,budget['inner_timeout_seconds'],gpu_pids,identity,out)

if __name__=='__main__':sys.exit(main())
