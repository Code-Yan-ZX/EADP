"""Validate real supervisor, worker argv and remaining three-day budget."""
from pathlib import Path

def validate(s,p,pid,pgid,sid,cwd,parent,child,now,project,run):
 project=Path(project).resolve();run=Path(run).resolve();expected_script=run/'controller.py';expected_mon=run/'monitor_run001';expected_progress=run/'run_001/journal.jsonl'
 if (pid!=s.get('worker_pid') or pid!=s.get('process_group') or pid!=pgid or pid!=sid or Path(cwd).resolve()!=Path(s['cwd']).resolve() or child!=s['command'] or len(child)!=4 or (Path(cwd)/child[1]).resolve()!=expected_script or child[2:]!=['--run-id','001']):raise ValueError('worker process/argv mismatch')
 if (s.get('timeout_seconds')!=p['outer_timeout_seconds'] or s.get('grace_seconds')!=p['grace_seconds'] or s.get('stale_seconds')!=p['stale_seconds'] or s.get('poll_seconds')!=5 or s.get('snapshot_seconds')!=3600 or s.get('progress_paths')!=[str(expected_progress)]):raise ValueError('supervisor limits mismatch')
 if len(parent)<3 or (Path(cwd)/parent[1]).resolve()!=project/'monitor/supervise.py' or parent.count('--')!=1 or parent[parent.index('--')+1:]!=child:raise ValueError('actual parent supervisor argv')
 def arg(flag):
  if parent.count(flag)!=1 or parent.index(flag)+1>=len(parent):raise ValueError('missing supervisor flag '+flag)
  return parent[parent.index(flag)+1]
 if ((Path(cwd)/arg('--run-dir')).resolve()!=expected_mon or float(arg('--timeout'))!=p['outer_timeout_seconds'] or float(arg('--grace-seconds'))!=p['grace_seconds'] or float(arg('--stale'))!=p['stale_seconds'] or (Path(cwd)/arg('--progress')).resolve()!=expected_progress or float(arg('--poll-seconds'))!=5 or float(arg('--snapshot-seconds'))!=3600):raise ValueError('actual parent limits mismatch')
 if now<p['budget_start_unix'] or now+p['outer_timeout_seconds']+p['acceptance_reserve_seconds']>p['budget_end_unix']:raise ValueError('insufficient remaining deadline')
 return True
