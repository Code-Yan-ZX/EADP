"""Bounded CPU-only retained container; requires independent preflight review."""
import argparse,json,os,sys,time
from pathlib import Path
import gate,preflight_gate

sys.path.insert(0,str(gate.BASE/'e4_c_latency'))
from timing_launch import run_bounded,validate_budget_window
from real_worker import PLAN

def validate_supervisor(plan,start,pid,pgid,sid,cwd,parent_argv,worker_argv,terminal_exists):
    cmd=plan['command'];worker=cmd[cmd.index('--')+1:]
    expected={'worker_pid':pid,'process_group':pid,'cwd':str(gate.BASE),'timeout_seconds':2100.0,
              'grace_seconds':120.0,'snapshot_seconds':3600.0,'poll_seconds':5.0,
              'stale_seconds':None,'progress_paths':[],'command':worker}
    if (any(start.get(k)!=v for k,v in expected.items()) or pgid!=pid or sid!=pid
        or cwd!=str(gate.BASE) or parent_argv!=cmd or worker_argv!=worker or terminal_exists):
        raise ValueError('preflight supervisor identity/budget mismatch')

def check_supervisor(plan):
    mon=gate.HERE/'monitor_preflight001';deadline=time.monotonic()+5
    while True:
        try:start=json.loads((mon/'started.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=deadline:raise ValueError('preflight supervisor absent')
            time.sleep(.05)
    def argv(pid):return [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
    pid=os.getpid();ppid=os.getppid()
    validate_supervisor(plan,start,pid,os.getpgid(pid),os.getsid(pid),os.getcwd(),argv(ppid),argv(pid),(mon/'terminal.json').exists())
    if os.getppid()!=ppid:raise ValueError('preflight supervisor disappeared')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    if a.run_id!='001':raise ValueError('only frozen preflight 001')
    plan=json.loads((gate.HERE/'preflight_plan.json').read_text());check_supervisor(plan);approval=preflight_gate.admit()
    budget=json.loads((gate.BASE/'e4_c_latency/budget_plan.json').read_text());budget['budget_source']=str((gate.BASE/'e4_c_latency'/budget['budget_source']).resolve())
    validate_budget_window(budget,time.time(),plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['grace_seconds'])
    out=gate.HERE/'container_preflight001';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(PLAN.parent));from run_panel import container_command,identity
    author=json.loads(PLAN.read_text());name='eadp-memory-real-cpu-001-'+str(os.getpid());worker=gate.HERE/'preflight_entry.py'
    cmd=container_command(author,[str(worker),'--out',str(out)],False,name)
    if cmd.count('--rm')!=1 or '--device' in cmd or '--gpus' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd:raise ValueError('CPU container command invalid')
    cmd.remove('--rm');launch={'command':cmd,'name':name,'controller':identity(os.getpid()),'approval':approval,
                               'gpu_requested':False,'worker_sha256':gate.runtime.sha(worker),'timeout_seconds':plan['inner_timeout_seconds']}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    result=run_bounded(cmd,name,plan['inner_timeout_seconds'],None,identity,out,require_gpu_idle=False)
    if gate.sources()!=approval['source_sha256']:raise ValueError('source changed during CPU preflight')
    return result

if __name__=='__main__':sys.exit(main())
