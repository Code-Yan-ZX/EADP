"""Supervised, exclusive-GPU oracle launcher; external approval defaults absent."""
import argparse,importlib.util,json,os,shutil,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SMOKE=ROOT/'m3_gpu_smoke'
sys.path[:0]=[str(HERE),str(SMOKE)]
import runtime_gate,e1_admission
spec=importlib.util.spec_from_file_location('m3_oracle_launcher_docker',SMOKE/'launcher.py')
safe=importlib.util.module_from_spec(spec);spec.loader.exec_module(safe)

def supervisor(plan,run_id):
    started=HERE/('monitor_runtime'+run_id)/'started.json';v=json.loads(started.read_text());expected=(HERE/('runtime_'+run_id)/'panel/panel_events.jsonl').resolve()
    if (v.get('worker_pid')!=os.getpid() or v.get('process_group')!=os.getpid() or v.get('timeout_seconds')!=plan['outer_timeout_seconds'] or
        v.get('stale_seconds')!=plan['progress_stale_seconds'] or v.get('progress_paths')!=[str(expected)] or
        v.get('snapshot_seconds')!=plan['snapshot_seconds'] or v.get('poll_seconds')!=plan['poll_seconds'] or v.get('grace_seconds')!=plan['grace_seconds']):
        raise ValueError('runtime supervisor budget/progress mismatch')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
    if not a.run_id.isalnum():raise ValueError('invalid run ID')
    plan=json.loads((HERE/'runtime_plan.json').read_text());supervisor(plan,a.run_id);approval=runtime_gate.validate(host=True)
    # Recheck the entire accepted E1/weights chain on the host before launch.
    e1_admission.validate()
    if shutil.disk_usage(HERE).free<10*1024**3:raise ValueError('insufficient free disk')
    sys.path.insert(0,str(safe.PLAN.parent))
    from run_panel import container_command,gpu_pids,identity
    if gpu_pids():raise ValueError('GPU occupied')
    runtime_gate.check_budget(plan,time.time())
    out=HERE/('runtime_'+a.run_id);out.mkdir(exist_ok=False);receipt=out/'host_receipt.json'
    with receipt.open('x') as f:json.dump(approval,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    source_plan=json.loads(safe.PLAN.read_text());name='eadp-m3-oracle-'+a.run_id+'-'+str(os.getpid())
    command=container_command(source_plan,[str(HERE/'runtime_worker.py'),'--execute','--out',str(out)],True,name)
    if command.count('--rm')!=1:raise ValueError('unexpected container cleanup policy')
    command.remove('--rm')
    controller=identity(os.getpid());launch={'command':command,'controller':controller,'timeout_seconds':plan['inner_timeout_seconds'],
        'host_receipt':{'path':str(receipt.resolve()),'sha256':runtime_gate.sha(receipt)},'approval':approval,
        'launcher_sha256':runtime_gate.sha(__file__),'worker_sha256':runtime_gate.sha(HERE/'runtime_worker.py'),'started_unix':time.time()}
    with (out/'launch.json').open('x') as f:json.dump(launch,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    interrupted=[None]
    def stop(signum,_):interrupted[0]=signum
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    result=safe.supervise(command,name,out,controller,plan['inner_timeout_seconds'],gpu_pids,identity,interrupted)
    if (result['returncode']!=0 or result['timed_out'] or result['interrupted'] is not None or result['errors'] or
        not result['container_stopped_verified'] or result['client_exited_while_container_running'] or result['gpu_pids_after']!=[]):return 1
    return 0
if __name__=='__main__':sys.exit(main())
