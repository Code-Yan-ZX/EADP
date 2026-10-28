"""Reserve one diagnostic slot without interrupting a generation worker.

Temporarily SIGSTOP only the explicitly identified CPU supervisor, wait for an
existing model to exit, and resume the supervisor as soon as the diagnostic has
a CUDA context. Its existing foreign-context gate then prevents a third model.
An independent bounded watchdog and finally block both resume the supervisor.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,state=tail[0],start_ticks=int(tail[19]),
                    argv=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supervisor-pid',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[3]
    out=args.output_dir.resolve()
    original=identity(args.supervisor_pid)
    expected='json_count_distribution_overlay.py --target '+str(root/'Qwen_vl/outputs/audit_followup_20261008/streamwait_continuation_20261008/schedule_legacy_streamwait_continuation.py')
    if original is None or expected not in original['argv'] or original['state'] in ('T','t','Z'):
        raise RuntimeError('Expected live, running CPU-only supervisor identity')
    audit=out/'slot_coordination.json'
    if audit.exists():
        raise FileExistsError(audit)
    record=dict(supervisor_before=original,generation_workers_signalled=False,events=[])

    def event(action,**kwargs):
        record['events'].append(dict(action=action,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs))
        tmp=audit.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(audit)
        print(json.dumps(record['events'][-1]),flush=True)

    def same():
        current=identity(args.supervisor_pid)
        return current is not None and current['start_ticks']==original['start_ticks'] and current['argv']==original['argv']

    def resume(reason):
        if same():
            os.kill(args.supervisor_pid,signal.SIGCONT)
            event('supervisor_resumed',reason=reason)

    def terminate(signum,frame):
        raise KeyboardInterrupt(signum)

    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    # Independent watchdog survives this coordinator being killed. It checks
    # process start time to avoid signalling a reused PID.
    watchdog_code="""import os,signal,sys,time
from pathlib import Path
pid=int(sys.argv[1]);start=int(sys.argv[2]);time.sleep(1200)
try:
    tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    if int(tail[19])==start:os.kill(pid,signal.SIGCONT)
except ProcessLookupError:pass
except FileNotFoundError:pass
"""
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,str(args.supervisor_pid),str(original['start_ticks'])],
                              stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
    child=None
    try:
        if not same():raise RuntimeError('Supervisor identity changed before reservation')
        os.kill(args.supervisor_pid,signal.SIGSTOP)
        event('supervisor_launches_paused',watchdog_pid=watchdog.pid,maximum_pause_seconds=1200)
        for _ in range(50):
            if identity(args.supervisor_pid)['state'] in ('T','t'):break
            time.sleep(.1)
        deadline=time.monotonic()+1100
        while True:
            if time.monotonic()>deadline:raise TimeoutError('No free slot in bounded reservation window')
            if not same() or identity(args.supervisor_pid)['state'] not in ('T','t'):
                raise RuntimeError('Supervisor no longer holds the launch reservation')
            rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
            contexts=[]
            for row in rows:
                pid,memory=row.split(',');pid=int(pid);memory=int(memory.split()[0]);meta=identity(pid)
                if memory==0 and meta and meta['argv'].startswith('/usr/local/bin/ollama runner '):continue
                contexts.append(pid)
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if len(contexts)<=1 and free>=22500:break
            time.sleep(2)
        command=[sys.executable,str(Path(__file__).with_name('run_next_text_guidance_panel.py')),
                 '--output-dir',str(out),'--n','64','--run']
        with (out/'run.log').open('x') as log:
            child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            event('diagnostic_loading_reserved',child_pid=child.pid,existing_contexts=contexts,free_mib=free)
            while child.poll() is None:
                gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
                if str(child.pid) in gpu_pids:
                    resume('Diagnostic CUDA context now makes the existing foreign-context gate block new launches')
                    break
                time.sleep(1)
            code=child.wait()
        event('diagnostic_exited',exit_code=code)
        if code:raise RuntimeError(f'Diagnostic failed with exit {code}; preserve outputs')
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill();child.wait()
        resume('Coordinator finally block')
        # Cancel only this small watchdog once the supervisor is resumed.
        if watchdog.poll() is None:
            watchdog.terminate();watchdog.wait()


if __name__=='__main__':
    main()
