"""Default is a stdlib CPU check. GPU run needs a separately reviewed root note."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import common as c


def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted({int(x) for x in value.split()})


def run(note_path):
    plan=c.read(c.PLAN);c.check(plan)
    assert c.sha(c.PLAN)==c.read(c.HERE/'READY.json')['plan_sha256']
    note_path=note_path.resolve();note=c.read(note_path);pids=gpu_pids()
    c.validate_allocation(note,c.sha(c.PLAN),pids,{pid:c.identity(pid) for pid in pids})
    assert not c.EXEC.exists() and not (c.HERE/'state.json').exists()
    own=c.identity(os.getpid())
    c.save(c.HERE/'launch.claim.json',dict(controller=own,plan_sha256=c.sha(c.PLAN),
        allocation_note_sha256=c.sha(note_path),launch_attempt=1,automatic_retry=False),exclusive=True)
    c.save(c.HERE/'allocation.json',dict(note=note,note_path=str(note_path),note_sha256=c.sha(note_path),
        actual_other_GPU_pids_before_launch=pids),exclusive=True)
    state=dict(status='starting',controller=own,plan_sha256=c.sha(c.PLAN),allocation_note_sha256=c.sha(note_path),
        started_utc=c.utc(),automatic_retry=False)
    c.save(c.HERE/'state.json',state)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        with (c.HERE/'execution.log').open('x') as log:
            process=subprocess.Popen([str(c.PYTHON),'-B',str(c.HERE/'worker.py'),str(note_path)],env=env,stdout=log,stderr=subprocess.STDOUT)
            original=c.identity(process.pid);assert original is not None
            state.update(status='running',worker=original,command=[str(c.PYTHON),'-B',str(c.HERE/'worker.py'),str(note_path)])
            c.save(c.HERE/'state.json',state)
            code=process.wait()
        state.update(status='worker_exited_validation_pending',worker_exit_code=code,worker_finished_utc=c.utc())
        c.save(c.HERE/'state.json',state)
        assert code==0 and c.identity(process.pid)!=original and process.pid not in gpu_pids()
        with (c.HERE/'verification.log').open('x') as log:
            verified=subprocess.run([str(c.PYTHON),'-B',str(c.HERE/'verify_terminal.py')],
                env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        state['CPU_verifier_exit_code']=verified.returncode;c.save(c.HERE/'state.json',state)
        assert verified.returncode==0
        c.check(plan)
        assert c.sha(note_path)==state['allocation_note_sha256']
        proof=c.read(c.EXEC/'independent_verification.json');assert proof['success'] and proof['n_questions']==16
        release=dict(status='complete',run_id=c.RUN_ID,plan_sha256=c.sha(c.PLAN),
            allocation_note_sha256=c.sha(note_path),controller=own,worker=original,worker_exit_code=0,
            actual_worker_exited=True,worker_GPU_pid_absent=True,summary_sha256=c.sha(c.EXEC/'summary.json'),
            independent_verification_sha256=c.sha(c.EXEC/'independent_verification.json'),
            measured_questions=proof['measured_questions'],unmeasured_questions=proof['unmeasured_questions'],
            controller_exit_must_be_confirmed_after_return=True,automatic_retry=False,benchmark_accuracy_computed=False)
        c.save(c.EXEC/'release.json',release,exclusive=True)
        state.update(status='complete',release_sha256=c.sha(c.EXEC/'release.json'),finished_utc=c.utc())
        c.save(c.HERE/'state.json',state);print(__import__('json').dumps(release))
    except BaseException as error:
        state.update(status='terminal_failure_no_retry',error=repr(error),finished_utc=c.utc())
        c.save(c.HERE/'failure.json',state);c.save(c.HERE/'state.json',state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='check',choices=['check','run'])
    parser.add_argument('--allocation-note',type=Path)
    args=parser.parse_args()
    if args.action=='check':
        c.check(c.read(c.PLAN))
        print(__import__('json').dumps(dict(status='prepared_not_scheduled',torch_imported='torch' in sys.modules,
            GPU_started=False,model_loaded=False,watcher_started=False)))
    else:
        assert args.allocation_note is not None;run(args.allocation_note)
