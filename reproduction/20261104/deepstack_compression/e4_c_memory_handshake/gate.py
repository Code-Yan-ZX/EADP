"""Review gate for synthetic integration rehearsal; never grants GPU access."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'e4_c_memory_runtime'))
import runtime_admission as runtime
R = BASE/'e1_independent_review'
PRIOR = R/'e4c_memory_runtime_container001_review.json'
PRIOR_REVIEWER = R/'review_e4c_memory_runtime_container001.py'
PRIOR_MON = R/'monitor_e4c_memory_runtime_container001'
REVIEW = R/'e4c_memory_fake_handshake_prelaunch_review.json'
REVIEWER = R/'review_e4c_memory_fake_handshake_prelaunch.py'
MON = R/'monitor_e4c_memory_fake_handshake_prelaunch'
CPU = HERE/'cpu_check005/verification.json'
CPUMON = HERE/'monitor_cpu005'
NAMES = ('PLAN.md', 'launch_plan.json', 'fake_worker.py', 'fake_sampler.py', 'harness.py', 'gate.py',
         'launch.py', 'cpu_check.py', 'test_handshake.py')


def sources(): return {str((HERE/n).resolve()): runtime.sha(HERE/n) for n in NAMES}
def read(p): return json.loads(Path(p).read_text())


def validate_supervision(started, plan, pid, pgid, sid, cwd, parent_argv, child_argv):
    command = plan['command']; worker = command[command.index('--')+1:]
    fields = {'worker_pid':pid, 'process_group':pid, 'cwd':str(BASE),
              'timeout_seconds':360.0, 'grace_seconds':90.0,
              'snapshot_seconds':3600.0, 'poll_seconds':2.0,
              'stale_seconds':None, 'progress_paths':[], 'command':worker}
    if (any(started.get(k) != v for k,v in fields.items())
            or pgid != pid or sid != pid or cwd != str(BASE)
            or parent_argv != command or child_argv != worker):
        raise ValueError('actual supervisor or launcher identity/command mismatch')


def verify_supervision():
    mon = HERE/'monitor_run001'; deadline = time.monotonic()+5
    plan = read(HERE/'launch_plan.json')
    while True:
        try:
            started = read(mon/'started.json')
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline: raise ValueError('live supervisor started evidence missing')
            time.sleep(.05)
    if (mon/'terminal.json').exists(): raise ValueError('supervisor already terminated')
    pid = os.getpid(); parent = os.getppid()
    def argv(p):
        return [x.decode() for x in Path(f'/proc/{p}/cmdline').read_bytes().split(b'\0') if x]
    parent_argv = argv(parent); child_argv = argv(pid)
    validate_supervision(started, plan, pid, os.getpgid(pid), os.getsid(pid), os.getcwd(), parent_argv, child_argv)
    if os.getppid() != parent or not Path(f'/proc/{parent}').exists():
        raise ValueError('supervisor disappeared during validation')


def admit():
    verify_supervision()
    upstream = runtime.admit()
    # This frozen reviewer writes its one fixed proof path; it has no --out CLI.
    runtime.supervisor(PRIOR_MON, PRIOR_REVIEWER)
    prior = read(PRIOR)
    if (prior.get('status') != 'passed' or prior.get('independent_review') is not True
            or prior.get('gpu_authorized') is not False or prior.get('errors') != []
            or prior.get('reviewer_sha256') != runtime.sha(PRIOR_REVIEWER)):
        raise ValueError('runtime container independent proof missing')
    pre = BASE/'e4_c_memory_runtime'
    bound = {'launch_sha256':pre/'container_preflight001/launch.json',
             'exit_sha256':pre/'container_preflight001/exit.json',
             'worker_preflight_sha256':pre/'container_preflight001/worker_preflight.json',
             'started_sha256':pre/'monitor_container_preflight001/started.json',
             'terminal_sha256':pre/'monitor_container_preflight001/terminal.json'}
    if any(prior.get(k) != runtime.sha(v) for k,v in bound.items()):
        raise ValueError('runtime container review evidence changed')
    runtime.supervisor(CPUMON, HERE/'cpu_check.py')
    cpu = read(CPU)
    actual = sources()
    if (cpu.get('status') != 'passed' or cpu.get('source_sha256_before') != actual
            or cpu.get('source_sha256_after') != actual or cpu.get('gpu_used') is not False):
        raise ValueError('synthetic integration CPU proof mismatch')
    runtime.supervisor(MON, REVIEWER, REVIEW)
    review = read(REVIEW)
    expected = {'status': 'passed', 'independent_review': True,
                'scope': 'no-GPU fake handshake only', 'gpu_authorized': False,
                'GT_opened': False, 'scoring_performed': False, 'errors': [],
                'source_sha256': actual, 'cpu_sha256': runtime.sha(CPU),
                'cpu_started_sha256': runtime.sha(CPUMON/'started.json'),
                'cpu_terminal_sha256': runtime.sha(CPUMON/'terminal.json'),
                'prior_review_sha256': runtime.sha(PRIOR),
                'reviewer_sha256': runtime.sha(REVIEWER)}
    if any(review.get(k) != v for k,v in expected.items()):
        raise ValueError('synthetic integration independent review mismatch')
    return {'scope':'no-GPU fake handshake only', 'gpu_authorized':False,
            'source_sha256':actual, 'runtime':upstream,
            'prior_review_sha256':runtime.sha(PRIOR), 'review_sha256':runtime.sha(REVIEW)}
