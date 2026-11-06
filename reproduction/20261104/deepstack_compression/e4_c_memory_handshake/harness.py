"""No-GPU integration candidate; no executable entry or GPU authority.

An independently reviewed launcher must prove source/command identity and
invoke execute().  A fabricated sample is never a GPU measurement.
"""
import os
import re
import signal
import sys
import threading
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROJECT = BASE.parents[2]
RUNTIME = BASE / 'e4_c_memory_runtime'
sys.path.insert(0, str(RUNTIME))
from controller_adapter import ControllerAdapter
from controller_io import contract
from docker_guard import DockerGuard
from lifecycle import run

IMAGE = 'sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
PYTHON = '/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
FAKE_UUID = 'GPU-deadbeef'


def commands(directory, nonce):
    d = Path(directory).resolve()
    if not re.fullmatch('[0-9a-f]{32}', nonce) or not d.is_dir():
        raise ValueError('private directory and nonce required')
    if not d.is_relative_to((BASE / 'e4_c_memory_handshake').resolve()):
        raise ValueError('output directory outside rehearsal directory')
    name = 'eadp-memory-fake-' + nonce
    create = ['docker', 'create', '--network', 'none', '--cap-drop', 'ALL',
              '--security-opt', 'no-new-privileges', '--user', f'{os.getuid()}:{os.getgid()}',
              '--name', name, '--label', 'eadp.memory.nonce='+nonce,
              '--mount', f'type=bind,src={PROJECT},dst={PROJECT},readonly',
              '--mount', 'type=bind,src=/home/dell/miniconda3,dst=/home/dell/miniconda3,readonly',
              '--mount', f'type=bind,src={d},dst={d}',
              '--env', 'CUDA_VISIBLE_DEVICES=', '--env', 'PYTHONDONTWRITEBYTECODE=1',
              '--entrypoint', PYTHON, IMAGE, str(BASE/'e4_c_memory_handshake/fake_worker.py'),
              '--directory', str(d), '--nonce', nonce]
    worker = [PYTHON, str(BASE/'e4_c_memory_handshake/fake_sampler.py'), '--directory', str(d)]
    monitor = d/'sampler_monitor'
    supervisor = [PYTHON, str(BASE/'monitor/supervise.py'), '--run-dir', str(monitor),
                  '--timeout', '150', '--snapshot-seconds', '3600', '--poll-seconds', '1',
                  '--grace-seconds', '15', '--'] + worker
    return name, create, worker, supervisor, monitor


class CheckedAdapter(ControllerAdapter):
    def observe_running(self, cid):
        observed = self.guard.inspect(cid)
        raw = observed['raw']
        host = raw.get('HostConfig') or {}
        cfg = raw.get('Config') or {}
        env = cfg.get('Env') or []
        d = str(self.directory)
        expected_mounts = {(str(PROJECT), str(PROJECT), False),
                           ('/home/dell/miniconda3', '/home/dell/miniconda3', False),
                           (d, d, True)}
        mounts = raw.get('Mounts') or []
        actual_mounts = {(m.get('Source'), m.get('Destination'), m.get('RW')) for m in mounts}
        if (raw.get('Image') != IMAGE or host.get('NetworkMode') != 'none'
                or host.get('DeviceRequests') not in (None, [])
                or host.get('Devices') not in (None, [])
                or host.get('PidMode') not in ('', None)
                or [x for x in env if x.startswith('CUDA_VISIBLE_DEVICES=')] != ['CUDA_VISIBLE_DEVICES=']
                or cfg.get('User') != f'{os.getuid()}:{os.getgid()}'
                or cfg.get('Entrypoint') != [PYTHON]
                or cfg.get('Cmd') != [str(BASE/'e4_c_memory_handshake/fake_worker.py'),
                                      '--directory', d, '--nonce', self.nonce]
                or len(mounts) != 3 or actual_mounts != expected_mounts
                or 'ALL' not in (host.get('CapDrop') or [])
                or not any(x.startswith('no-new-privileges') for x in (host.get('SecurityOpt') or []))):
            raise ValueError('actual synthetic container configuration mismatch')
        return super().observe_running(cid)


def execute(directory, nonce):
    """Caller must first obtain independent review; deliberately no CLI."""
    name, create, worker, supervisor, monitor = commands(directory, nonce)
    guard = DockerGuard(name, nonce, create)
    ops = CheckedAdapter(guard, directory, 'C-ON', nonce, FAKE_UUID,
                         5_000_000_000, 90, supervisor, worker, monitor, ready_timeout=60)
    def audit(*args):
        result = contract.audit(*args)
        result['scope'] = 'synthetic fake sampler; no GPU or NVML memory measurement'
        result['fabricated_sample_bytes'] = result.pop('nvml_process_sampled_peak_bytes')
        return result
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('controller must run on main thread')
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {s: signal.getsignal(s) for s in signals}
    def interrupt(sig, _frame):
        for s in signals: signal.signal(s, signal.SIG_IGN)
        raise InterruptedError('controller received signal '+str(sig))
    try:
        for s in signals: signal.signal(s, interrupt)
        return run(ops, contract.admit_release, audit)
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)


if __name__ == '__main__':
    raise SystemExit('locked: independent prelaunch review and reviewed launcher required')
