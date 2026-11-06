"""Isolated-process timing controller candidate; GPU entry remains locked.

The pure dispatch contract is exercised with fake runtime/timing objects. Real
GPU admission, observer lifetime, deadline and actual entry are not implemented.
"""
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVICES = ('/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools')


def write_once(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def dispatch(jobs, expected_jobs, runtime, bounded, author, root, approval, read):
    """Run all fixed jobs in order; identity differences are never discarded.

    Execution failures stop the run. Identity differences are retained and
    make the final result invalid, but do not omit later scheduled requests.
    This function is NOT a public GPU entry point.
    """
    if jobs != expected_jobs:
        raise ValueError('jobs do not match frozen schedule')
    if len(jobs) != 12 or [j['order'] for j in jobs] != list(range(12)):
        raise ValueError('process count or sequence')
    if sum(r['phase'] == 'warmup' for j in jobs for r in j['requests']) != 60:
        raise ValueError('warmup count')
    if sum(r['phase'] == 'measured' for j in jobs for r in j['requests']) != 320:
        raise ValueError('measured count')
    root = Path(root)
    root.mkdir()
    results = []
    for job in jobs:
        order = job['order']
        if runtime.gpu_pids() != []:
            raise ValueError('GPU busy')
        directory = root / f'job_{order:03d}'
        directory.mkdir()
        out = directory / 'observations'
        out.mkdir()
        name = f'eadp-paired-timing-{order:03d}-{os.getpid()}'
        command = runtime.container_command(author, [str(HERE / 'worker.py'), '--order', str(order), '--out', str(out)], True, name)
        devices = [command[i + 1] for i, token in enumerate(command[:-1]) if token == '--device']
        env = [command[i + 1] for i, token in enumerate(command[:-1]) if token == '--env']
        if command.count('--rm') != 1 or '--gpus' in command or sorted(devices) != sorted(DEVICES) or env.count('CUDA_VISIBLE_DEVICES=0') != 1:
            raise ValueError('invalid GPU container command')
        command.remove('--rm')
        write_once(directory / 'launch.json', {'order': order, 'job': job, 'approval': approval, 'controller': runtime.identity(os.getpid()), 'command': command, 'name': name, 'gpu_pids_before': []})
        rc = bounded.run_bounded(command, name, 1800, runtime.gpu_pids, runtime.identity, directory, require_gpu_idle=True)
        if rc:
            raise RuntimeError(f'process {order} failed')
        result = read(out / 'worker_complete.json')
        if result.get('order') != order or result.get('requests') != len(job['requests']) or result.get('all_completed') is not True or type(result.get('identity_mismatches')) is not int or result['identity_mismatches'] < 0:
            raise ValueError(f'incomplete process {order}')
        results.append({'order': order, 'identity_mismatches': result['identity_mismatches'], 'requests': result['requests']})
    mismatch_count = sum(x['identity_mismatches'] for x in results)
    status = 'passed' if not mismatch_count else 'invalid_identity'
    write_once(root / 'controller_complete.json', {'status': status, 'processes': results, 'identity_mismatches': mismatch_count, 'GT_opened': False,
        'controller_end_monotonic_ns': time.monotonic_ns(), 'time_namespace': os.readlink('/proc/self/ns/time')})
    return 0 if not mismatch_count else 1


def main():
    raise RuntimeError('GPU timing entry locked pending complete frozen admission and observer')


if __name__ == '__main__':
    main()
