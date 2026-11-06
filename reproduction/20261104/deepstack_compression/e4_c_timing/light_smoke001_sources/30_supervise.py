#!/usr/bin/env python3
"""Run one explicitly supplied command with bounded lifetime and progress monitoring."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def proc_group(pgid):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces and parentheses: split after its final ')'.
            raw = (entry / 'stat').read_text()
            rest = raw[raw.rfind(')') + 2:].split()
            if int(rest[2]) == pgid:
                members.append({'pid': int(entry.name), 'state': rest[0],
                                'ppid': int(rest[1]), 'start_ticks': int(rest[19])})
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members, key=lambda x: x['pid'])


def progress(paths):
    result = {}
    for path in paths:
        try:
            s = path.stat()
            result[str(path)] = {'size': s.st_size, 'mtime_ns': s.st_mtime_ns,
                                 'inode': s.st_ino}
        except FileNotFoundError:
            result[str(path)] = None
    return result


def health(paths, pgid, filesystem_paths):
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        gpu_info = {'returncode': gpu.returncode, 'stdout': gpu.stdout.strip(), 'stderr': gpu.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_info = {'error': str(exc)}
    disks = {}
    for path in filesystem_paths:
        try:
            u = shutil.disk_usage(path)
            disks[str(path)] = {'total': u.total, 'used': u.used, 'free': u.free}
        except OSError as exc:
            disks[str(path)] = {'error': str(exc)}
    try:
        mem = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
               for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith(('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:'))}
    except OSError as exc:
        mem = {'error': str(exc)}
    return {'utc': now(), 'gpu': gpu_info, 'disk': disks, 'memory': mem,
            'process_group': proc_group(pgid), 'progress': progress(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='new empty directory; never reuse an earlier run')
    parser.add_argument('--timeout', type=float, required=True, help='maximum runtime in seconds')
    parser.add_argument('--stale', type=float, help='seconds without a monitored file change; disabled if omitted')
    parser.add_argument('--progress', type=Path, action='append', default=[],
                        help='worker-owned progress file; repeatable; at least one required with --stale')
    parser.add_argument('--snapshot-seconds', type=float, default=3600)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--grace-seconds', type=float, default=20)
    parser.add_argument('--disk', type=Path, action='append', default=[], help='additional filesystem to inspect')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if not a.command or min(a.timeout, a.snapshot_seconds, a.poll_seconds, a.grace_seconds) <= 0:
        parser.error('command and positive timeout/intervals required')
    if a.stale is not None and (a.stale <= 0 or not a.progress):
        parser.error('--stale requires positive seconds and at least one --progress file')
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    watched = [p.resolve() for p in a.progress]
    disk_paths = [Path('/'), run] + [p.resolve() for p in a.disk]
    manifest = {'created_utc': now(), 'command': a.command, 'cwd': os.getcwd(),
                'timeout_seconds': a.timeout, 'stale_seconds': a.stale,
                'progress_paths': list(map(str, watched)), 'snapshot_seconds': a.snapshot_seconds,
                'poll_seconds': a.poll_seconds, 'grace_seconds': a.grace_seconds,
                'disk_paths': list(map(str, disk_paths))}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    stop_signal = [None]
    def on_signal(signum, _frame):
        stop_signal[0] = signum
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    start = time.monotonic()
    reason = 'launch_error'
    child = None
    try:
        with (run / 'stdout.log').open('xb', buffering=0) as out, (run / 'stderr.log').open('xb', buffering=0) as err, (run / 'health.jsonl').open('x', buffering=1) as log:
            child = subprocess.Popen(a.command, stdout=out, stderr=err, start_new_session=True)
            manifest['worker_pid'] = child.pid
            manifest['process_group'] = child.pid
            (run / 'started.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
            prev = progress(watched)
            last_change = start
            next_snapshot = start
            while True:
                current = time.monotonic()
                if current >= next_snapshot:
                    record = health(watched, child.pid, disk_paths)
                    record['elapsed_seconds'] = current - start
                    log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    next_snapshot = time.monotonic() + a.snapshot_seconds
                current_progress = progress(watched)
                if current_progress != prev:
                    prev = current_progress
                    last_change = time.monotonic()
                code = child.poll()
                current = time.monotonic()
                if code is not None:
                    reason = 'completed' if code == 0 else 'worker_failed'
                    break
                if stop_signal[0] is not None:
                    reason = 'supervisor_signal_' + str(stop_signal[0])
                    break
                if current - start >= a.timeout:
                    reason = 'hard_timeout'
                    break
                if a.stale is not None and current - last_change >= a.stale:
                    reason = 'progress_stale'
                    break
                time.sleep(min(a.poll_seconds, max(0.01, a.timeout - (current - start))))
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=a.grace_seconds)
                except subprocess.TimeoutExpired:
                    # The still-running leader anchors ownership of this process group.
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            log.write(json.dumps({'event': 'terminal_snapshot', **health(watched, child.pid, disk_paths)}, ensure_ascii=False) + '\n')
    except BaseException as exc:
        reason = 'supervisor_exception'
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=a.grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        (run / 'exception.txt').write_text(repr(exc) + '\n')
    finally:
        terminal = {'utc': now(), 'reason': reason, 'worker_pid': child.pid if child else None,
                    'worker_returncode': child.poll() if child else None,
                    'remaining_process_group': proc_group(child.pid) if child else [],
                    'elapsed_seconds': time.monotonic() - start}
        (run / 'terminal.json').write_text(json.dumps(terminal, indent=2) + '\n')
    return 0 if reason == 'completed' and not terminal['remaining_process_group'] else 1


if __name__ == '__main__':
    sys.exit(main())
