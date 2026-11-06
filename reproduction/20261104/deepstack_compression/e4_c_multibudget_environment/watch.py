"""Low-frequency host process observations; never queries GPU or alters jobs."""
import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


def wall():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_sync(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    f.flush()
    os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stop', type=Path, required=True)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--duration', type=float, default=7500)
    args = p.parse_args()
    if args.interval < 60 or not 0 < args.duration <= 7500 or args.stop.exists():
        raise ValueError('invalid bounds or pre-existing stop marker')
    os.nice(10)
    ns = os.readlink('/proc/self/ns/time')
    start = time.monotonic()
    with args.output.open('x', encoding='utf8') as f:
        i = 0
        while time.monotonic() - start < args.duration and not args.stop.exists():
            t0, w0 = time.monotonic_ns(), wall()
            try:
                q = subprocess.run(['ps', '-eo', 'user,pid,pcpu,pmem,stat,etime,comm', '--sort=-pcpu'],
                                   capture_output=True, text=True, timeout=10, check=False)
                payload = {'returncode': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr}
            except Exception as exc:
                payload = {'returncode': None, 'error_type': type(exc).__name__, 'error': str(exc)}
            t1, w1 = time.monotonic_ns(), wall()
            write_sync(f, {'event': 'snapshot', 'index': i, 'time_namespace': ns,
                           'query_start_ns': t0, 'query_end_ns': t1,
                           'query_start_utc': w0, 'query_end_utc': w1, **payload})
            # Brackets the first line's write+fsync; the second line's own
            # write+fsync has no measured end and must remain disclosed.
            t2, w2 = time.monotonic_ns(), wall()
            write_sync(f, {'event': 'logging_complete', 'index': i,
                           'time_namespace': ns, 'query_start_ns': t0,
                           'first_fsync_end_ns': t2, 'first_fsync_end_utc': w2})
            i += 1
            target = time.monotonic() + args.interval
            while time.monotonic() < target and not args.stop.exists():
                time.sleep(min(1, max(0, target - time.monotonic())))
        write_sync(f, {'event': 'watch_end', 'time_namespace': ns,
                       'end_ns': time.monotonic_ns(), 'end_utc': wall(),
                       'reason': 'stop_marker' if args.stop.exists() else 'duration_expired',
                       'samples': i})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
