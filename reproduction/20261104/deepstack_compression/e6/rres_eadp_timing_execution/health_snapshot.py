"""Immutable post-run snapshots of the append-only automatic health logs.

No scheduler is stopped or modified. Snapshot capture waits until a bounded
in-flight pre-end health call has had time to finish; missing/failed records
remain unknown to the post-run reviewer, not evidence of no interference.
"""
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_stable_prefix(path):
    path = Path(path)
    with path.open('rb') as f:
        length = os.fstat(f.fileno()).st_size
        data = f.read(length)
    if len(data) != length or (data and not data.endswith(b'\n')):
        raise ValueError('partial append-only log snapshot')
    for line in data.splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError('malformed health log record')
    return data


def write_once(path, data):
    with Path(path).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def capture(windows, events, dest, *, process_end_ns, clock=time.monotonic_ns):
    if type(process_end_ns) is not int or clock() < process_end_ns + 50_000_000_000:
        raise ValueError('wait 50 seconds after controller end for bounded health query')
    dest = Path(dest)
    if not dest.is_dir():
        raise ValueError('snapshot directory missing')
    begin = clock()
    w, e = read_stable_prefix(windows), read_stable_prefix(events)
    end = clock()
    if end < begin:
        raise ValueError('clock moved backward')
    write_once(dest / 'automatic_health_windows_snapshot001.jsonl', w)
    write_once(dest / 'automatic_health_events_snapshot001.jsonl', e)
    record = {'status':'captured', 'read_start_monotonic_ns':begin, 'read_end_monotonic_ns':end,
              'time_namespace':os.readlink('/proc/self/ns/time'), 'controller_end_monotonic_ns':process_end_ns,
              'wait_after_controller_end_ns':begin-process_end_ns, 'windows_bytes':len(w), 'events_bytes':len(e),
              'rule':'all complete bytes present in each append-only file at the beginning of its read',
              'limits':['unknown unobserved activity', 'health logging fsync after query window is not bounded',
                        'health failures or absent records cannot establish non-overlap']}
    write_once(dest / 'automatic_health_snapshot001.json', (json.dumps(record, indent=2)+'\n').encode())
    return record


def main():
    raise RuntimeError('post-run snapshot entry locked until run and acceptance plan are frozen')


if __name__ == '__main__': main()
