"""Validate retained observations and annotate supplied timing windows; no deletion."""
import json


def validate(rows, expected_namespace):
    if len(rows) < 3 or len(rows) % 2 != 1:
        raise ValueError('missing paired snapshot/log events')
    end = rows[-1]
    if end.get('event') != 'watch_end' or end.get('time_namespace') != expected_namespace or end.get('reason') != 'stop_marker' or end.get('samples') != (len(rows)-1)//2:
        raise ValueError('watch end missing, expired or wrong identity')
    windows = []
    prev = -1
    watch_start = None
    for i in range((len(rows)-1) // 2):
        a, b = rows[2*i:2*i+2]
        if a.get('event') != 'snapshot' or b.get('event') != 'logging_complete' or a.get('index') != i or b.get('index') != i:
            raise ValueError('snapshot sequence or index')
        if a.get('time_namespace') != expected_namespace or b.get('time_namespace') != expected_namespace:
            raise ValueError('time namespace mismatch')
        x, y, z = a.get('query_start_ns'), a.get('query_end_ns'), b.get('first_fsync_end_ns')
        if i == 0: watch_start = a.get('watch_start_monotonic_ns')
        if type(watch_start) is not int or a.get('watch_start_monotonic_ns') != watch_start or watch_start > x: raise ValueError('observer start mismatch')
        if not all(type(v) is int for v in (x, y, z)) or not prev < x <= y <= z or b.get('query_start_ns') != x:
            raise ValueError('invalid event window')
        if not all(isinstance(s, str) and s for s in (a.get('query_start_utc'), a.get('query_end_utc'), b.get('first_fsync_end_utc'))):
            raise ValueError('missing wall timestamps')
        if a.get('returncode') != 0 or not isinstance(a.get('stdout'), str):
            raise ValueError('ps failed or result unknown')
        windows.append((x, y, z))
        prev = z
    if type(end.get('end_ns')) is not int or end['end_ns'] < prev or not end.get('end_utc'):
        raise ValueError('invalid watch end')
    return windows


def overlaps(windows, start, end):
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError('invalid measurement window')
    return {'query_indices': [i for i, (a, b, _) in enumerate(windows) if a <= end and start <= b],
            'known_query_and_first_fsync_indices': [i for i, (a, _, c) in enumerate(windows) if a <= end and start <= c],
            'second_fsync_overlap': 'unknown'}


def read_jsonl(path):
    with open(path, encoding='utf8') as f:
        return [json.loads(line) for line in f]
