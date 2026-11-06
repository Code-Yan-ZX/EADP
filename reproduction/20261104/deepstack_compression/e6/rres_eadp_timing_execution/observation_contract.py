"""Parse measured observer/query windows; never infer absence of unseen work."""


def _valid_window(a, b):
    return type(a) is int and type(b) is int and 0 < a < b


def observer_windows(records, namespace, clock_id):
    """Return ps, Docker and conservative first-fsync windows with labels."""
    if not records or records[-1].get('event') != 'observer_end':
        raise ValueError('observer termination unknown')
    end = records[-1]
    if end.get('time_namespace') != namespace or end.get('monotonic_clock') != clock_id:
        raise ValueError('observer clock identity')
    body = records[:-1]
    if not body or len(body) % 2 or end.get('samples') != len(body)//2 or end.get('reason') != 'stop':
        raise ValueError('observer sample count')
    windows = []
    previous_sample_end = None
    for index in range(len(body)//2):
        snap, logged = body[2*index:2*index+2]
        if snap.get('event') != 'snapshot' or logged.get('event') != 'logging_complete' or snap.get('index') != index or logged.get('index') != index:
            raise ValueError('observer sample sequence')
        if any(x.get('time_namespace') != namespace or x.get('monotonic_clock') != clock_id for x in (snap,logged)):
            raise ValueError('observer clock mismatch')
        previous_end = None
        for kind in ('ps', 'docker'):
            item = snap.get(kind)
            if not isinstance(item, dict) or item.get('returncode') != 0:
                raise ValueError('observer query failed')
            a,b = item.get('query_start_ns'), item.get('query_end_ns')
            if not _valid_window(a,b) or (previous_end is not None and a < previous_end):
                raise ValueError('observer query timing')
            if kind == 'ps' and previous_sample_end is not None and a < previous_sample_end:
                raise ValueError('observer sample ordering')
            windows.append({'source': kind, 'start_monotonic_ns':a, 'end_monotonic_ns':b, 'time_namespace':namespace})
            previous_end = b
        fsync_end = logged.get('first_fsync_end_ns')
        if not _valid_window(previous_end,fsync_end):
            raise ValueError('observer fsync bound')
        windows.append({'source':'observer_first_fsync_conservative', 'start_monotonic_ns':previous_end, 'end_monotonic_ns':fsync_end, 'time_namespace':namespace})
        previous_sample_end = fsync_end
    if type(end.get('end_ns')) is not int or end['end_ns'] < windows[-1]['end_monotonic_ns']:
        raise ValueError('observer end ordering')
    return {'windows':windows, 'samples':len(body)//2, 'second_fsync':'unknown', 'unobserved_background':'unknown'}


def supervisor_windows(records, namespace):
    if not records:
        raise ValueError('supervisor health unavailable')
    windows=[]
    for r in records:
        interval = r.get('health_snapshot_window_monotonic_ns')
        if r.get('time_namespace') != namespace or not isinstance(interval,list) or len(interval)!=2 or not _valid_window(*interval):
            raise ValueError('supervisor health window unknown')
        windows.append({'source':'supervisor_health', 'start_monotonic_ns':interval[0], 'end_monotonic_ns':interval[1], 'time_namespace':namespace})
    return windows


def automatic_windows(records, namespace):
    windows=[]
    for r in records:
        if r.get('event')!='manual_health_check' or r.get('time_namespace')!=namespace or not _valid_window(r.get('start_monotonic_ns'), r.get('end_monotonic_ns')):
            raise ValueError('automatic health window unknown')
        details=r.get('details')
        if not isinstance(details,dict) or details.get('health_exit_code')!=0:
            raise ValueError('automatic health check failed')
        windows.append({'source':'automatic_health', 'start_monotonic_ns':r['start_monotonic_ns'], 'end_monotonic_ns':r['end_monotonic_ns'], 'time_namespace':namespace})
    return {'windows':windows, 'unrecorded_activity':'unknown', 'log_fsync':'unknown'}


def automatic_lifecycle(events, start, end):
    """Bound failed calls around the run without claiming missing coverage.

    The automatic monitor's health subprocess has a 45 second timeout. A
    failure logged through 45 seconds after a request may have overlapped it;
    its actual query window is unknown. Successful entries still require the
    corresponding actual query windows and matching clock namespace.
    """
    if not _valid_window(start,end):
        raise ValueError('invalid measurement span')
    last = 0
    uncertain = []
    for i, event in enumerate(events):
        timestamp = event.get('monotonic_ns')
        if type(timestamp) is not int or timestamp < last:
            raise ValueError('automatic monitor event time unknown')
        last = timestamp
        if event.get('event') == 'health_failed' and start <= timestamp <= end + 45_000_000_000:
            uncertain.append(i)
    return {'failed_event_indices': uncertain, 'coverage':'unknown unless independently established',
            'unobserved_background':'unknown'}
