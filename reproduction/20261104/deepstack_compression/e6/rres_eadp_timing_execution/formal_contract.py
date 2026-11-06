"""Pure post-run contracts for same-arm natural timing requests.

No source data, device, model, or ground truth are read here. A future formal
verifier must supply already accepted references, raw files and full event rows.
"""
CHECKS = frozenset(('tokens_match', 'response_match', 'processor_match', 'image_match', 'first_criteria_length_match', 'criteria_count_match'))
CLOCKS = ('t0_ns', 'first_ns', 'generate_end_ns', 'response_end_ns')


def check_job(job, events, references, raw_files, accepted_raw, rows):
    """Check every request and raw output in order, including warmups.

    For rres, accepted_raw contains the prior accepted full sequence (including
    prompt). For eadp it contains its already accepted generated continuation.
    No cross-arm equality is implied. Any missing/mismatching request is fatal;
    callers must retain the full output and report the failure rather than
    deleting mismatches.
    """
    requests = job['requests']
    if len(events) != 2 * len(requests) or set(raw_files) != set(range(len(requests))):
        raise ValueError('request denominator')
    for i, request in enumerate(requests):
        start, done = events[2 * i:2 * i + 2]
        if start.get('event') != 'started' or done.get('event') != 'completed' or start.get('index') != i or done.get('index') != i or start.get('request') != request or done.get('request') != request or done.get('arm') != job['arm']:
            raise ValueError('request order or identity')
        key = (job['arm'], request['task'], str(request['qid']))
        reference = references[key]
        raw = raw_files[i]
        if raw.get('arm') != job['arm'] or raw.get('request') != request or raw.get('full_raw_sequence') != [accepted_raw[key]]:
            raise ValueError('raw same-arm identity')
        if done.get('output_ids') != reference['output_ids'] or done.get('response') != reference['answer']:
            raise ValueError('output same-arm mismatch')
        checks = done.get('checks')
        if not isinstance(checks, dict) or frozenset(checks) != CHECKS or any(v is not True for v in checks.values()):
            raise ValueError('request identity checks')
        if done.get('processor') != reference['processor']:
            raise ValueError('processor identity')
        if done.get('image_sha256') != rows[(request['task'], str(request['qid']))]['image_sha256']:
            raise ValueError('image identity')
        generation = done.get('generation')
        if not isinstance(generation, dict) or generation.get('prepare_calls') != 1:
            raise ValueError('runtime generation preparation')
        explicit, effective = generation.get('explicit_kwargs'), generation.get('effective_generation_config')
        if not isinstance(explicit, dict) or not isinstance(effective, dict):
            raise ValueError('runtime generation configuration')
        if any(type(explicit.get(k)) is not type(v) or type(effective.get(k)) is not type(v) or explicit.get(k) != v or effective.get(k) != v for k, v in reference['generation_scalars'].items()) or effective.get('eos_token_id') != reference['eos_ids']:
            raise ValueError('runtime generation recipe')
        if type(effective.get('num_beams')) is not int or effective['num_beams'] != 1 or type(effective.get('num_return_sequences')) is not int or effective['num_return_sequences'] != 1 or effective.get('use_cache') is not True:
            raise ValueError('runtime beam or cache configuration')
        times = done.get('times')
        if not isinstance(times, dict) or any(type(times.get(k)) is not int for k in CLOCKS):
            raise ValueError('clock values')
        t0, first, generated, response = (times[k] for k in CLOCKS)
        if not 0 < t0 < first <= generated <= response:
            raise ValueError('clock ordering')
        first_length = len(accepted_raw[key]) - len(reference['output_ids']) + 1 if job['arm'] == 'rres' else 1
        if times.get('criteria_calls') != len(reference['output_ids']) or times.get('first_criteria_length') != first_length or times.get('raw_persisted_after_all_endpoints') is not True:
            raise ValueError('natural generation boundary')
    return {'requests': len(requests), 'warmups': sum(x['phase'] == 'warmup' for x in requests),
            'measured': sum(x['phase'] == 'measured' for x in requests)}


def known_overlaps(times, time_namespace, windows):
    """Return exact recorded overlaps; no conclusion about unobserved activity.

    Query windows are half-open [start,end); conservatively the request is
    closed [T0,response_end], including the endpoint synchronization. Any
    missing/different namespace or malformed record is unknown and
    must be rejected by the caller rather than treated as non-overlap.
    """
    if not isinstance(time_namespace, str) or not time_namespace.startswith('time:['):
        raise ValueError('unknown request time namespace')
    start, end = times['t0_ns'], times['response_end_ns']
    if type(start) is not int or type(end) is not int or not 0 < start <= end:
        raise ValueError('invalid request interval')
    result = []
    for index, w in enumerate(windows):
        a, b = w.get('start_monotonic_ns'), w.get('end_monotonic_ns')
        if w.get('time_namespace') != time_namespace or type(a) is not int or type(b) is not int or not 0 < a < b:
            raise ValueError('unknown health observation window')
        if a <= end and start < b:
            result.append(index)
    return result
