"""24 isolated Docker jobs: reusable host controller, GPU CLI remains locked."""
import json, os, time
from pathlib import Path


def append(path, item):
    with Path(path).open('a', encoding='utf-8') as f:
        json.dump(item, f, ensure_ascii=False, separators=(',', ':')); f.write('\n'); f.flush(); os.fsync(f.fileno())


def validate_jobs(jobs):
    if len(jobs) != 24 or [j.get('order') for j in jobs] != list(range(24)):
        raise ValueError('schedule must contain exactly 24 ordered jobs')
    keys = [(j.get('task'), str(j.get('qid')), j.get('arm'), j.get('phase'), j.get('repeat')) for j in jobs]
    if len(set(keys)) != 24 or sum(j.get('phase') == 'warmup' for j in jobs) != 6 or sum(j.get('phase') == 'measurement' for j in jobs) != 18:
        raise ValueError('phase or job inventory mismatch')
    bases = {(task, qid, arm) for task, qid, arm, _, _ in keys}
    if len(bases) != 6: raise ValueError('six distinct image-arm identities required')
    for task, qid, arm in bases:
        phases = {(p, rep) for t, q, a, p, rep in keys if (t, q, a) == (task, qid, arm)}
        if phases != {('warmup', 0), ('measurement', 0), ('measurement', 1), ('measurement', 2)}:
            raise ValueError('each arm requires one warmup and three measurements')
    return True


def run_all(jobs, out, run_job, verify_job, validate_supervisor, budget_end, inner_seconds, cleanup_seconds, now=time.time):
    """Persist all jobs or a failing partial journal. The caller must prove admission first."""
    validate_jobs(jobs); validate_supervisor()
    if inner_seconds <= 0 or cleanup_seconds <= 0 or now() + len(jobs) * inner_seconds + cleanup_seconds >= budget_end:
        raise ValueError('insufficient absolute budget')
    out = Path(out); out.mkdir(exist_ok=False); journal = out / 'controller.jsonl'; seen = set()
    append(journal, {'event': 'controller_started', 'jobs': len(jobs)})
    try:
        for job in jobs:
            validate_supervisor()
            if now() + (len(jobs) - job['order']) * inner_seconds + cleanup_seconds >= budget_end:
                raise ValueError('remaining absolute budget insufficient')
            append(journal, {'event':'job_started','job':job})
            item = run_job(job, out / ('job_%03d' % job['order']))
            host = item.get('container_identity', {}).get('host_init', {})
            cid = item.get('container_identity', {}).get('container_id')
            process = (host.get('pid'), host.get('start_ticks'), host.get('boot_id'))
            if (not isinstance(cid, str) or len(cid) != 64 or type(process[0]) is not int or process[0] <= 0
                    or type(process[1]) is not int or process[1] <= 0 or not process[2]
                    or cid in seen or process in seen): raise ValueError('invalid/reused container or host process')
            verify_job(job, out / ('job_%03d' % job['order']), item)
            seen.add(cid); seen.add(process)
            append(journal, {'event':'job_complete','job':job,'container_id':cid,'host_init':host})
        append(journal, {'event':'all_completed','jobs':24,'warmups':6,'measured':18})
    except BaseException as exc:
        append(journal, {'event':'controller_failed','error_type':type(exc).__name__,'error':str(exc)})
        raise
    return {'jobs':24,'warmups':6,'measured':18,'all_samples_retained':True}

if __name__ == '__main__': raise SystemExit('GPU locked; frozen launcher and independent admission required')
