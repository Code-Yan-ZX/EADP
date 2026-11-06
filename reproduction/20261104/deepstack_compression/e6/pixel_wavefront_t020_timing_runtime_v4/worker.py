"""Single-process timing worker. CLI deliberately locked pending frozen admission."""
import json, os, hashlib, time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def validate_release(job, launch, release, hostname, launch_path, plan_sha):
    cid = release.get('container_id'); host = release.get('host_init') or {}
    if (not isinstance(cid, str) or len(cid) != 64 or any(x not in '0123456789abcdef' for x in cid)
            or hostname != cid[:12] or release.get('job') != job
            or launch.get('job') != job or launch.get('plan_sha256') != plan_sha
            or release.get('plan_sha256') != plan_sha or release.get('launch_sha256') != sha(launch_path)
            or release.get('controller') != launch.get('controller')
            or type(host.get('pid')) is not int or host['pid'] <= 0
            or type(host.get('start_ticks')) is not int or host['start_ticks'] <= 0
            or not host.get('boot_id')):
        raise ValueError('release identity mismatch')
    return True


def wait_release(job, jobdir, launch, hostname, plan_sha, timeout=60, clock=time.monotonic, sleep=time.sleep):
    # Host publishes release with atomic link; partial/unpublished files are not accepted.
    path = Path(jobdir) / 'release.json'; start = clock()
    while not path.exists():
        if clock() - start >= timeout: raise TimeoutError('release absent')
        sleep(.2)
    with path.open() as stream: release = json.load(stream)
    validate_release(job, launch, release, hostname, Path(jobdir) / 'launch.json', plan_sha)
    return release


def run_one(job, jobdir, prepared, row, e1ref, continuation, check):
    """Unsafe continuation: caller must first pass independently reviewed GPU gate."""
    d = Path(jobdir); ref = prepared['references'][(job['task'], str(job['qid']), job['arm'])]
    shape = ref['processor']['input_ids']['shape']
    if len(shape) != 2 or shape[0] != 1 or type(shape[1]) is not int or shape[1] < 1:
        raise ValueError('invalid accepted processor shape')
    raw = d / 'raw.json'; result_path = d / 'record.json'; called = 0
    def emit(full):
        nonlocal called
        if called or not isinstance(full, list) or len(full) != 1 or not isinstance(full[0], list):
            raise ValueError('raw output inventory')
        called += 1
        # Input length comes from the accepted processor identity and is verified
        # against the actual processor after generation by the continuation.
        write_once(raw, {'job': job, 'input_length': shape[1], 'input_length_source': 'accepted_processor_shape_pending_actual_validation', 'full_ids': full[0]})
    try:
        value = continuation(prepared, job, row, e1ref, emit)
        if called != 1 or value.get('processor_verified') is not True: raise ValueError('generation evidence missing')
        metrics = check(job, value, ref, raw, d)
        write_once(result_path, {'job': job, 'record': value, 'metrics': metrics})
        return value, metrics
    except BaseException as exc:
        write_once(d / 'worker_failure.json', {'job': job, 'error_type': type(exc).__name__, 'error': str(exc), 'raw_exists': raw.exists()})
        raise


def main():
    raise RuntimeError('GPU locked: source freeze, actual no-GPU preflight and independent GPU gate are not complete')

if __name__ == '__main__': main()
