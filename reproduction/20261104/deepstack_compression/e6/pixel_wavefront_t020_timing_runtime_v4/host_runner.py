"""Host lifecycle implementation for one job; caller must independently gate GPU.
The real Docker lifecycle is supplied by the already reviewed bounded runner.
This draft has no CLI and grants no execution authority.
"""
import json, os, time
from pathlib import Path


def create_launch(job, jobdir, command, name, controller, plan_sha):
    d = Path(jobdir); d.mkdir(exist_ok=False)
    launch = {'job': job, 'command': command, 'container_name': name,
              'controller': controller, 'plan_sha256': plan_sha}
    with (d / 'launch.json').open('x') as f:
        json.dump(launch, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    return launch


def publish_release(job, jobdir, launch, identity, controller, expected_worker, plan_sha, worker_sha):
    import hashlib
    d = Path(jobdir); cid = identity.get('container_id'); host = identity.get('host_init') or {}; argv = identity.get('host_argv', [])
    if (launch.get('job') != job or launch.get('controller') != controller or launch.get('plan_sha256') != plan_sha
            or not isinstance(cid, str) or len(cid) != 64 or any(x not in '0123456789abcdef' for x in cid)
            or cid not in identity.get('host_cgroup', '') or argv.count(str(expected_worker)) != 1
            or argv.count('--job-index') != 1 or argv[argv.index('--job-index') + 1:argv.index('--job-index') + 2] != [str(job['order'])]
            or type(host.get('pid')) is not int or host['pid'] <= 0
            or type(host.get('start_ticks')) is not int or host['start_ticks'] <= 0 or not host.get('boot_id')
            or hashlib.sha256(Path(expected_worker).read_bytes()).hexdigest() != worker_sha):
        raise ValueError('host/container identity mismatch')
    launch_sha = hashlib.sha256((d / 'launch.json').read_bytes()).hexdigest()
    value = {'job': job, 'plan_sha256': plan_sha, 'launch_sha256': launch_sha,
             'container_id': cid, 'host_init': host, 'controller': controller}
    pending, final = d / 'release.pending', d / 'release.json'
    with pending.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(pending, final); pending.unlink()
    return value


def bounded_one(job, jobdir, command, name, controller, plan_sha, worker_path, worker_sha,
                bounded, gpu_pids, identity_fn, supervisor_check, timeout):
    """Concrete bounded Docker call. Precondition: reviewed admission and supervisor."""
    if gpu_pids() != []: raise ValueError('GPU not idle')
    supervisor_check()
    launch = create_launch(job, jobdir, command, name, controller, plan_sha)
    def release(identity):
        supervisor_check()
        publish_release(job, jobdir, launch, identity, controller, worker_path, plan_sha, worker_sha)
    code = bounded(command, name, timeout, gpu_pids, identity_fn, Path(jobdir),
                   require_gpu_idle=True, on_identity=release)
    result = json.loads((Path(jobdir) / 'exit.json').read_text())
    if (code != 0 or result.get('returncode') != 0 or result.get('container_exit_code') != 0
            or result.get('container_stopped_verified') is not True or result.get('gpu_pids_after') != []
            or result.get('timed_out') is not False or result.get('interrupted') is not None
            or result.get('client_exited_while_container_running') is not False or result.get('failure_type') is not None):
        raise ValueError('container did not exit cleanly')
    return result

if __name__ == '__main__': raise SystemExit('GPU locked; no reviewed launcher')
