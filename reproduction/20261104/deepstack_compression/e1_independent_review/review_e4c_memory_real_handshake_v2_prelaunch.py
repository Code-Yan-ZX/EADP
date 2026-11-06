"""Independent prelaunch review of the bounded, no-model NVML handshake."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / 'e4_c_memory_real_handshake_v2'
R = Path(__file__).resolve()
IMAGE = 'sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def monitor(directory, expected):
    start = read(directory / 'started.json')
    end = read(directory / 'terminal.json')
    cwd = Path(start['cwd'])
    resolved = [(cwd / x).resolve() for x in start['command'] if Path(x).name == expected.name]
    assert resolved == [expected.resolve()]
    assert end['reason'] == 'completed' and end['worker_returncode'] == 0
    assert not end['remaining_process_group'] and start['worker_pid'] == end['worker_pid']
    return start, end

def container_evidence(name, target):
    out = D / name
    launch, end = read(out / 'launch.json'), read(out / 'exit.json')
    cmd = launch['command']
    assert cmd[:2] == ['docker', 'run'] and cmd[cmd.index('--network') + 1] == 'none'
    assert '--gpus' not in cmd and '--device' not in cmd and 'CUDA_VISIBLE_DEVICES=' in cmd
    assert IMAGE in cmd and str(target) in cmd
    assert launch['plan_sha256'] == sha(D / 'plan001.json')
    assert end['returncode'] == 0 and not end['timed_out'] and end['interrupted'] is None
    assert end['container_stopped_verified'] and end['container_exit_code'] == 0
    inspect = end['last_inspect']
    assert inspect['image'] == IMAGE and inspect['running'] is False and inspect['pid'] == 0
    assert inspect['exit_code'] == 0 and inspect['host_devices'] == []
    assert inspect['device_requests'] in (None, []) and str(target) in inspect['command']
    cid = end['container_identity']['container_id']
    assert cid == inspect['id'] and cid in end['container_identity']['host_cgroup']
    actual = json.loads(subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, check=True, timeout=15).stdout)[0]
    assert actual['Id'] == cid and actual['Image'] == IMAGE
    assert not actual['State']['Running'] and actual['State']['Pid'] == 0 and actual['State']['ExitCode'] == 0
    assert not actual['HostConfig']['Devices'] and not actual['HostConfig']['DeviceRequests']
    return launch, end

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    plan = read(D / 'plan001.json')
    assert plan['scope'] == 'minimal real CUDA allocation and host NVML handshake'
    assert (plan['allocation_bytes'], plan['hold_seconds']) == (16 * 1024 * 1024, 8)
    assert (plan['sample_interval_ns'], plan['max_sample_gap_ns']) == (500000000, 5000000000)
    assert (plan['worker_timeout_seconds'], plan['sampler_timeout_seconds'], plan['outer_timeout_seconds']) == (60, 150, 300)
    sources = plan['source_sha256']
    assert len(sources) == 96 and all(sha(k) == v for k, v in sources.items())
    for name in ('controller.py', 'worker.py', 'sampler.py', 'gate.py', 'container_cpu_regression.py', 'test_handshake.py'):
        assert str((D / name).resolve()) in sources
    cpu_start, _ = monitor(D / 'monitor_cpu001', D / 'cpu_check.py')
    monitor(D / 'monitor_preflight001', D / 'preflight_launch.py')
    monitor(D / 'monitor_container_cpu001', D / 'container_cpu_regression.py')
    assert '--out' in cpu_start['command'] and Path(cpu_start['command'][-1]) == D / 'cpu001.json'
    cpu = read(D / 'cpu001.json')
    assert cpu['status'] == 'passed' and cpu['tests'] == 5 and cpu['plan_sha256'] == sha(D / 'plan001.json')
    assert cpu['gpu_used'] is False and cpu['model_loaded'] is False and cpu['GT_opened'] is False
    pre = read(D / 'preflight001.json')
    assert pre['status'] == 'passed' and all(pre[k] is False for k in ('cuda_initialized', 'gpu_used', 'model_loaded', 'GT_opened'))
    container_evidence('run_preflight001', D / 'worker.py')
    extra, _ = container_evidence('container_cpu001', D / 'test_handshake.py')
    assert extra['probe_sha256'] == sha(D / 'container_cpu_regression.py')
    log = (D / 'container_cpu001/docker.log').read_text()
    assert 'Ran 5 tests' in log and '\nOK\n' in log
    assert all(name in log for name in ('test_foreign_preloaded_gate_rejected_before_gpu', 'test_direct_worker_refuses_before_cuda', 'test_sampler_rejects_unbound_output'))
    result = {'status': 'passed', 'independent_review': True, 'scope': 'minimal real GPU memory handshake prelaunch',
              'GT_opened': False, 'model_loaded': False, 'errors': [],
              'plan_sha256': sha(D / 'plan001.json'), 'cpu_sha256': sha(D / 'cpu001.json'),
              'cpu_started_sha256': sha(D / 'monitor_cpu001/started.json'), 'cpu_terminal_sha256': sha(D / 'monitor_cpu001/terminal.json'),
              'preflight_sha256': sha(D / 'preflight001.json'), 'preflight_started_sha256': sha(D / 'monitor_preflight001/started.json'),
              'preflight_terminal_sha256': sha(D / 'monitor_preflight001/terminal.json'), 'preflight_exit_sha256': sha(D / 'run_preflight001/exit.json'),
              'reviewer_sha256': sha(R), 'container_cpu_started_sha256': sha(D / 'monitor_container_cpu001/started.json'),
              'container_cpu_terminal_sha256': sha(D / 'monitor_container_cpu001/terminal.json'),
              'container_cpu_launch_sha256': sha(D / 'container_cpu001/launch.json'), 'container_cpu_exit_sha256': sha(D / 'container_cpu001/exit.json'),
              'container_cpu_log_sha256': sha(D / 'container_cpu001/docker.log'), 'container_cpu_probe_sha256': sha(D / 'container_cpu_regression.py'),
              'limitations': ['Prelaunch review is not a real sample, model measurement, or proof of uninterrupted device exclusivity.',
                              'Sampling can miss transient peaks; process usage differs from tensor allocation and device usage.']}
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
