"""Independent prelaunch review of the natural TTFT pilot; no CUDA or model."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

B = Path(__file__).resolve().parent.parent
D = B / 'e6/pixel_natural_timing_pilot_v4'
R = Path(__file__).resolve()
IMAGE = 'sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())

def monitor(name, script):
    d = D / name
    s, t = read(d / 'started.json'), read(d / 'terminal.json')
    cwd = Path(s['cwd'])
    assert [(cwd/x).resolve() for x in s['command'] if Path(x).name == script] == [(D/script).resolve()]
    assert s['worker_pid'] == t['worker_pid'] and t['reason'] == 'completed' and t['worker_returncode'] == 0 and not t['remaining_process_group']
    return s

def inspect_container(launch, ex, expected_worker):
    command = launch['command']
    assert command[:2] == ['docker', 'run'] and command[command.index('--network')+1] == 'none'
    assert '--gpus' not in command and '--device' not in command and 'CUDA_VISIBLE_DEVICES=' in command and IMAGE in command
    assert str(expected_worker) in command and ex['returncode'] == 0 and ex['container_exit_code'] == 0
    assert ex['container_stopped_verified'] and not ex['timed_out'] and ex['interrupted'] is None and ex['failure_type'] is None
    assert ex['client_exited_while_container_running'] is False and ex['controller'] == launch['controller']
    ident, saved = ex['container_identity'], ex['last_inspect']
    cid = ident['container_id']
    assert saved['id'] == cid and cid in ident['host_cgroup'] and str(expected_worker) in ident['host_argv']
    obj = json.loads(subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True, check=True, timeout=15).stdout)[0]
    assert obj['Id'] == cid and obj['Image'] == IMAGE and not obj['State']['Running'] and obj['State']['Pid'] == 0
    assert obj['State']['ExitCode'] == 0 and not obj['State']['OOMKilled'] and not obj['HostConfig']['Devices'] and not obj['HostConfig']['DeviceRequests']
    assert obj['HostConfig']['NetworkMode'] == 'none' and str(expected_worker) in obj['Config']['Cmd']
    return cid

def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', type=Path, required=True); out = a.parse_args().out
    plan = read(D/'plan001.json'); h = sha(D/'plan001.json'); source = plan['sha256']
    assert len(source) == 100 and all(sha(p) == v for p, v in source.items())
    assert all(str((D/p).resolve()) in source for p in ('gpu_launch.py','gpu_bounded.py','gpu_worker_candidate.py','admission.py','worker.py','observer.py'))
    cpu_start = monitor('monitor_cpu001','cpu_check.py')
    pre_start = monitor('monitor_preflight001','preflight_launch.py')
    hand_start = monitor('monitor_handshake001','handshake_launch.py')
    cpu = read(D/'cpu001.json'); pre = read(D/'preflight001.json')
    assert cpu['status'] == 'passed' and cpu['source_sha256'] == source and cpu['plan_sha256'] == h
    assert 'Ran 36 tests' in cpu['test_stderr'] and cpu['GT_opened'] is False and cpu['gpu_used'] is False
    assert pre['status'] == 'passed' and pre['source_plan_sha256'] == h and pre['same_arm_references'] == 9 and pre['weight_sha256_verified'] == 4
    assert all(pre[k] is False for k in ('cuda_initialized','model_loaded','GT_opened','scoring_performed'))
    assert len(pre['anchors']) == 3 and len(pre['jobs']) == 9 and pre['jobs'] == plan['jobs']
    for item in pre['weight_stats']:
        st = Path(item['path']).stat()
        assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns] == item['stat']
    pre_launch, pre_exit = read(D/'run_preflight001/launch.json'), read(D/'run_preflight001/exit.json')
    assert pre_launch['plan_sha256'] == h and pre_launch['controller']['pid'] == pre_start['worker_pid']
    inspect_container(pre_launch, pre_exit, D/'preflight.py')
    hd = D/'run_handshake001/job_000'; hl, he, hs, release = [read(hd/x) for x in ('launch.json','exit.json','handshake.json','release.json')]
    assert hl['plan_sha256'] == h and hl['controller']['pid'] == hand_start['worker_pid'] and hl['job'] == plan['jobs'][0]
    cid = inspect_container(hl, he, D/'gpu_worker_candidate.py')
    assert '--handshake-only' in hl['command'] and hs['status'] == 'passed' and hs['cuda_initialized'] is False and hs['model_loaded'] is False
    assert hs['job'] == release['job'] == plan['jobs'][0] and hs['container_id'] == release['container_id'] == cid
    assert hs['launch_sha256'] == release['launch_sha256'] == sha(hd/'launch.json') and hs['release_sha256'] == sha(hd/'release.json')
    assert release['plan_sha256'] == h and release['controller'] == hl['controller'] and release['host_init'] == he['container_identity']['host_init']
    # Independently load same-arm evidence and ensure all actual endpoints are present.
    sys.path.insert(0,str(D)); import reference_candidate as refs
    references = refs.load_refs(plan['anchors'])
    assert len(references) == 9 and all(v['tokens'] and v['eos_ids'] and 1 <= v['selected_count'] <= 1024 for v in references.values())
    from pilot_contract import validate_schedule
    assert validate_schedule(plan['jobs'],plan['anchors'])
    paths = {'cpu':D/'cpu001.json','cpu_started':D/'monitor_cpu001/started.json','cpu_terminal':D/'monitor_cpu001/terminal.json',
             'preflight':D/'preflight001.json','preflight_launch':D/'run_preflight001/launch.json','preflight_exit':D/'run_preflight001/exit.json',
             'preflight_started':D/'monitor_preflight001/started.json','preflight_terminal':D/'monitor_preflight001/terminal.json',
             'handshake':hd/'handshake.json','handshake_launch':hd/'launch.json','handshake_exit':hd/'exit.json','handshake_release':hd/'release.json',
             'handshake_started':D/'monitor_handshake001/started.json','handshake_terminal':D/'monitor_handshake001/terminal.json'}
    result = {'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'plan_sha256':h,
              'reviewer_sha256':sha(R),'same_arm_references':9,'cpu_tests':36,
              'limitations':['No GPU/model run was performed by this review.', 'Three anchors and one timed call per arm cannot establish population latency.',
                             'Allocator peaks do not establish NVML process or total device peaks.', 'Existing mixed DEV records were parsed for reference identity.']}
    result.update({k+'_sha256':sha(v) for k,v in paths.items()})
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')

if __name__ == '__main__': main()
