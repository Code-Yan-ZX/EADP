"""Independent CPU-only review for the E4-C *worker preflight*, not GPU admission."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
HERE = BASE / 'e4_c_timing'
sys.path.insert(0, str(HERE))
from light_admission import sources

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def supervised(start, end, script, output=None):
    s, t = read(start), read(end)
    assert t['reason'] == 'completed' and t['worker_returncode'] == 0 and t['remaining_process_group'] == []
    assert s['worker_pid'] == t['worker_pid'] and s['process_group'] == s['worker_pid']
    cmd, cwd = s['command'], Path(s['cwd'])
    assert len([x for x in cmd if (cwd / x).resolve() == script.resolve()]) == 1
    if output is not None:
        assert cmd.count('--out') == 1 and (cwd / cmd[cmd.index('--out')+1]).resolve() == output.resolve()
    return s

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    assert args.out.resolve() == BASE / 'e1_independent_review/e4c_light_worker_preflight_review.json'
    paths = {
        'reference': HERE/'light_reference001.json', 'reference_started': HERE/'monitor_light_reference001/started.json',
        'reference_terminal': HERE/'monitor_light_reference001/terminal.json', 'cpu_proof': HERE/'cpu_check054/verification.json',
        'cpu_started': HERE/'monitor_cpu_check054/started.json', 'cpu_terminal': HERE/'monitor_cpu_check054/terminal.json',
        'preflight_launch': HERE/'light_preflight009/launch.json', 'preflight_exit': HERE/'light_preflight009/exit.json',
        'preflight_started': HERE/'monitor_light_preflight009/started.json', 'preflight_terminal': HERE/'monitor_light_preflight009/terminal.json',
    }
    supervised(paths['reference_started'], paths['reference_terminal'], HERE/'light_reference_gate.py', paths['reference'])
    supervised(paths['cpu_started'], paths['cpu_terminal'], HERE/'run_cpu_checks.py')
    ps = supervised(paths['preflight_started'], paths['preflight_terminal'], HERE/'light_preflight_launch.py')
    assert ps['command'].count('--run-id') == 1 and ps['command'][ps['command'].index('--run-id')+1] == '009'
    ref = read(paths['reference']); assert ref['status'] == 'passed' and len(ref['rows']) == 3 and len(ref['plan']['jobs']) == 16
    assert len(ref['b_records']) == 8 and len(ref['old_on']) == 4
    cpu = read(paths['cpu_proof']); assert cpu['status'] == 'passed' and cpu['returncode'] == 0
    assert cpu['source_sha256_before'] == cpu['source_sha256_after']
    for path, digest in cpu['source_sha256_after'].items(): assert sha(path) == digest, path
    for stream in ('stdout', 'stderr'): assert sha(HERE/'cpu_check054'/f'{stream}.log') == cpu[stream+'_sha256']
    stderr = (HERE/'cpu_check054/stderr.log').read_text(); assert 'Ran 49 tests' in stderr and stderr.rstrip().endswith('OK')
    expected = {str(p): sha(p) for p in sources()}; assert len(expected) == 36
    assert all(cpu['source_sha256_after'].get(k) == v for k, v in expected.items())
    launch, end = read(paths['preflight_launch']), read(paths['preflight_exit'])
    assert launch['source_sha256'] == expected and launch['gpu_requested'] is False and launch['worker_preflight'] is False
    cmd = launch['command']; assert '--gpus' not in cmd and '--device' not in cmd and launch['controller']['pid'] == ps['worker_pid']
    assert end['returncode'] == 0 and end['interrupted'] is None and end['timed_out'] is False and end['error'] is None and end['source_unchanged'] is True
    assert end['container_stopped_verified'] is True and end['client_exited_while_container_running'] is False
    ident, old = end['identity'], end['inspect']; assert ident['container_id'] == old['id']
    actual = json.loads(subprocess.check_output(['docker', 'inspect', ident['container_id']], text=True))[0]
    st = actual['State']; assert st['Running'] is False and st['Pid'] == 0 and st['ExitCode'] == 0 and not st['OOMKilled']
    assert actual['Id'] == old['id'] and actual['Image'] == old['image'] == ident['image']
    assert actual['Config']['Entrypoint'] == old['config']['Entrypoint'] and actual['Config']['Cmd'] == old['config']['Cmd']
    assert actual['HostConfig']['DeviceRequests'] in (None, []) and actual['HostConfig']['Devices'] in (None, [])
    assert actual['Config']['Cmd'] == [str(HERE/'light_container_probe.py')]
    log = (HERE/'light_preflight009/container.log').read_text(); result = json.loads([x for x in log.splitlines() if x.startswith('{')][-1])
    assert result['status'] == 'passed' and result['cuda_initialized'] is False and result['model_loaded'] is False and result['predictions'] is False
    assert result['budget_runner'] == str(BASE/'e4/budget_runner.py') and result['light_arm'] == str(HERE/'light_arm.py')
    q = {'status':'passed', 'independent_review':True, 'GT_opened':False, 'scoring_performed':False, 'errors':[],
         'scope':'worker_preflight_only', 'reviewer_sha256':sha(__file__), 'source_sha256':expected,
         'container_log_sha256':sha(HERE/'light_preflight009/container.log'),
         'limitations':'CPU-only import preflight. Actual complete worker preflight is not yet performed; no GPU diagnostic or performance conclusion is approved.'}
    for name, path in paths.items(): q[name+'_path'] = str(path); q[name+'_sha256'] = sha(path)
    with args.out.open('x') as f: json.dump(q, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'status':'passed','scope':q['scope'],'source_count':len(expected)}))

if __name__ == '__main__': main()
