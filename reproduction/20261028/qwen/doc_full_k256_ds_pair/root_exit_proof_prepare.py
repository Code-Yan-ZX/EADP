"""Create root exit proof only after the caller actually observed PTY exit 0.

This script cannot observe the root-owned exec session. Its caller must first
poll that exact session to completion and explicitly attest the returned exit.
It reads terminal metadata, never predictions, observations or ground truth.
"""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def current_process(pid):
    try:
        stat = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return {'pid': pid, 'start_ticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def docker_not_running(container_id):
    result = subprocess.run(['docker', 'inspect', container_id], capture_output=True, text=True)
    if result.returncode:
        require(result.returncode == 1 and ('No such object:' in result.stderr or 'No such container:' in result.stderr),
                'docker inspect did not prove absence: ' + result.stderr.strip())
        return {'inspect_exit_code': result.returncode, 'container_absent': True, 'running': False}
    info = json.loads(result.stdout)
    require(len(info) == 1 and info[0]['Id'] == container_id, 'unexpected docker inspect identity')
    require(info[0]['State']['Running'] is False, 'original container still running')
    return {'inspect_exit_code': 0, 'container_absent': False, 'running': False}


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()})


def prepare(root, handle, observed_exit, expected_plan_sha, get_process=current_process,
            inspect=docker_not_running, get_gpu_pids=gpu_pids):
    root = Path(root); out = root / 'execution'
    require(observed_exit == 0, 'root must observe the actual PTY exit code 0')
    paths = {'plan': root/'plan.json', 'state': root/'state.json', 'launch': root/'root_launch_process_snapshot.json',
             'claim': root/'launch.claim.json', 'allocation': root/'root_gpu_allocation.json',
             'release': out/'release.json', 'verification': out/'independent_verification.json',
             'complete': out/'native_complete.json', 'runtime': out/'runtime.json'}
    require(not (root/'worker_failure.json').exists() and not (root/'failure.json').exists(), 'failure record exists')
    hashes = {key: sha(path) for key, path in paths.items()}
    data = {key: read(path) for key, path in paths.items() if key != 'runtime'}
    p, s, l, claim, a, release, v, complete = (data[k] for k in
        ('plan', 'state', 'launch', 'claim', 'allocation', 'release', 'verification', 'complete'))
    require(hashes['plan'] == expected_plan_sha and len(expected_plan_sha) == 64, 'unexpected plan hash')
    require(l['root_exec_handle'] == handle, 'PTY handle does not match launch snapshot')
    require(p['run_id'] == a['run_id'] == 'Qwen_Doc5349_K256_native_selection_prefill_DS_pair', 'run ID mismatch')
    require(a['authorization'] == 'explicit_root_gpu_allocation' and a['single_attempt'] is True and a['automatic_retry'] is False, 'allocation mismatch')
    require(all(x['plan_sha256'] == hashes['plan'] for x in (s, l, claim, a, release, v, complete)), 'plan chain mismatch')
    require(all(x['allocation_sha256'] == hashes['allocation'] for x in (s, l, claim, release)), 'allocation chain mismatch')
    require(s['status'] == 'complete' and s['native_exit_code'] == 0 and s['CPU_verifier_exit_code'] == 0 and
            s['original_container_exited'] is True and s['GPU_pid_absent'] is True, 'controller not complete')
    require(l['controller'] == s['controller'] == claim['controller'] and l['docker_client'] == s['docker_client'] and
            l['container_identity'] == s['container_identity'], 'original identities mismatch')
    require(release['success'] is True and release['native_container_exited'] is True and
            s['release_sha256'] == hashes['release'] and release['independent_verification_sha256'] == hashes['verification'], 'release mismatch')
    require(v['success'] is True and v['n_questions'] == 5349 and v['n_generations'] == 10698 and v['native_anchors'] == 32,
            'verification incomplete')
    require(all(v[x] == 10698 for x in ('processor_identity_checks','selection_checks','prefill_intervention_checks','independent_token_redecodes')), 'verification counters incomplete')
    require(v['CPU_only'] is True and v['model_loaded'] is False and v['CUDA_initialized'] is False and
            v['GT_read'] is False and v['scoring_started'] is False, 'verification boundary mismatch')
    require(v['native_complete_sha256'] == hashes['complete'] and v['runtime_sha256'] == hashes['runtime'], 'terminal metadata mismatch')
    require(complete['success'] is True and complete['n_questions'] == 5349 and complete['n_generations'] == 10698 and
            complete['native_anchors'] == 32 and complete['GT_read'] is False and complete['scoring_started'] is False, 'native completion mismatch')
    for key in ('prediction_sha256','observation_sha256','ordered_prediction_sha256','ordered_observation_sha256','native_anchors_sha256'):
        require(v[key] == complete[key], 'verification/complete mismatch: ' + key)
    roles = {'controller':s['controller'], 'docker_client':s['docker_client'], 'host_init':s['container_identity']['host_init']}
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(a['boot_id'] == boot and all(x['boot_id'] == boot for x in roles.values()), 'boot changed; cannot establish original process state')
    checks = []
    for role, original in roles.items():
        now = get_process(original['pid'])
        require(now is None or (now['boot_id'], now['start_ticks']) != (original['boot_id'], original['start_ticks']),
                'original process remains: ' + role)
        checks.append({'role':role, 'original':original, 'original_exited':True, 'current_identity':now})
    cid = s['container_identity']['container_id']
    inspect_result = inspect(cid)
    require(inspect_result['running'] is False, 'original container running')
    active_gpu_pids = get_gpu_pids()
    require(roles['host_init']['pid'] not in active_gpu_pids, 'original container host PID still reported by GPU')
    # Protect against changing terminal evidence while building the proof.
    require(all(sha(path) == hashes[key] for key, path in paths.items()), 'terminal evidence changed during checks')
    return {'root_owned_exec_handle':handle, 'root_owned_exec_actual_exit_code':observed_exit,
            'root_pty_exit_evidence':'Caller attests actual root-owned PTY completion; script cannot observe session itself.',
            'all_original_processes_exited':True, 'container_not_running':True, 'container_id':cid,
            'plan_sha256':hashes['plan'], 'controller_state_sha256':hashes['state'], 'launch_snapshot_sha256':hashes['launch'],
            'allocation_sha256':hashes['allocation'], 'release_sha256':hashes['release'],
            'independent_verification_sha256':hashes['verification'], 'process_checks':checks,
            'docker_inspection':inspect_result, 'gpu_pids_after':active_gpu_pids,
            'created_utc':datetime.now(timezone.utc).isoformat()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root-observed-handle', required=True, type=int)
    ap.add_argument('--root-observed-exit-code', required=True, type=int)
    ap.add_argument('--expected-plan-sha256', required=True)
    args = ap.parse_args()
    target = HERE/'root_terminal_process_proof.json'
    require(not target.exists(), 'proof already exists; refusing overwrite')
    proof = prepare(HERE, args.root_observed_handle, args.root_observed_exit_code, args.expected_plan_sha256)
    payload = (json.dumps(proof, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    # Exclusive creation prevents accidental replacement. Remove a partial file
    # if a write fails; no proof is emitted unless the whole write succeeds.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    print(json.dumps({'path':str(target), 'sha256':sha(target), 'root_owned_exec_handle':args.root_observed_handle}))

if __name__ == '__main__':
    main()
