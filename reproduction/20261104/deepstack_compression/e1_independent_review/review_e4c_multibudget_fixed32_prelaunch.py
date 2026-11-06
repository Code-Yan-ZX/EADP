"""Independent fixed32 diagnostic GPU prelaunch review; no GPU is used here."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B / 'e4_c_multibudget_fixed32'
A = B / 'e4_c_multibudget_fixed32_accept'
P = D / 'run_preflight001'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def terminal(directory, script):
    s, t = read(directory / 'started.json'), read(directory / 'terminal.json')
    if (t.get('reason') != 'completed' or t.get('worker_returncode') != 0 or
        t.get('remaining_process_group') != [] or s.get('worker_pid') != t.get('worker_pid') or
        (Path(s['cwd']) / s['command'][1]).resolve() != script.resolve()):
        raise ValueError('supervised command mismatch ' + str(directory))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out != R / 'e4c_multibudget_fixed32_prelaunch_review.json':
        raise ValueError('unexpected output path')
    plan = read(D / 'execution_plan005.json')
    cpu = read(D / 'execution_cpu005.json')
    sources = {path: sha(path) for path in plan['source_sha256']}
    if len(sources) != 164 or sources != plan['source_sha256'] or sources != cpu['source_sha256']:
        raise ValueError('execution sources changed')
    if plan.get('GPU_authorized_by_plan') is not False or plan.get('budgets') != [128, 256, 512, 1024]:
        raise ValueError('execution plan scope')
    terminal(D / 'monitor_execution_cpu005', D / 'freeze_execution.py')
    if cpu.get('status') != 'passed' or cpu.get('test_returncode') != 0 or cpu.get('plan_sha256') != sha(D / 'execution_plan005.json'):
        raise ValueError('execution CPU evidence')
    pre = R / 'e4c_multibudget_fixed32_worker_preflight_review.json'
    precode = R / 'review_e4c_multibudget_fixed32_worker_preflight.py'
    review = read(pre)
    terminal(R / 'monitor_e4c_multibudget_fixed32_worker_preflight', precode)
    if (review.get('status') != 'passed' or review.get('independent_review') is not True or
        review.get('scope') != 'fixed32_cpu_preflight_only' or review.get('source_sha256') != sources or
        review.get('reviewer_sha256') != sha(precode) or review.get('plan_sha256') != sha(D / 'execution_plan005.json') or
        review.get('GT_opened') is not False or review.get('predictions_opened') is not True or review.get('errors') != []):
        raise ValueError('preflight reviewer changed')
    mon = D / 'monitor_preflight001'
    s = terminal(mon, D / 'launch.py')
    if s.get('timeout_seconds') != 900 or s.get('grace_seconds') != 90 or '--preflight-only' not in s['command']:
        raise ValueError('preflight supervisor contract')
    launch, ex, worker, processor = map(read, (P/'launch.json', P/'exit.json', P/'worker_preflight.json', P/'processor_preflight.json'))
    if (launch.get('stage') != 'CPU worker preflight' or launch.get('approval_bundle') is not None or
        launch.get('plan_sha256') != sha(D/'execution_plan005.json') or
        launch.get('worker_sha256') != sha(D/'worker.py') or launch.get('launcher_sha256') != sha(D/'launch.py') or
        s['worker_pid'] != launch['controller']['pid'] or ex.get('controller') != launch['controller'] or
        ex.get('returncode') != 0 or ex.get('container_exit_code') != 0 or ex.get('timed_out') is not False or
        ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
        ex.get('client_exited_while_container_running') is not False):
        raise ValueError('preflight launch or exit')
    cmd = launch['command']
    if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or '--rm' in cmd:
        raise ValueError('preflight exposed GPU or removed container')
    cid = ex['container_identity']['container_id']
    doc = json.loads(subprocess.run(['docker', 'inspect', cid], check=True, capture_output=True, text=True, timeout=15).stdout)[0]
    state = doc['State']
    if (doc['Id'] != cid or doc['Image'] != ex['container_identity']['image'] or state['Running'] or
        state['Pid'] != 0 or state['ExitCode'] != 0 or state['OOMKilled'] or
        doc['HostConfig']['Devices'] or doc['HostConfig']['DeviceRequests'] or
        doc['HostConfig']['NetworkMode'] != 'none' or
        doc['Config']['Cmd'] != [str(D/'worker.py'), '--out', str(P), '--preflight-only'] or
        'CUDA_VISIBLE_DEVICES=' not in doc['Config']['Env']):
        raise ValueError('actual Docker state or device exposure')
    if (worker.get('status') != 'passed' or worker.get('plan_sha256') != sha(D/'execution_plan005.json') or
        worker.get('processor_preflight_sha256') != sha(P/'processor_preflight.json') or
        worker.get('GT_opened') is not False or worker.get('scoring_performed') is not False or
        worker.get('cuda_initialized') is not False or worker.get('model_loaded') is not False or
        processor.get('status') != 'passed' or processor.get('cuda_initialized') is not False or
        processor.get('model_loaded') is not False or processor.get('GT_opened') is not False or
        processor.get('same_arm_references') != 12 or len(processor.get('anchors', [])) != 3 or
        processor.get('weight_payloads_hashed') != 4 or len(processor.get('weight_stats', {})) != 4):
        raise ValueError('worker or processor preflight mismatch')
    acceptance = A / 'verification_plan001.json'
    ac = read(acceptance)
    terminal(A / 'monitor_freeze001', A / 'freeze.py')
    if (ac.get('status') != 'prepared' or ac.get('run_id') != '001' or ac.get('GPU_authorized') is not False or
        'Ran 1 test' not in ac.get('tests', '') or
        any(sha(path) != digest for path, digest in ac['source_sha256'].items())):
        raise ValueError('acceptance plan or source changed')
    fields = {'plan_sha256': D/'execution_plan005.json', 'prior_review_sha256': R/'e4c_budget_pilot_v2_preflight_review.json',
              'worker_preflight_sha256': P/'worker_preflight.json', 'processor_preflight_sha256': P/'processor_preflight.json',
              'preflight_launch_sha256': P/'launch.json', 'preflight_exit_sha256': P/'exit.json',
              'preflight_started_sha256': mon/'started.json', 'preflight_terminal_sha256': mon/'terminal.json',
              'reviewer_sha256': Path(__file__), 'execution_cpu_sha256': D/'execution_cpu005.json',
              'execution_cpu_started_sha256': D/'monitor_execution_cpu005/started.json',
              'execution_cpu_terminal_sha256': D/'monitor_execution_cpu005/terminal.json',
              'acceptance_plan_sha256': acceptance, 'acceptance_freeze_started_sha256': A/'monitor_freeze001/started.json',
              'acceptance_freeze_terminal_sha256': A/'monitor_freeze001/terminal.json'}
    proof = {'status': 'passed', 'independent_review': True, 'scope': 'gpu_fixed32_diagnostic_prelaunch',
             'GT_opened': False, 'predictions_opened': True, 'scoring_performed': False,
             'source_sha256': sources, 'errors': [], 'container_id': cid,
             'limitations': ['Only 12 anchor pairs and 32 generated tokens, not a latency experiment.',
                             'Full per-step KV arrays are not archived; equal KV fingerprints are checked.',
                             'Old DEV output was parsed for existing same-arm identity, no new GT.',
                             'Post-run source snapshot cannot exclude transient changes during execution.']}
    proof.update({key: sha(path) for key, path in fields.items()})
    with out.open('x') as f:
        json.dump(proof, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
