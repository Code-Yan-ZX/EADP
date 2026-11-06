"""Independent prelaunch review of complete fixed32 reference generation."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B/'e4_c_multibudget_references192'
A = B/'e4_c_multibudget_references192_accept'
F = B/'e4_c_multibudget_fixed32_accept'
P = D/'run_preflight001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def terminal(folder, script):
    s, t = read(folder/'started.json'), read(folder/'terminal.json')
    if (t.get('reason') != 'completed' or t.get('worker_returncode') != 0 or
        t.get('remaining_process_group') != [] or s.get('worker_pid') != t.get('worker_pid') or
        (Path(s['cwd'])/s['command'][1]).resolve() != script.resolve()):
        raise ValueError('supervisor mismatch '+str(folder))
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out != R/'e4c_multibudget_references192_prelaunch_review.json':
        raise ValueError('unplanned review output')
    plan_path = D/'execution_plan001.json'
    p, c = read(plan_path), read(D/'execution_cpu001.json')
    actual = {name: sha(name) for name in p['source_sha256']}
    if (len(actual) != 235 or actual != p['source_sha256'] or actual != c['source_sha256'] or
        p.get('jobs') != 888 or p.get('GPU_authorized_by_plan') is not False or
        c.get('status') != 'passed' or c.get('plan_sha256') != sha(plan_path) or
        c.get('test_returncode') != 0):
        raise ValueError('plan/CPU source mismatch')
    terminal(D/'monitor_execution_cpu001', D/'freeze_execution.py')
    temp, tempcode, tempmon = (R/'e4c_multibudget_references192_worker_preflight_review.json',
        R/'review_e4c_multibudget_references192_worker_preflight.py', R/'monitor_e4c_multibudget_references192_worker_preflight')
    tr = read(temp)
    terminal(tempmon, tempcode)
    if (tr.get('status') != 'passed' or tr.get('independent_review') is not True or
        tr.get('scope') != 'references192_cpu_preflight_only' or tr.get('reviewer_sha256') != sha(tempcode) or
        tr.get('plan_sha256') != sha(plan_path) or tr.get('source_sha256') != actual or tr.get('errors') != []):
        raise ValueError('temporary review mismatch')
    pm = D/'monitor_preflight001'
    st = terminal(pm, D/'launch.py')
    launch, ex, worker, proc = map(read, (P/'launch.json', P/'exit.json', P/'worker_preflight.json', P/'processor_preflight.json'))
    if (st.get('timeout_seconds') != 900 or st.get('grace_seconds') != 90 or '--preflight-only' not in st['command'] or
        st.get('worker_pid') != launch['controller']['pid'] or launch.get('approval_bundle') is not None or
        launch.get('plan_sha256') != sha(plan_path) or launch.get('worker_sha256') != sha(D/'worker.py') or
        launch.get('launcher_sha256') != sha(D/'launch.py') or ex.get('controller') != launch['controller'] or
        ex.get('returncode') != 0 or ex.get('container_exit_code') != 0 or ex.get('timed_out') is not False or
        ex.get('interrupted') is not None or ex.get('container_stopped_verified') is not True or
        ex.get('client_exited_while_container_running') is not False):
        raise ValueError('preflight exit/launch mismatch')
    cmd = launch['command']
    if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or '--rm' in cmd:
        raise ValueError('no-GPU container command mismatch')
    cid = ex['container_identity']['container_id']
    doc = json.loads(subprocess.run(['docker','inspect',cid], check=True, capture_output=True, text=True, timeout=15).stdout)[0]
    state, host = doc['State'], doc['HostConfig']
    if (doc['Id'] != cid or doc['Image'] != ex['container_identity']['image'] or state['Running'] or state['Pid'] != 0 or
        state['ExitCode'] != 0 or state['OOMKilled'] or host['NetworkMode'] != 'none' or host['Devices'] or
        host['DeviceRequests'] or 'CUDA_VISIBLE_DEVICES=' not in doc['Config']['Env'] or
        doc['Config']['Cmd'] != [str(D/'worker.py'),'--out',str(P),'--preflight-only']):
        raise ValueError('actual Docker state or GPU devices mismatch')
    if (worker.get('status') != 'passed' or worker.get('plan_sha256') != sha(plan_path) or
        worker.get('processor_preflight_sha256') != sha(P/'processor_preflight.json') or
        worker.get('cuda_initialized') is not False or worker.get('model_loaded') is not False or
        worker.get('GT_opened') is not False or worker.get('scoring_performed') is not False or
        proc.get('status') != 'passed' or proc.get('inputs_checked') != 222 or len(proc.get('processor_inputs',[])) != 222 or
        proc.get('references_checked') != 888 or proc.get('jobs') != 888 or proc.get('weight_payloads_hashed') != 4 or
        len(proc.get('weight_stats',{})) != 4 or proc.get('cuda_initialized') is not False or
        proc.get('model_loaded') is not False or proc.get('GT_opened') is not False or proc.get('GPU_authorized') is not False):
        raise ValueError('processor or worker preflight mismatch')
    proposal = read(B/'e4_c_multibudget_expansion192/proposal002.json')
    identity = {(x['task'],str(x['qid']),x['group_id']):x for x in proposal['measurement_identities']+proposal['warmup_identities']}
    observed = {(x['task'],str(x['qid']),x['group_id']):x for x in proc['processor_inputs']}
    if (len(identity)!=222 or len(observed)!=222 or set(identity)!=set(observed) or
        proc.get('proposal_sha256')!=sha(B/'e4_c_multibudget_expansion192/proposal002.json') or
        any(observed[k].get('processor')!=x.get('processor_reference') or observed[k].get('image_start')!=x.get('image_start_reference') for k,x in identity.items())):
        raise ValueError('222 processor identities do not match independently reviewed proposal')
    formal, freview = F/'verification001.json', R/'e4c_multibudget_fixed32_result_review.json'
    fm, rm = F/'monitor_verify001', R/'monitor_e4c_multibudget_fixed32_result_002'
    terminal(fm, F/'verify.py')
    terminal(rm, R/'review_e4c_multibudget_fixed32_result.py')
    fixed = read(formal)
    reviewed = read(freview)
    if (fixed.get('status') != 'passed' or fixed.get('pairs') != 12 or fixed.get('generations') != 24 or
        fixed.get('GT_opened') is not False or fixed.get('scoring_performed') is not False or
        fixed.get('timing_claims') is not False or fixed.get('errors') != [] or
        reviewed.get('status') != 'passed' or reviewed.get('independent_review') is not True or
        reviewed.get('pairs') != 12 or reviewed.get('generations') != 24 or reviewed.get('formal_sha256') != sha(formal) or
        reviewed.get('reviewer_sha256') != sha(R/'review_e4c_multibudget_fixed32_result.py') or reviewed.get('errors') != []):
        raise ValueError('fixed32 identity evidence mismatch')
    spec = importlib.util.spec_from_file_location('bulk_review_gate', D/'gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.admit_fixed_identity()  # Also rehash all 24 archived arrays and source snapshots.
    ap = A/'verification_plan001.json'
    acceptance = read(ap)
    terminal(A/'monitor_freeze001', A/'freeze.py')
    if (acceptance.get('status') != 'prepared' or acceptance.get('run_id') != '001' or
        acceptance.get('GPU_authorized') is not False or 'Ran 2 tests' not in acceptance.get('tests','') or
        any(sha(path) != digest for path,digest in acceptance['source_sha256'].items())):
        raise ValueError('future acceptance plan mismatch')
    fields = {'plan_sha256': plan_path, 'execution_cpu_sha256': D/'execution_cpu001.json',
        'execution_cpu_started_sha256': D/'monitor_execution_cpu001/started.json',
        'execution_cpu_terminal_sha256': D/'monitor_execution_cpu001/terminal.json',
        'temp_review_sha256': temp, 'temp_review_started_sha256': tempmon/'started.json',
        'temp_review_terminal_sha256': tempmon/'terminal.json', 'worker_preflight_sha256': P/'worker_preflight.json',
        'processor_preflight_sha256': P/'processor_preflight.json', 'preflight_launch_sha256': P/'launch.json',
        'preflight_exit_sha256': P/'exit.json', 'preflight_started_sha256': pm/'started.json',
        'preflight_terminal_sha256': pm/'terminal.json', 'fixed_formal_sha256': formal,
        'fixed_formal_started_sha256': fm/'started.json', 'fixed_formal_terminal_sha256': fm/'terminal.json',
        'fixed_review_sha256': freview, 'fixed_review_started_sha256': rm/'started.json',
        'fixed_review_terminal_sha256': rm/'terminal.json', 'reviewer_sha256': Path(__file__),
        'acceptance_plan_sha256': ap, 'acceptance_freeze_started_sha256': A/'monitor_freeze001/started.json',
        'acceptance_freeze_terminal_sha256': A/'monitor_freeze001/terminal.json'}
    result = {'status':'passed','independent_review':True,'scope':'references192_gpu_prelaunch',
        'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'source_sha256':actual,'errors':[],
        'container_id':cid,'limitations':['888 fresh fixed32 references; not a latency or quality experiment.',
        'Existing DEV prediction files parsed for input and same-arm identity; no new GT.',
        'Prior 12-pair identity covers three anchors; complete per-image references still require independent acceptance.',
        'A post-run source snapshot cannot exclude transient source changes during execution.']}
    result.update({key:sha(path) for key,path in fields.items()})
    with out.open('x') as f:
        json.dump(result,f,indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
