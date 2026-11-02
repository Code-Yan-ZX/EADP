"""Prepare, check, or explicitly start one CPU-only terminal Qwen audit.

Wait for the whole public DocVQA->AI2D container, never an individual task PID.
No generation, GPU model, API judge, source change, or automatic audit retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import audit_qwen as audit

HERE, QWEN = audit.HERE, audit.QWEN
RUN = QWEN / 'public_fa2_full_run02'
PLAN = QWEN / 'public_fa2_full_run02_plan.json'
PREPARATION = HERE / 'terminal_audit_preparation.json'
PREFIX = 'public_fa2_full_run02.terminal_audit'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
EXPECTED_PLAN = 'd1afef9e1f470d432105cb8669e25f54b9906d85b54f7685689a84b539feb467'
EXPECTED_POLICY = 'fe0bad1314831c344aac49617e6da0e338e9a4a6da5254a3de5a0cdea46764f0'
CONTROLLER = {'pid': 2364766, 'startticks': 1331402021}
CONTAINER = '9c25877ad50b305cac131e1c023a19678601501fd3c9b9ad43364d00c2c9a769'
IMAGE = 'sha256:91d767ae00637d58384a6ca70cbfe7b8c9bfa9f8920bbe4ddaaa9af178100faf'
INIT = {'pid': 2364864, 'startticks': 1331402049}
TERMINAL_STATUS = 'generation_launcher_exited_validation_pending'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, record):
    target = HERE / (PREFIX + '.' + name + '.json')
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def identity(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        return {'pid': pid, 'startticks': int(text[text.rindex(')') + 2:].split()[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def commandline(pid):
    return [v.decode() for v in Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0') if v]


def in_container(cgroup):
    parts = [line.split(':', 2)[2].split('/') for line in cgroup.splitlines() if line.strip()]
    return bool(parts) and all(CONTAINER in p or 'docker-' + CONTAINER + '.scope' in p for p in parts)


def task_for_command(command, plan):
    python = plan['command'][plan['command'].index('--entrypoint') + 1]
    entry = next(p for p in plan['source_sha256'] if p.endswith('/Qwen_vl/VLMEvalKit/run.py'))
    for name in audit.COUNTS:
        if command == [python, entry, '--model', plan['model_name'], '--data', name,
                       '--work-dir', str(Path(plan['environment']['OUTPUT_ROOT']) / 'eadp'), '--verbose']:
            return name
    raise AssertionError('Unregistered native argv inside the fixed Qwen container')


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader'], text=True)
    return text, [int(s.split(',', 1)[0]) for s in text.splitlines() if s.strip()]


def observe_native(plan):
    snapshot, pids = gpu_snapshot()
    rows = []
    for pid in pids:
        before = identity(pid)
        try:
            cgroup = Path('/proc', str(pid), 'cgroup').read_text()
            if not in_container(cgroup):
                continue
            command = commandline(pid)
            task = task_for_command(command, plan)
            if before is None or identity(pid) != before:
                continue  # An ordinary task exit may race a read-only observation.
            rows.append(dict(identity=before, task=task, cmdline=command, cgroup=cgroup, observed_utc=utc()))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return snapshot, rows


def inspect_container():
    template = '{"id":{{json .Id}},"image":{{json .Image}},"running":{{json .State.Running}},"pid":{{json .State.Pid}},"exit_code":{{json .State.ExitCode}}}'
    result = subprocess.run(['docker', 'inspect', '--format', template, CONTAINER], capture_output=True, text=True)
    if result.returncode:
        if 'No such object' in result.stderr or 'No such container' in result.stderr:
            return {'exists': False, 'container_id': CONTAINER, 'stderr': result.stderr}
        raise RuntimeError('Cannot establish container state: ' + result.stderr)
    record = json.loads(result.stdout)
    assert record['id'] == CONTAINER and record['image'] == IMAGE
    return dict(exists=True, **record)


def frozen_gate():
    prepared = read(PREPARATION)
    assert audit.sha(PLAN) == EXPECTED_PLAN
    assert audit.sha(HERE / 'empty_answer_policy.json') == EXPECTED_POLICY
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prepared['boot_id']
    for path, expected in prepared['frozen_files'].items():
        assert audit.sha(path) == expected, ('Frozen audit/source changed', path)
    assert 'torch' not in sys.modules
    return prepared


def terminal_gate(prepared, workers):
    """Only small control-plane records; never scores a live prediction prefix."""
    controller = identity(CONTROLLER['pid'])
    try:
        state = read(RUN / 'state.json')
    except json.JSONDecodeError:
        if controller == CONTROLLER:
            return False, {'reason': 'waiting_root_controller_record_write'}
        state = read(RUN / 'state.json')  # A writer that exited cannot finish a corrupt record.
    assert state['controller_pid'] == CONTROLLER['pid'] and state['plan_sha256'] == EXPECTED_PLAN
    if controller == CONTROLLER:
        return False, {'reason': 'waiting_whole_run_packaged_controller_exit', 'launcher_state': state['status']}
    # Re-read once after observed exit to cover the normal final-record transition.
    state = read(RUN / 'state.json')
    assert state['status'] == TERMINAL_STATUS, 'Root controller exited without its final launcher record'
    assert isinstance(state['launcher_exit_code'], int)
    assert state['docker_client_pid'] == prepared['docker_client']['pid']
    client = identity(prepared['docker_client']['pid'])
    assert client != prepared['docker_client'], 'Original docker client is still alive'
    container = inspect_container()
    assert not container.get('running'), 'Root controller ended while its Docker container is still running'
    init = identity(INIT['pid'])
    assert init != INIT, 'Original container init is still alive'
    snapshot, pids = gpu_snapshot()
    exits = []
    for worker in workers:
        original = worker['identity']; current = identity(original['pid'])
        assert current != original, 'An observed Qwen native worker is still alive'
        assert original['pid'] not in pids, 'An observed Qwen worker remains on GPU'
        exits.append(dict(task=worker['task'], original_identity=original, original_exited=True,
                          current_identity_if_reused=current, GPU_pid_absent=True))
    # Also reject a later unrecorded GPU worker from this same container.
    for pid in pids:
        try:
            assert not in_container(Path('/proc', str(pid), 'cgroup').read_text()), 'Qwen container still owns a GPU process'
        except (FileNotFoundError, ProcessLookupError):
            continue
    return True, dict(reason='whole_public_sequence_and_actual_processes_terminated', checked_utc=utc(),
        original_controller=CONTROLLER, controller_original_exited=True, current_controller_identity_if_reused=controller,
        original_docker_client=prepared['docker_client'], docker_client_original_exited=True, current_client_identity_if_reused=client,
        container=container, original_container_init=INIT, container_init_original_exited=True, current_init_identity_if_reused=init,
        worker_exits=exits, nvidia_smi=snapshot, launcher_state=state,
        launcher_state_sha256=audit.sha(RUN / 'state.json'))


def prepare():
    assert not (HERE / (PREFIX + '.claim.json')).exists()
    assert audit.sha(PLAN) == EXPECTED_PLAN and audit.sha(HERE / 'empty_answer_policy.json') == EXPECTED_POLICY
    plan = read(PLAN); audit.source_checks(plan)
    state = read(RUN / 'state.json'); original = read(RUN / 'process_identity.json')
    assert identity(CONTROLLER['pid']) == CONTROLLER
    assert commandline(CONTROLLER['pid'])[1:] == ['reproduction/20261028/qwen/run_packaged.py', 'run', '--gpu-authorized']
    client = identity(state['docker_client_pid']); assert client
    assert commandline(client['pid']) == plan['command']
    assert original['container_id'] == CONTAINER and original['image'] == IMAGE
    assert original['host_init_pid'] == INIT['pid'] and int(original['host_start_ticks']) == INIT['startticks']
    assert identity(INIT['pid']) == INIT
    container = inspect_container(); assert container['running'] and container['pid'] == INIT['pid']
    _, workers = observe_native(plan)
    files = [Path(__file__).resolve(), HERE / 'test_terminal_audit.py', HERE / 'audit_qwen.py', HERE / 'test_audit_qwen.py',
             HERE / 'empty_answer_policy.json', HERE / (PLAN.stem + '.preparation_audit.json'),
             HERE / 'current_cached_images.json', RUN / 'process_identity.json',
             RUN / 'docvqa_gpu_worker_identity.json', PLAN]
    files += [HERE / (name + suffix) for name in audit.COUNTS
              for suffix in ('.canonical.jsonl', '.canonical_manifest.json')]
    files += [Path(p) for group in ('source_sha256', 'frozen_evidence') for p in plan[group]]
    payloads = []
    for row in read(QWEN / 'qwen_weight_identity.json')['files']:
        assert row['matches'] and row['sha256'] == row['expected_lfs_sha256']
        payloads.append(dict(path=row['path'], expected_sha256=row['sha256'], kind='official_Qwen_weight', stat=audit.identity(row['path'])))
    record = dict(status='prepared_not_started', prepared_utc=utc(), CPU_only=True,
        GPU_loaded=False, watcher_started=False, root_review_required_before_run=True,
        generation_plan_sha256=EXPECTED_PLAN, empty_policy_sha256=EXPECTED_POLICY,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        root_controller=CONTROLLER, root_controller_cmdline=commandline(CONTROLLER['pid']),
        docker_client=client, container_id=CONTAINER, container_image=IMAGE, container_init=INIT,
        initially_observed_native_workers=workers,
        frozen_files={str(p): audit.sha(p) for p in files}, prior_SHA_and_current_model_stat_guards=payloads,
        expected_counts=audit.COUNTS,
        audit_completion_meaning='Only the CPU audit has terminated. Never a claim of paper reproduction or original No-empty gate success.',
        terminal_policy='One audit attempt; save every failure and native exit/API evidence. No generation, API call, score-direction gate, or retry.')
    PREPARATION.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record


def raw_coverage(plan):
    diagnostics = {}; errors = []
    for name, expected in audit.COUNTS.items():
        candidates = {p.resolve() for extension in ('xlsx', 'tsv', 'json')
                      for p in RUN.rglob(plan['model_name'] + '_' + name + '.' + extension)}
        item = dict(expected_count=expected, raw_candidates=[str(p) for p in sorted(candidates)],
                    coverage_complete=False, generation_valid=None, empty_count=None,
                    planned_no_empty_gate_passed=None)
        try:
            assert len(candidates) == 1, 'Missing or ambiguous raw prediction table'
            path = candidates.pop(); rows = audit.read_table(path)
            item.update(observed_count=len(rows), raw_path=str(path), raw_sha256=audit.sha(path))
            canonical, _ = audit.canonical_dataset(name, plan['datasets'][name])
            item.update(audit.validate_predictions(canonical, rows, expected))
        except Exception as error:
            item['error'] = repr(error); errors.append(dict(stage='raw_coverage', dataset=name, error=repr(error)))
        diagnostics[name] = item
    return diagnostics, errors


def audit_terminal(prepared, gate, workers):
    plan = read(PLAN); errors = []
    coverage, raw_errors = raw_coverage(plan); errors.extend(raw_errors)
    guards = []
    try:
        frozen_gate(); audit.source_checks(plan)
    except Exception as error:
        errors.append(dict(stage='frozen_source_or_payload', error=repr(error), traceback=traceback.format_exc()))
    for record in prepared['prior_SHA_and_current_model_stat_guards']:
        try:
            assert audit.identity(record['path']) == record['stat'], ('Payload stat changed', record['path'])
            guards.append(dict(path=record['path'], prior_verified_sha256=record['expected_sha256'],
                               current_stat_matches=True, kind=record['kind']))
        except Exception as error:
            errors.append(dict(stage='frozen_source_or_payload', path=record['path'], error=repr(error), traceback=traceback.format_exc()))
    save('model_stat_guards', guards)
    stdout = HERE / (PREFIX + '.verify.stdout.log'); stderr = HERE / (PREFIX + '.verify.stderr.log')
    result = None; returncode = None
    if not any(error['stage'] == 'frozen_source_or_payload' for error in errors):
        env = {'PATH': os.environ.get('PATH', ''), 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        with stdout.open('x') as out, stderr.open('x') as err:
            completed = subprocess.run([PYTHON, str(HERE / 'audit_qwen.py'), 'verify', '--plan', str(PLAN),
                                        '--run-dir', str(RUN)], cwd=HERE, env=env, stdout=out, stderr=err)
        returncode = completed.returncode
        if returncode == 0:
            result = read(HERE / (RUN.name + '.verification.json'))
            assert result['plan_sha256'] == EXPECTED_PLAN and result['auditor']['sha256'] == prepared['frozen_files'][str(HERE / 'audit_qwen.py')]
        else:
            errors.append(dict(stage='existing_CPU_verifier', returncode=returncode,
                               stdout=str(stdout), stderr=str(stderr)))
    complete = all(item['coverage_complete'] for item in coverage.values())
    valid = all(item['generation_valid'] for item in coverage.values()) if complete else None
    empty = sum(item['empty_count'] for item in coverage.values()) if complete else None
    no_empty = all(item['planned_no_empty_gate_passed'] for item in coverage.values()) if complete else None
    tasks = {worker['task'] for worker in workers}
    if tasks != set(audit.COUNTS):
        errors.append(dict(stage='native_stage_observation', error='Did not observe both registered task workers', observed_tasks=sorted(tasks)))
    if not valid and complete:
        errors.append(dict(stage='generation_validity', error='Failure placeholders remain; full-denominator scores are diagnostic'))
    native_logs = [dict(path=str(p), sha256=audit.sha(p), bytes=p.stat().st_size) for p in sorted(RUN.rglob('*.log'))]
    native_error_lines = []
    for path in sorted(RUN.rglob('*.log')):
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if any(word in line for word in ['Traceback', 'ERROR', 'Exception', 'Error', 'API', 'exact matching', 'fallback']):
                native_error_lines.append(dict(path=str(path), line_number=number, text=line))
    report = dict(audit_terminated=True, completed_utc=utc(), CPU_only=True, GPU_or_API_used=False,
        generation_or_source_modified=False, generation_retry_performed=False,
        completion_is_not_paper_reproduction_success=True, paper_reproduction_success_claimed=False,
        generation_plan_sha256=EXPECTED_PLAN, audit_preparation_sha256=audit.sha(PREPARATION),
        empty_answer_policy_sha256=EXPECTED_POLICY, terminal_process_proof=gate,
        all_native_stages_observed=tasks == set(audit.COUNTS), observed_native_workers=workers,
        raw_dataset_diagnostics=coverage, coverage_complete=complete, generation_valid=valid,
        empty_count=empty, planned_no_empty_gate_passed=no_empty,
        existing_CPU_verifier_exit_code=returncode, existing_CPU_verification=result,
        full_prediction_audit_passed=not errors and bool(result and result['complete']),
        public_launcher_exit_code=gate['launcher_state']['launcher_exit_code'],
        public_launcher_exited_successfully=gate['launcher_state']['launcher_exit_code'] == 0,
        author_score_artifacts_complete=result['author_score_artifacts_complete'] if result else None,
        native_logs=native_logs, native_error_lines=native_error_lines, errors=errors,
        limits='Native launcher/API failures and missing score artifacts remain separate from complete raw CPU scoring. Original No-empty gate is reported without changing it; no score-direction acceptance or paper-argv claim.')
    save('report', report)
    return report


def main(action, root_review_note):
    if action == 'prepare':
        record = prepare()
        print(json.dumps({k: record[k] for k in ('status', 'CPU_only', 'watcher_started', 'expected_counts')}))
        return
    prepared = frozen_gate(); workers = prepared['initially_observed_native_workers'][:]
    if action == 'check':
        ready, gate = terminal_gate(prepared, workers)
        print(json.dumps(dict(CPU_only=True, would_start_watcher=False, would_launch_GPU=False, terminal_ready=ready, gate=gate)))
        return
    assert root_review_note, 'Root must review and explicitly start this CPU watcher'
    descriptor = os.open(HERE / (PREFIX + '.claim.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(dict(root_review_note=root_review_note, claimed_utc=utc(), identity=identity(os.getpid()),
                       preparation_sha256=audit.sha(PREPARATION), automatic_retry=False), stream, indent=2)
    state = dict(status='waiting_whole_public_sequence_terminal', identity=identity(os.getpid()),
                 CPU_only=True, GPU_loaded=False, root_review_note=root_review_note)
    save('state', state); previous = None
    while True:
        _, observed = observe_native(read(PLAN))
        known = {(w['identity']['pid'], w['identity']['startticks']) for w in workers}
        workers += [w for w in observed if (w['identity']['pid'], w['identity']['startticks']) not in known]
        save('observed_workers', workers)
        ready, gate = terminal_gate(prepared, workers)
        if gate['reason'] != previous:
            state.update(gate=gate, observed_utc=utc()); save('state', state); previous = gate['reason']
        if ready:
            break
        time.sleep(10)
    state.update(status='auditing_complete_terminal_outputs', terminal_gate=gate); save('state', state)
    report = audit_terminal(prepared, gate, workers)
    # Complete means the audit ended, including a preserved negative diagnostic.
    state.update(status='complete', audit_terminated=True, full_prediction_audit_passed=report['full_prediction_audit_passed'],
                 completed_utc=utc(), report_sha256=audit.sha(HERE / (PREFIX + '.report.json')),
                 paper_reproduction_success_claimed=False)
    save('state', state)
    assert 'torch' not in sys.modules
    print(json.dumps({k: state[k] for k in ('status', 'audit_terminated', 'full_prediction_audit_passed', 'paper_reproduction_success_claimed')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'run'), default='check', nargs='?')
    parser.add_argument('--root-review-note')
    args = parser.parse_args()
    try:
        main(args.action, args.root_review_note)
    except BaseException as error:
        if args.action == 'run' and (HERE / (PREFIX + '.claim.json')).exists():
            save('failure', dict(status='terminal_failure_no_retry', utc=utc(), error=repr(error),
                                 traceback=traceback.format_exc(), GPU_loaded=False, generation_retry_performed=False,
                                 audit_terminated=True, full_prediction_audit_passed=False,
                                 coverage_complete=None, generation_valid=None, empty_count=None,
                                 planned_no_empty_gate_passed=None, paper_reproduction_success_claimed=False))
        raise
