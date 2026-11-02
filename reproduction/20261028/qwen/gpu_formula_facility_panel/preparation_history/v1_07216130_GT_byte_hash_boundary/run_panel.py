"""CPU default; one future Docker run requires a separate exact root allocation."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose import PLAN, sha, validate, save


def gpu_pids():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return sorted({int(line.strip()) for line in value.splitlines() if line.strip()})


def identity(pid):
    try:
        ticks = int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None
    return dict(pid=pid, start_ticks=ticks, session_id=os.getsid(pid),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def command(plan, note_path):
    path = Path(plan['container_runtime_source'])
    spec = importlib.util.spec_from_file_location('unchanged_Qwen_container_runtime', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.IMAGE == plan['container_image']
    result = module.command([str(HERE / 'diagnose.py'), '--execute', '--allocation-note', str(note_path)],
        gpu=True, name=plan['container_name'], extra_env={'PYTHONPATH': plan['import_overlay']})
    if 'planned_container_command' in plan:
        normalized = list(result); normalized[-1] = 'FUTURE_ROOT_REVIEWED_ALLOCATION.json'
        assert normalized == plan['planned_container_command']
    return result


def run(plan, note_path):
    validate(plan)
    assert sha(PLAN) == json.loads((HERE / 'READY.json').read_text())['run_plan_sha256']
    note = json.loads(note_path.read_text())
    assert note['authorization'] == 'explicit_root_gpu_allocation' and note['run_id'] == plan['run_id']
    assert note['run_plan_sha256'] == sha(PLAN) and note['GPU_slot'] in ('A', 'B')
    assert gpu_pids() == sorted(note['allowed_other_gpu_pids']) and len(gpu_pids()) < 2
    allowed = note['allowed_other_gpu_processes']
    assert sorted(item['pid'] for item in allowed) == sorted(note['allowed_other_gpu_pids'])
    for item in allowed:
        actual = identity(item['pid']); assert actual is not None
        assert {k: actual[k] for k in ('pid', 'start_ticks', 'boot_id')} == item
    assert not (HERE / 'execution').exists() and not (HERE / 'state.json').exists()
    assert identity(os.getpid()) is not None
    actual_command = command(plan, note_path)
    with (HERE / 'launch.claim.json').open('x') as stream:
        json.dump(dict(controller=identity(os.getpid()), allocation=note, automatic_retry=False), stream, indent=2)
    state = dict(status='starting', controller=identity(os.getpid()), allocation=note,
                 allocation_note_sha256=sha(note_path), run_plan_sha256=sha(PLAN), command=actual_command,
                 started_utc=datetime.now(timezone.utc).isoformat())
    save(HERE / 'state.json', state)
    try:
        with (HERE / 'execution.log').open('x') as log:
            worker = subprocess.Popen(state['command'], stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running', docker_client=identity(worker.pid)); save(HERE / 'state.json', state)
            while worker.poll() is None:
                if 'container_identity' not in state:
                    probe = subprocess.run(['docker', 'inspect', plan['container_name']], capture_output=True, text=True)
                    if probe.returncode == 0:
                        item = json.loads(probe.stdout)[0]; pid = item['State']['Pid']
                        if pid and identity(pid):
                            assert item['Image'] == plan['container_image']
                            cgroup = Path('/proc', str(pid), 'cgroup').read_text()
                            assert item['Id'] in cgroup
                            state['container_identity'] = dict(container_id=item['Id'], image=item['Image'],
                                host_init=identity(pid), host_cgroup=cgroup,
                                original_host_cmdline=Path('/proc', str(pid), 'cmdline').read_bytes().decode().replace('\x00', ' '),
                                recorded_utc=datetime.now(timezone.utc).isoformat())
                            save(HERE / 'state.json', state)
                time.sleep(1)
            result = worker.wait()
        state.update(status='container_exited_validation_pending', exit_code=result)
        save(HERE / 'state.json', state)
        assert result == 0
        validate(plan)
        summary = json.loads((HERE / 'execution/summary.json').read_text())
        assert summary['status'] in ('complete', 'complete_with_explicit_unmeasured')
        assert summary['run_plan_sha256'] == sha(PLAN) and summary['n_questions'] == summary['native_anchors'] == 16
        assert summary['native_get_calls'] == 16 and summary['identity_controls'] == 16
        assert summary['measured_cases'] + summary['unmeasured_cases'] == 64
        assert summary['generated_answers'] == 0 and not summary['benchmark_scores_computed'] and not summary['GT_read']
        assert sha(HERE / 'execution/per_question.jsonl') == summary['per_question_sha256']
        assert sha(HERE / 'execution/runtime.json') == summary['runtime_sha256']
        records = [json.loads(line) for line in (HERE / 'execution/per_question.jsonl').read_text().splitlines()]
        assert len(records) == 16 and [r['position'] for r in records] == [r['position'] for r in plan['panel']]
        for record in records: assert sha(record['cache']['path']) == record['cache']['sha256']
        recorded = state['container_identity']; original = recorded['host_init']
        now = identity(original['pid'])
        assert now is None or now['start_ticks'] != original['start_ticks']
        assert original['pid'] not in gpu_pids()
        probe = subprocess.run(['docker', 'inspect', recorded['container_id']], capture_output=True, text=True)
        assert probe.returncode != 0 or not json.loads(probe.stdout)[0]['State']['Running']
        state.update(status='complete', summary_sha256=sha(HERE / 'execution/summary.json'),
                     original_container_init_exited=True, GPU_pid_absent=True,
                     controller_exit_must_be_confirmed_after_run_returns=True,
                     finished_utc=datetime.now(timezone.utc).isoformat())
        save(HERE / 'state.json', state)
    except BaseException as error:
        save(HERE / 'failure.json', dict(status='terminal_failure_no_retry', error=repr(error), automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run'], nargs='?', default='check')
    parser.add_argument('--allocation-note', type=Path)
    args = parser.parse_args(); plan = json.loads(PLAN.read_text()); validate(plan)
    if args.action == 'check':
        assert 'torch' not in sys.modules
        print(json.dumps(dict(status='prepared_not_started', torch_imported=False, GPU_started=False,
                              container_started=False, run_plan_sha256=sha(PLAN))))
        return
    assert args.allocation_note is not None
    run(plan, args.allocation_note.resolve())


if __name__ == '__main__': main()
