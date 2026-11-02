"""CPU prepare/check; one launch only after a root-owned GPU allocation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import worker

OUT, NEXT = worker.OUT, worker.NEXT
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
BASE_PLAN_SHA = '7de9048ad555683ca95d1bba27193c30c641b2decd5ae2d4626a359b11e9462d'
read = lambda p: json.loads(Path(p).read_text())
sha = worker.sha
save = lambda p, d: Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
utc = lambda: datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        return dict(pid=pid, start_ticks=int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return sorted({int(line.strip()) for line in text.splitlines() if line.strip()})


def check(plan):
    assert plan['n'] == 5000 and plan['parameters']['visual_token_num'] == 64 and plan['parameters']['budget_nominal'] == 320
    assert sha(NEXT / 'full_default_K128.plan.json') == BASE_PLAN_SHA == plan['base_plan']['sha256']
    full, proof = worker.adapted_module()
    assert proof == plan['worker_adapter']
    full.check_plan(plan)
    assert plan['parameters'] == dict(read(NEXT / 'full_default_K128.plan.json')['parameters'], visual_token_num=64, budget_nominal=320)
    for entry in plan['additional_frozen_sources']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    from freeze_plan import check_prerequisite
    check_prerequisite(plan)
    assert 'torch' not in sys.modules


def prepare():
    from freeze_plan import prepare as freeze
    freeze()


def validate_allocation(note, plan_sha):
    assert note['authorization'] == 'explicit_root_gpu_allocation'
    assert note['run_id'] == 'NeXT_full_K64_question_only_public_comment'
    assert note['plan_sha256'] == plan_sha and note['GPU_slot'] in ('A', 'B')
    assert note['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    current = gpu_pids()
    assert current == sorted(note['allowed_other_gpu_pids']) and len(current) < 2
    assert sorted(p['pid'] for p in note['allowed_other_gpu_processes']) == current
    assert all(identity(p['pid']) == p for p in note['allowed_other_gpu_processes'])
    return current


def run(note_path):
    plan = read(worker.PLAN)
    check(plan)
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve()
    assert sha(worker.PLAN) == read(OUT / 'READY.json')['plan_sha256']
    note = read(note_path)
    initial = validate_allocation(note, sha(worker.PLAN))
    assert not worker.RUN.exists()
    controller = identity(os.getpid())
    with (OUT / 'launch.claim.json').open('x') as stream:
        json.dump(dict(controller=controller, plan_sha256=sha(worker.PLAN), allocation_note_sha256=sha(note_path),
                       launch_attempt=1, automatic_retry=False), stream, indent=2)
    save(OUT / 'allocation.json', dict(note=note, note_sha256=sha(note_path), actual_GPU_pids_before_launch=initial))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(NEXT) + ':' + str(full_author_path()), PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith(('USE_LLAVA_ARCH_', 'ANCHORZIP_', 'EADP_', 'ENTROPY_')):
            env.pop(key)
    state = dict(status='starting', controller=controller, started_utc=utc(), plan_sha256=sha(worker.PLAN), automatic_retry=False)
    try:
        with (OUT / 'generation.log').open('x') as stream:
            process = subprocess.Popen([PYTHON, str(OUT / 'worker.py'), str(note_path.resolve())], env=env, stdout=stream, stderr=subprocess.STDOUT)
            state.update(status='generating', native_worker=identity(process.pid))
            assert state['native_worker'] is not None
            save(OUT / 'controller_state.json', state)
            code = process.wait()
        state.update(native_exit_code=code, status='native_exited', native_finished_utc=utc())
        save(OUT / 'controller_state.json', state)
        assert code == 0, 'Native failed; preserve partial outputs and do not retry'
        cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
        with (OUT / 'verification.log').open('x') as stream:
            verified = subprocess.run([PYTHON, str(OUT / 'verify.py')], env=cpu_env, stdout=stream, stderr=subprocess.STDOUT)
        assert verified.returncode == 0
        check(plan)
        current = identity(process.pid)
        assert current != state['native_worker'] and process.pid not in gpu_pids()
        proof = read(worker.RUN / 'independent_verification.json')
        assert proof['success'] and proof['n_rows'] == proof['token_decoding_checks'] == proof['input_budget_observation_checks'] == 5000
        assert proof['actual_guidance_checks'] == proof['per_question_score_checks'] == 5000
        release = dict(success=True, n_rows=5000, run_id=plan['run_id'],
            plan_sha256=sha(worker.PLAN), official_score_sha256=sha(worker.RUN / 'official.score.json'),
            independent_verification_sha256=sha(worker.RUN / 'independent_verification.json'),
            actual_worker_exit=dict(original=state['native_worker'], original_exited=True, gpu_pid_absent=True, current_if_reused=current),
            controller_identity=controller, controller_exit_must_be_confirmed_after_return=True,
            official_value_percent=proof['official_accuracy_percent'], independent_value_percent=proof['independent_accuracy_percent'],
            actual_guidance_checks=5000, input_hypothesis_only=True, paper_argv_confirmed=False,
            score_direction_gate=False, automatic_retry=False)
        save(worker.RUN / 'release.json', release)
        state.update(status='complete', finished_utc=utc(), release_sha256=sha(worker.RUN / 'release.json'), independent_verification_sha256=sha(worker.RUN / 'independent_verification.json'))
        save(OUT / 'controller_state.json', state)
        print(json.dumps(release))
    except BaseException as error:
        state.update(status='terminal_failure_no_retry', error=repr(error), finished_utc=utc())
        save(OUT / 'failure.json', state)
        save(OUT / 'controller_state.json', state)
        raise


def full_author_path():
    return NEXT.parents[2] / '_reproduction/author_eadp/LLaVA'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['prepare', 'check', 'run'])
    parser.add_argument('--allocation-note', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        check(read(worker.PLAN))
        print(json.dumps(dict(CPU_validation=True, model_loaded=False, GPU_started=False, would_launch=False)))
    else:
        assert args.allocation_note is not None
        run(args.allocation_note)
