"""Container launch with bounded cleanup, GPU exclusivity, and proof gates."""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROOT = BASE.parents[2]
AUTHOR_PLAN = ROOT / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path.insert(0, str(BASE / 'e4_c_latency'))
from timing_launch import run_bounded, validate_budget_window
from worker import validate_plan as preflight, sha, save


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'smoke'], required=True)
    args = parser.parse_args()
    plan_path = HERE / 'plan001.json'
    plan = json.loads(plan_path.read_text())
    preflight(plan)
    if sha(BASE / 'initial_plan.json') != plan['budget_source_sha256']:
        raise ValueError('user deadline source changed')
    initial = json.loads((BASE / 'initial_plan.json').read_text())
    budget = {'budget_source': str(BASE / 'initial_plan.json'),
              'budget_source_sha256': plan['budget_source_sha256'],
              'budget_start_unix': initial['budget_start_unix'], 'budget_end_unix': initial['budget_end_unix']}
    inner, outer, grace = (600, 720, 90) if args.stage == 'preflight' else (1800, 1920, 90)
    validate_budget_window(budget, time.time(), inner, outer, grace)
    monitor = HERE / ('monitor_' + args.stage + '001')
    started = json.loads((monitor / 'started.json').read_text())
    if (started['worker_pid'] != os.getpid() or started['timeout_seconds'] != outer
            or started['grace_seconds'] != grace):
        raise ValueError('supervisor identity/config mismatch')
    if args.stage == 'smoke' and (started.get('stale_seconds') != 420 or started.get('progress_paths') != [
            str(HERE / 'run_smoke001/observations/progress.jsonl')]):
        raise ValueError('stale progress supervision mismatch')
    runtime = load(AUTHOR_PLAN.parent / 'run_panel.py', 'eadp_smoke_run_panel')
    if args.stage == 'smoke' and runtime.gpu_pids():
        raise ValueError('GPU compute process present')
    if shutil.disk_usage(HERE).free < 32 * 1024**3:
        raise ValueError('insufficient experiment disk')
    if args.stage == 'smoke':
        pre = json.loads((HERE / 'preflight001.json').read_text())
        preterminal = json.loads((HERE / 'monitor_preflight001/terminal.json').read_text())
        prestarted = json.loads((HERE / 'monitor_preflight001/started.json').read_text())
        prelaunch = json.loads((HERE / 'run_preflight001/launch.json').read_text())
        preexit = json.loads((HERE / 'run_preflight001/exit.json').read_text())
        review = json.loads((HERE / 'independent_prelaunch_review.json').read_text())
        review_terminal_path = Path(review['review_terminal_path']).resolve()
        reviewer_path = Path(review['reviewer_path']).resolve()
        review_dir = (BASE / 'e1_independent_review').resolve()
        if review_dir not in review_terminal_path.parents or review_dir not in reviewer_path.parents:
            raise ValueError('review provenance outside independent audit directory')
        review_terminal = json.loads(review_terminal_path.read_text())
        review_started_path = review_terminal_path.parent / 'started.json'
        review_started = json.loads(review_started_path.read_text())
        review_command = review_started.get('command', [])
        review_cwd = Path(review_started['cwd'])
        resolved_command = [str((review_cwd / x).resolve()) if not Path(x).is_absolute() else x for x in review_command]
        precommand = prestarted.get('command', [])
        prelaunch_path = (Path(prestarted['cwd']) / precommand[-3]).resolve() if len(precommand) >= 3 else None
        image = json.loads(AUTHOR_PLAN.read_text())['container_image']
        if (pre['status'] != 'passed' or pre['source_plan_sha256'] != sha(plan_path)
                or pre.get('cuda_initialized') is not False or pre.get('case_count') != 1
                or preterminal['reason'] != 'completed' or preterminal['worker_returncode'] != 0
                or preterminal['remaining_process_group'] != []
                or preterminal.get('worker_pid') != prestarted.get('worker_pid')
                or prestarted.get('worker_pid') != prelaunch.get('controller', {}).get('pid')
                or prestarted.get('worker_pid') != preexit.get('controller', {}).get('pid')
                or prelaunch_path != HERE / 'launch.py' or precommand[-2:] != ['--stage', 'preflight']
                or prelaunch.get('plan_sha256') != sha(plan_path)
                or preexit.get('container_stopped_verified') is not True or preexit.get('container_exit_code') != 0
                or preexit.get('returncode') != 0 or preexit.get('failure_type') is not None
                or preexit.get('container_identity', {}).get('image') != image
                or preexit.get('container_identity', {}).get('container_id') != preexit.get('last_inspect', {}).get('id')
                or preexit.get('last_inspect', {}).get('command') != [str(HERE / 'worker.py'), '--preflight-only', '--out', str(HERE / 'preflight001.json')]
                or preexit.get('last_inspect', {}).get('host_devices') != []
                or preexit.get('last_inspect', {}).get('device_requests') is not None
                or review.get('status') != 'passed' or review.get('independent_review') is not True
                or review.get('plan_sha256') != sha(plan_path)
                or review.get('preflight_sha256') != sha(HERE / 'preflight001.json')
                or review.get('preflight_started_sha256') != sha(HERE / 'monitor_preflight001/started.json')
                or review.get('preflight_terminal_sha256') != sha(HERE / 'monitor_preflight001/terminal.json')
                or review.get('preflight_exit_sha256') != sha(HERE / 'run_preflight001/exit.json')
                or review.get('reviewer_sha256') != sha(reviewer_path)
                or review.get('review_started_sha256') != sha(review_started_path)
                or str(reviewer_path) not in resolved_command or '--out' not in review_command
                or review_command.index('--out') + 1 >= len(resolved_command)
                or resolved_command[review_command.index('--out') + 1] != str(HERE / 'independent_prelaunch_review.json')
                or review_terminal.get('reason') != 'completed' or review_terminal.get('worker_returncode') != 0
                or review_terminal.get('remaining_process_group') != []
                or review_terminal.get('worker_pid') != review_started.get('worker_pid')):
            raise ValueError('independent review or CPU preflight gate failed')
    out = HERE / ('run_' + args.stage + '001')
    out.mkdir(exist_ok=False)
    name = 'eadp-authorgraph-' + args.stage + '-001-' + str(os.getpid())
    worker_args = [str(HERE / 'worker.py')]
    if args.stage == 'preflight':
        worker_args += ['--preflight-only', '--out', str(HERE / 'preflight001.json')]
    else:
        worker_args += ['--out', str(out / 'observations')]
    command = runtime.container_command(json.loads(AUTHOR_PLAN.read_text()), worker_args, args.stage == 'smoke', name)
    if command.count('--rm') != 1:
        raise ValueError('unexpected container retention')
    command.remove('--rm')
    save(out / 'launch.json', {'plan_sha256': sha(plan_path), 'stage': args.stage,
                              'container_name': name, 'command': command,
                              'controller': runtime.identity(os.getpid()), 'started_unix': time.time()})
    return run_bounded(command, name, inner, runtime.gpu_pids, runtime.identity, out)


if __name__ == '__main__':
    sys.exit(main())
