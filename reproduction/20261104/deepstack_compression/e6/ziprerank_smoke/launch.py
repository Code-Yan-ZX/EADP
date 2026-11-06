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
from worker import preflight, sha, save


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
    inner, outer, grace = (600, 720, 90) if args.stage == 'preflight' else (3600, 3720, 90)
    validate_budget_window(budget, time.time(), inner, outer, grace)
    monitor = HERE / ('monitor_' + args.stage + '001')
    started = json.loads((monitor / 'started.json').read_text())
    if (started['worker_pid'] != os.getpid() or started['timeout_seconds'] != outer
            or started['grace_seconds'] != grace):
        raise ValueError('supervisor identity/config mismatch')
    runtime = load(AUTHOR_PLAN.parent / 'run_panel.py', 'eadp_smoke_run_panel')
    if runtime.gpu_pids():
        raise ValueError('GPU compute process present')
    if shutil.disk_usage(HERE).free < 32 * 1024**3:
        raise ValueError('insufficient experiment disk')
    if args.stage == 'smoke':
        pre = json.loads((HERE / 'preflight001.json').read_text())
        preterminal = json.loads((HERE / 'monitor_preflight001/terminal.json').read_text())
        review = json.loads((HERE / 'independent_prelaunch_review.json').read_text())
        if (pre['status'] != 'passed' or pre['source_plan_sha256'] != sha(plan_path)
                or preterminal['reason'] != 'completed' or preterminal['worker_returncode'] != 0
                or preterminal['remaining_process_group'] != [] or review.get('status') != 'passed'
                or review.get('plan_sha256') != sha(plan_path)
                or review.get('preflight_sha256') != sha(HERE / 'preflight001.json')):
            raise ValueError('independent review or CPU preflight gate failed')
    out = HERE / ('run_' + args.stage + '001')
    out.mkdir(exist_ok=False)
    name = 'eadp-zip-' + args.stage + '-001-' + str(os.getpid())
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
