"""Launch one GPU container; separate logs and exit evidence for each attempt."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path('/media/disk2/YZX/research/EADP_v2')
OLD = R / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(OLD))
from run_panel import container_command, gpu_pids, identity

ap = argparse.ArgumentParser()
ap.add_argument('--resume', action='store_true')
args = ap.parse_args()
before = gpu_pids()
assert before == [], before
plan = json.loads((OLD / 'run_plan.json').read_text())
name = 'eadp-dson-full-perf-%d' % os.getpid()
worker_args = [str(HERE / 'benchmark.py')] + (['--resume'] if args.resume else [])
cmd = container_command(plan, worker_args, True, name)
stamp = '%d_%d' % (int(time.time()), os.getpid())
launch = HERE / ('launch_' + stamp + '.json')
launch.write_text(json.dumps(dict(command=cmd, controller=identity(os.getpid()), started_unix=time.time(),
                                  gpu_compute_pids_before=before), indent=2) + '\n')
print('starting container', name, flush=True)
with (HERE / ('execution_' + stamp + '.log')).open('x') as log:
    try:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        code = result.returncode
    except BaseException as e:
        (HERE / ('launch_failure_' + stamp + '.json')).write_text(json.dumps(dict(error=repr(e)), indent=2) + '\n')
        raise
(HERE / ('process_exit_' + stamp + '.json')).write_text(json.dumps(dict(container_exit_code=code,
    controller=identity(os.getpid()), finished_unix=time.time(), gpu_compute_pids_after=gpu_pids()), indent=2) + '\n')
print('container exit', code, flush=True)
sys.exit(code)
