"""Run one retained, bounded, no-device container for weight stat mapping."""
import importlib.util, json, os, sys
from pathlib import Path
D=Path(__file__).resolve().parent; B=D.parents[1]; ROOT=B.parents[2]
sys.path.insert(0,str(B/'e4_c_latency'))
import timing_launch
author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
spec=importlib.util.spec_from_file_location('_stat_author_runtime',author.parent/'run_panel.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
out=D/'run001';out.mkdir(exist_ok=False)
name='eadp-wavefront-weight-stat-cpu-'+str(os.getpid())
cmd=runtime.container_command(json.loads(author.read_text()),[str(D/'worker.py')],False,name)
assert cmd.count('--rm')==1 and '--gpus' not in cmd and '--device' not in cmd
cmd.remove('--rm')
with (out/'launch.json').open('x') as f:
 json.dump({'name':name,'command':cmd,'controller':runtime.identity(os.getpid())},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
raise SystemExit(timing_launch.run_bounded(cmd,name,180,None,runtime.identity,out,require_gpu_idle=False))
