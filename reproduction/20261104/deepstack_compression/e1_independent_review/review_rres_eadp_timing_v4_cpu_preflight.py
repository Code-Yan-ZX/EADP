"""Independent prelaunch CPU audit. No model, GPU, Docker, GT or timings."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_eadp_timing_pilot_v4';ROOT=B.parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(mon):
 s=read(mon/'started.json');t=read(mon/'terminal.json')
 if Path(s['cwd']).resolve()!=B or s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[]:raise ValueError('supervision '+str(mon))
 return s
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_eadp_timing_v4_cpu_preflight_review.json'
 if a.out.resolve()!=dest or dest.exists() or (D/'preflight001').exists():raise ValueError('wrong destination or preflight already started')
 plan=read(D/'preflight_plan001.json');sources=plan['source_sha256']
 if plan['scope']!='paired_timing_cpu_preflight_only' or plan['GPU_authorized'] is not False or plan['GT_opened'] is not False or plan['existing_predictions_opened'] is not True or plan['inputs']!=62 or plan['references']!=124 or len(sources)!=293 or any(sha(k)!=v for k,v in sources.items()):raise ValueError('plan/source')
 cpu=clean(D/'monitor_cpu004');refs=clean(D/'monitor_references001');sched=clean(D/'monitor_schedule001');freeze=clean(D/'monitor_preflight_freeze001')
 if cpu['command'][1:]!=['-m','unittest','discover','-s','e6/rres_eadp_timing_pilot_v4','-p','test_*.py'] or refs['command'][1]!='-c' or 'reference_inputs.load()' not in refs['command'][2] or (Path(sched['cwd'])/sched['command'][1]).resolve()!=D/'freeze_schedule.py' or (Path(freeze['cwd'])/freeze['command'][1]).resolve()!=D/'freeze_preflight.py':raise ValueError('actual CPU supervision command')
 if 'rows 62 references 124 GT False' not in (D/'monitor_references001/stdout.log').read_text():raise ValueError('reference completion')
 imp=clean(D/'monitor_import_order001')
 if (Path(imp['cwd'])/imp['command'][1]).resolve()!=D/'check_import_order.py' or 'passed actual E4 and paired runner path/object/persist call; 62 inputs/124 same-arm references; no GT/GPU' not in (D/'monitor_import_order001/stdout.log').read_text():raise ValueError('real import order evidence')

 sys.path.insert(0,str(B/'e6/rres_eadp_paired_timing_assessment'));import contract_oracle as oracle
 schedule=read(D/'schedule001.json');oracle.schedule(oracle.flatten_processes(schedule['processes']),schedule['measured_inputs'],schedule['warmup_inputs'])
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PAIRED_TIMING_CANDIDATE='rres_eadp_timing_pilot_v4')
 tests=[]
 for directory in (D,B/'e6/rres_eadp_paired_timing_assessment'):
  r=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(directory),'-p','test_*.py'],cwd=B,env=env,capture_output=True,text=True,timeout=180)
  if r.returncode:raise RuntimeError('independent CPU tests: '+r.stderr)
  tests.append(r.stderr)
 # Actual author command constructor; no subprocess, device or Docker launch.
 loader=load('_independent_author_loader',B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py');runtime=loader.load_author(sources);author=read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 command=runtime.container_command(author,[str(D/'preflight_entry.py'),'--out',str(D/'preflight001')],False,'independent-command-preview')
 if '--gpus' in command or '--device' in command or command.count('--mount')!=12 or 'CUDA_VISIBLE_DEVICES=' not in command or command.count('--rm')!=1:raise ValueError('noGPU command')
 # Reproduce v2's failed class inspection in the actual v3 loader.
 import inspect
 dynamic=load('_independent_module_load',D/'module_load.py');name='_independent_adapter_inspect';adapter=dynamic.load(name,B/'e4_c_budget_pilot_v2/adapter.py')
 if Path(inspect.getfile(adapter.BudgetLightAdapter)).resolve()!=B/'e4_c_budget_pilot_v2/adapter.py':raise ValueError('adapter class identity')
 try:dynamic.load(name,B/'e4_c_budget_pilot_v2/adapter.py')
 except ValueError:pass
 else:raise ValueError('preloaded name not rejected')
 del sys.modules[name]
 gate=load('_independent_pair_gate',D/'preflight_gate.py')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU gate unexpectedly open')
 if any(sha(k)!=v for k,v in sources.items()) or (D/'preflight001').exists():raise ValueError('source drift/preflight race')
 result={'status':'passed','independent_review':True,'preflight_only':True,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'preflight_plan001.json'),'source_sha256':sources,'reviewer_sha256':sha(__file__),'cpu_test_stderr':tests,'noGPU_preview_mounts':12,'limits':['Only no-device import preflight is authorized; model/GPU work still locked.','Real Qwen, outputs and end-to-end endpoints require separate GPU diagnostic and review.','Time observer, execution controller, full formal acceptance and analysis gates are not yet finalized.','Existing accepted DEV predictions opened only as references; no labels opened.']}
 with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
