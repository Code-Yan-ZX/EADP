"""Source, identity and independent prelaunch gates; no GPU calls."""
import importlib.util,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review'
spec=importlib.util.spec_from_file_location('_budget_ttft_admission_identity',HERE/'identity_gate.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
def read(p):return json.loads(Path(p).read_text())
def plan():
 p=read(HERE/'runtime_plan001.json')
 for path,digest in p['source_sha256'].items():
  if identity.sha(path)!=digest:raise ValueError('timing source changed '+path)
 return p
def admit_gpu():
 p=plan();accepted=identity.admit_identity();proof=REV/'e4c_budget_ttft_v3_prelaunch_review.json';reviewer=REV/'review_e4c_budget_ttft_v3_prelaunch.py'
 if not proof.is_file():raise ValueError('GPU locked: timing independent prelaunch missing')
 r=read(proof);rp=HERE/'run_preflight001';fs=HERE/'monitor_preflight001/started.json';ft=HERE/'monitor_preflight001/terminal.json'
 fields={'plan_sha256':HERE/'runtime_plan001.json','identity_review_sha256':REV/'e4c_budget_identity001_review.json',
         'worker_preflight_sha256':rp/'worker_preflight.json','processor_preflight_sha256':rp/'processor_preflight.json',
         'preflight_launch_sha256':rp/'launch.json','preflight_exit_sha256':rp/'exit.json','preflight_started_sha256':fs,
         'preflight_terminal_sha256':ft,'reviewer_sha256':reviewer}
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='budget_ttft_v3_gpu_prelaunch' or r.get('errors')!=[] or
     r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or
     any(r.get(k)!=identity.sha(v) for k,v in fields.items())):raise ValueError('timing prelaunch proof mismatch')
 identity.terminal(fs,ft);s=identity.terminal(REV/'monitor_e4c_budget_ttft_v3_prelaunch/started.json',REV/'monitor_e4c_budget_ttft_v3_prelaunch/terminal.json')
 cmd=s['command'];cwd=Path(s['cwd']);paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if paths.count(str(reviewer))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(proof):raise ValueError('wrong timing reviewer command')
 w=read(rp/'worker_preflight.json');ex=read(rp/'exit.json');st=ex.get('last_inspect') or {}
 if (w.get('status')!='passed' or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('jobs')!=48 or
     w.get('runtime_plan_sha256')!=identity.sha(HERE/'runtime_plan001.json') or w.get('identity')!=accepted or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or
     st.get('running') is not False or st.get('pid')!=0 or st.get('host_devices') or st.get('device_requests')):raise ValueError('timing worker preflight invalid')
 return {'plan_sha256':identity.sha(HERE/'runtime_plan001.json'),'identity':accepted,'prelaunch_review_sha256':identity.sha(proof),'source_sha256':p['source_sha256']}
