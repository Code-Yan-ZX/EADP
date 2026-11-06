"""Exploratory 32-image performance pilot is intentionally still locked."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 p=read(HERE/'runtime_plan001.json')
 if p.get('stage')!='post-hoc 32-image 4-budget natural/fixed32 timing pilot' or p.get('jobs')!=1520:raise ValueError('invalid timing plan')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('timing source changed '+path)
 return p
def admit_gpu():
 import importlib.util
 p=plan();admit_cpu_preflight()
 path=HERE/'fixed_reference_gate.py';spec=importlib.util.spec_from_file_location('_multi_timing_fixed_ref_admit',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);refs=m.load()
 if len(refs)!=248:raise ValueError('GPU locked: 248 reference inventory incomplete')
 proof=REV/'e4c_multibudget_worker_v3_prelaunch_review.json';reviewer=REV/'review_e4c_multibudget_worker_v3_prelaunch.py'
 if not proof.is_file():raise ValueError('GPU locked: independent timing prelaunch proof missing')
 r=read(proof);rp=HERE/'run_preflight001';ms=HERE/'monitor_preflight001/started.json';mt=HERE/'monitor_preflight001/terminal.json';temp=REV/'e4c_multibudget_worker_v3_preflight_review.json';ts=REV/'monitor_e4c_multibudget_worker_v3_preflight/started.json';tt=REV/'monitor_e4c_multibudget_worker_v3_preflight/terminal.json';acc=BASE/'e4_c_multibudget_references_accept_v3';run=BASE/'e4_c_multibudget_references_v3';fm=acc/'monitor_verify001';rm=REV/'monitor_e4c_multibudget_references_v3_result';accept=BASE/'e4_c_multibudget_worker_accept_v3'
 fields={'plan_sha256':HERE/'runtime_plan001.json','execution_cpu_sha256':HERE/'execution_cpu001.json','execution_cpu_started_sha256':HERE/'monitor_execution_cpu001/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu001/terminal.json','temp_review_sha256':temp,'temp_review_started_sha256':ts,'temp_review_terminal_sha256':tt,'worker_preflight_sha256':rp/'worker_preflight.json','processor_preflight_sha256':rp/'processor_preflight.json','preflight_launch_sha256':rp/'launch.json','preflight_exit_sha256':rp/'exit.json','preflight_started_sha256':ms,'preflight_terminal_sha256':mt,'reference_formal_sha256':acc/'verification001.json','reference_formal_started_sha256':fm/'started.json','reference_formal_terminal_sha256':fm/'terminal.json','reference_review_sha256':REV/'e4c_multibudget_references_v3_result_review.json','reference_review_started_sha256':rm/'started.json','reference_review_terminal_sha256':rm/'terminal.json','acceptance_plan_sha256':accept/'verification_plan001.json','acceptance_freeze_started_sha256':accept/'monitor_freeze001/started.json','acceptance_freeze_terminal_sha256':accept/'monitor_freeze001/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='multibudget_worker_v3_gpu_prelaunch' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('timing GPU prelaunch proof mismatch')
 terminal(ms,mt);terminal(accept/'monitor_freeze001/started.json',accept/'monitor_freeze001/terminal.json');s=terminal(REV/'monitor_e4c_multibudget_worker_v3_prelaunch/started.json',REV/'monitor_e4c_multibudget_worker_v3_prelaunch/terminal.json');cmd=resolved_command(s)
 if cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('wrong timing reviewer command')
 w=read(rp/'worker_preflight.json');ex=read(rp/'exit.json');st=ex.get('last_inspect') or {}
 if (w.get('status')!='passed' or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('inputs')!=62 or w.get('jobs')!=1520 or w.get('plan_sha256')!=sha(HERE/'runtime_plan001.json') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or st.get('running') is not False or st.get('pid')!=0 or st.get('host_devices') or st.get('device_requests')):raise ValueError('timing no-GPU preflight invalid')
 return {'plan_sha256':sha(HERE/'runtime_plan001.json'),'references_formal_sha256':sha(acc/'verification001.json'),'references_review_sha256':sha(REV/'e4c_multibudget_references_v3_result_review.json'),'gpu_review_sha256':sha(proof),'source_sha256':p['source_sha256']}

def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervised process not completed')
 return s
def resolved_command(s):
 cwd=Path(s['cwd']);return [x if x.startswith('-') else str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in s['command']]
def admit_cpu_preflight():
 p=plan();cpu=read(HERE/'execution_cpu001.json');cs=HERE/'monitor_execution_cpu001/started.json';ct=HERE/'monitor_execution_cpu001/terminal.json';s=terminal(cs,ct)
 if resolved_command(s).count(str(HERE/'freeze_execution.py'))!=1 or cpu.get('status')!='passed' or cpu.get('test_returncode')!=0 or cpu.get('predictions_opened') is not True or cpu.get('model_loaded') is not False or cpu.get('GPU_authorized') is not False or cpu.get('plan_sha256')!=sha(HERE/'runtime_plan001.json') or cpu.get('source_sha256')!=p['source_sha256']:raise ValueError('timing CPU proof mismatch')
 proof=REV/'e4c_multibudget_worker_v3_preflight_review.json';reviewer=REV/'review_e4c_multibudget_worker_v3_preflight.py'
 if not proof.is_file():raise ValueError('CPU container locked: independent source review unavailable')
 r=read(proof);ms=REV/'monitor_e4c_multibudget_worker_v3_preflight/started.json';mt=REV/'monitor_e4c_multibudget_worker_v3_preflight/terminal.json';s=terminal(ms,mt);cmd=resolved_command(s)
 fields={'plan_sha256':HERE/'runtime_plan001.json','execution_cpu_sha256':HERE/'execution_cpu001.json','execution_cpu_started_sha256':cs,'execution_cpu_terminal_sha256':ct,'reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='multibudget_worker_v3_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('timing CPU review mismatch')
 return r
