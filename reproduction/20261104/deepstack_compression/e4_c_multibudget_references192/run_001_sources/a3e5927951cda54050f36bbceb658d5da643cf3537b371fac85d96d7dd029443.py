"""Fail-closed planned 888-arm reference provenance; no GPU authority alone."""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervised process not completed')
 return s
def resolved_command(s):
 cwd=Path(s['cwd']);return [x if x.startswith('-') else str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in s['command']]
def plan():
 p=read(HERE/'execution_plan001.json')
 if p.get('stage')!='222-input four-budget fixed32 reference generation' or p.get('jobs')!=888:raise ValueError('wrong reference plan')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('reference source changed '+path)
 return p
def prior():
 path=BASE/'e4_c_budget_ttft_pilot_v3/identity_gate.py'
 if sha(path)!='690ab54edca0490d0c6aad7f1443992171f033942de49802b737160dfe0f53f8':raise ValueError('prior identity gate changed before import')
 prop=HERE/'proposal.py'
 if sha(prop)!='4006b7b9ee63c00ac060bdf3fcd629b4d41a1e76ccc5ecb014431d93e94f2df4':raise ValueError('proposal gate changed before import')
 pspec=importlib.util.spec_from_file_location('_bulk_192_proposal_gate',prop);pm=importlib.util.module_from_spec(pspec);pspec.loader.exec_module(pm)
 if len(pm.load())!=222:raise ValueError('proposal identity')
 spec=importlib.util.spec_from_file_location('_bulk_reference_prior_identity',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.admit_identity()
def execution_cpu(p):
 proof=read(HERE/'execution_cpu001.json');start=HERE/'monitor_execution_cpu001/started.json';end=HERE/'monitor_execution_cpu001/terminal.json';s=terminal(start,end)
 if resolved_command(s).count(str(HERE/'freeze_execution.py'))!=1 or proof.get('status')!='passed' or proof.get('test_returncode')!=0 or proof.get('GPU_authorized') is not False or proof.get('predictions_opened') is not True or proof.get('plan_sha256')!=sha(HERE/'execution_plan001.json') or proof.get('source_sha256')!=p['source_sha256']:raise ValueError('bulk CPU source proof mismatch')
 return proof
def admit_cpu_preflight(p):
 execution_cpu(p)
 proof=REV/'e4c_multibudget_references192_worker_preflight_review.json';reviewer=REV/'review_e4c_multibudget_references192_worker_preflight.py'
 if not proof.is_file():raise ValueError('CPU container locked: independent preflight review missing')
 r=read(proof);start=REV/'monitor_e4c_multibudget_references192_worker_preflight/started.json';end=REV/'monitor_e4c_multibudget_references192_worker_preflight/terminal.json';s=terminal(start,end);cmd=resolved_command(s)
 fields={'plan_sha256':HERE/'execution_plan001.json','execution_cpu_sha256':HERE/'execution_cpu001.json','execution_cpu_started_sha256':HERE/'monitor_execution_cpu001/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu001/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='references192_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('reference CPU preflight review mismatch')
 return r
def admit_fixed_identity():
 formal=BASE/'e4_c_multibudget_fixed32_accept/verification001.json';fmon=BASE/'e4_c_multibudget_fixed32_accept/monitor_verify001';run=BASE/'e4_c_multibudget_fixed32';proof=REV/'e4c_multibudget_fixed32_result_review.json';reviewer=REV/'review_e4c_multibudget_fixed32_result.py';rmon=REV/'monitor_e4c_multibudget_fixed32_result_002';snap=run/'run_001_sources/manifest.json'
 if not formal.is_file() or not proof.is_file():raise ValueError('GPU locked: completed 12-pair fixed32 identity proof not available')
 f=read(formal);r=read(proof);terminal(fmon/'started.json',fmon/'terminal.json');s=terminal(rmon/'started.json',rmon/'terminal.json');cmd=resolved_command(s)
 fields={'formal_sha256':formal,'formal_started_sha256':fmon/'started.json','formal_terminal_sha256':fmon/'terminal.json','run_started_sha256':run/'monitor_run001/started.json','run_terminal_sha256':run/'monitor_run001/terminal.json','snapshot_manifest_sha256':snap,'reviewer_sha256':reviewer}
 if (f.get('status')!='passed' or f.get('pairs')!=12 or f.get('generations')!=24 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('errors')!=[] or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('pairs')!=12 or r.get('generations')!=24 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('timing_claims') is not False or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof)):raise ValueError('fixed32 identity proof mismatch')
 terminal(run/'monitor_run001/started.json',run/'monitor_run001/terminal.json')
 if f.get('run_started_sha256')!=sha(run/'monitor_run001/started.json') or f.get('run_terminal_sha256')!=sha(run/'monitor_run001/terminal.json') or f.get('snapshot_manifest_sha256')!=sha(snap):raise ValueError('fixed32 run evidence mismatch')
 arrays=f.get('array_sha256');root=(run/'run_001/diagnostic/arrays').resolve()
 if not isinstance(arrays,dict) or len(arrays)!=24:raise ValueError('fixed32 array inventory missing')
 for path,digest in arrays.items():
  if not Path(path).resolve().is_relative_to(root) or sha(path)!=digest:raise ValueError('fixed32 array changed')
 snapshot=read(snap);source=read(run/'execution_plan005.json')['source_sha256'];copies=snapshot.get('sources',{})
 if set(copies)!=set(source):raise ValueError('fixed32 source snapshot inventory mismatch')
 for original,digest in source.items():
  item=copies[original];copy=Path(item['snapshot']).resolve()
  if item.get('source')!=original or item.get('sha256')!=digest or not copy.is_relative_to(snap.parent.resolve()) or sha(copy)!=digest:raise ValueError('fixed32 source snapshot changed')
 return {'formal_sha256':sha(formal),'review_sha256':sha(proof),'snapshot_manifest_sha256':sha(snap)}
def admit_gpu():
 p=plan();prior();admit_cpu_preflight(p);fixed=admit_fixed_identity();proof=REV/'e4c_multibudget_references192_prelaunch_review.json';reviewer=REV/'review_e4c_multibudget_references192_prelaunch.py'
 if not proof.is_file():raise ValueError('GPU locked: bulk reference independent prelaunch proof not available')
 r=read(proof);rp=HERE/'run_preflight001';ms=HERE/'monitor_preflight001/started.json';mt=HERE/'monitor_preflight001/terminal.json';temp=REV/'e4c_multibudget_references192_worker_preflight_review.json';ts=REV/'monitor_e4c_multibudget_references192_worker_preflight/started.json';tt=REV/'monitor_e4c_multibudget_references192_worker_preflight/terminal.json';fm=BASE/'e4_c_multibudget_fixed32_accept/monitor_verify001';rm=REV/'monitor_e4c_multibudget_fixed32_result_002'
 fields={'plan_sha256':HERE/'execution_plan001.json','execution_cpu_sha256':HERE/'execution_cpu001.json','execution_cpu_started_sha256':HERE/'monitor_execution_cpu001/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu001/terminal.json','temp_review_sha256':temp,'temp_review_started_sha256':ts,'temp_review_terminal_sha256':tt,'worker_preflight_sha256':rp/'worker_preflight.json','processor_preflight_sha256':rp/'processor_preflight.json','preflight_launch_sha256':rp/'launch.json','preflight_exit_sha256':rp/'exit.json','preflight_started_sha256':ms,'preflight_terminal_sha256':mt,'fixed_formal_sha256':BASE/'e4_c_multibudget_fixed32_accept/verification001.json','fixed_formal_started_sha256':fm/'started.json','fixed_formal_terminal_sha256':fm/'terminal.json','fixed_review_sha256':REV/'e4c_multibudget_fixed32_result_review.json','fixed_review_started_sha256':rm/'started.json','fixed_review_terminal_sha256':rm/'terminal.json','reviewer_sha256':reviewer}
 accept=BASE/'e4_c_multibudget_references192_accept';fields.update({'acceptance_plan_sha256':accept/'verification_plan001.json','acceptance_freeze_started_sha256':accept/'monitor_freeze001/started.json','acceptance_freeze_terminal_sha256':accept/'monitor_freeze001/terminal.json'})
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='references192_gpu_prelaunch' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('reference GPU prelaunch proof mismatch')
 terminal(ms,mt);terminal(accept/'monitor_freeze001/started.json',accept/'monitor_freeze001/terminal.json');s=terminal(REV/'monitor_e4c_multibudget_references192_prelaunch/started.json',REV/'monitor_e4c_multibudget_references192_prelaunch/terminal.json');cmd=resolved_command(s)
 if cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('wrong reference reviewer command')
 w=read(rp/'worker_preflight.json');v=read(rp/'processor_preflight.json');ex=read(rp/'exit.json');st=ex.get('last_inspect') or {}
 if v.get('inputs_checked')!=222 or v.get('references_checked')!=888 or v.get('jobs')!=888 or v.get('GT_opened') is not False or v.get('model_loaded') is not False or v.get('cuda_initialized') is not False or v.get('proposal_sha256')!=p.get('proposal_sha256'):raise ValueError('222-input processor preflight incomplete')
 if (w.get('status')!='passed' or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or w.get('plan_sha256')!=sha(HERE/'execution_plan001.json') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or st.get('running') is not False or st.get('pid')!=0 or st.get('host_devices') or st.get('device_requests')):raise ValueError('reference no-GPU preflight invalid')
 return {'plan_sha256':sha(HERE/'execution_plan001.json'),'fixed_identity':fixed,'gpu_review_sha256':sha(proof),'source_sha256':p['source_sha256']}
def check_supervisor(s,runid,now,p):
 progress=HERE/('run_'+runid)/'diagnostic/journal.jsonl';cmd=resolved_command(s)
 if (s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or s.get('timeout_seconds')!=p['outer_timeout_seconds'] or s.get('stale_seconds')!=p['stale_seconds'] or s.get('progress_paths')!=[str(progress)] or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or s.get('grace_seconds')!=p['grace_seconds'] or cmd.count(str(HERE/'launch.py'))!=1 or s['command'].count('--run-id')!=1 or s['command'][s['command'].index('--run-id')+1]!=runid):raise ValueError('reference supervisor mismatch')
 initial=read(BASE/'initial_plan.json')
 if sha(BASE/'initial_plan.json')!=p['budget_source_sha256'] or now+p['outer_timeout_seconds']+p['acceptance_reserve_seconds']>initial['budget_end_unix'] or now<initial['budget_start_unix']:raise ValueError('reference deadline insufficient')
 return True
