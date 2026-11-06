"""Freeze candidate no-GPU preflight sources; no scoring or model access."""
import json,os
from pathlib import Path
import runtime_gate as gate
D=Path(__file__).resolve().parent;B=D.parents[1];v3=B/'e4_c_budget_ttft_pilot_v3/runtime_plan001.json';old=gate.read(v3)
for p,h in old['source_sha256'].items():
 if gate.sha(p)!=h:raise ValueError('prior reviewed source changed '+p)
for m in (D/'monitor_preflight_cpu001',D/'monitor_validate001',D/'monitor_freeze001'):
 s=gate.read(m/'started.json');t=gate.read(m/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('preliminary supervision failed')
proof=gate.read(D/'manifest_validation001.json')
if proof['status']!='passed' or proof['selections']!=1152 or proof['questions']!=384 or proof['reviewer_sha256']!=gate.sha(D/'validate_manifest.py'):raise ValueError('manifest audit failed')
random=gate.read(D/'plan001.json');source=dict(old['source_sha256']);source.update(random['source_sha256'])
files=[Path(__file__),D/'runtime_gate.py',D/'preflight_worker.py',D/'preflight_launch.py',D/'test_preflight.py',D/'plan001.json',D/'selection_manifest001.jsonl',D/'validate_manifest.py',D/'manifest_validation001.json',v3]
for m in (D/'monitor_preflight_cpu001',D/'monitor_validate001',D/'monitor_freeze001'):files.extend((m/'started.json',m/'terminal.json',m/'stderr.log'))
for p in files:source[str(p.resolve())]=gate.sha(p)
plan={'stage':'preflight_only','source_sha256':source,'random_plan_sha256':gate.sha(D/'plan001.json'),'manifest_sha256':gate.sha(D/'selection_manifest001.jsonl'),'budget_source_sha256':gate.sha(B/'initial_plan.json'),'inner_timeout_seconds':1200,'outer_timeout_seconds':1500,'grace_seconds':90,'GPU_authorized':False,'GT_opened':False,'predictions_opened':False,'limitations':['preflight only checks imports and weight payload; no E1 actual output or random model identity','full random GPU execution and independent endpoint review not yet implemented']}
with (D/'runtime_plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('source count',len(source))
