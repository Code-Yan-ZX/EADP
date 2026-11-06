"""Bind actual noGPU results and source for a no-device import preflight."""
import json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';I=B/'e1_independent_review';A=B/'e6/rres_k256_full_analysis_v2'
sys.path.insert(0,str(F));import pilot_gate
read=pilot_gate.read;sha=pilot_gate.sha;terminal=pilot_gate.terminal
prior=pilot_gate.require();cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu_freeze001',D/'cpu_check.py')
if cpu.get('status')!='passed' or cpu['source_sha256_before']!=cpu['source_sha256_after']:raise ValueError('CPU proof')
pre=I/'rres_k256_dev384_preflight_result_review.json';pr=I/'review_rres_k256_dev384_preflight_result.py';terminal(I/'monitor_rres_k256_dev384_preflight_result',pr,pre);q=read(pre)
pre_fields={'plan_sha256':F/'preflight_plan001.json','worker_preflight_sha256':F/'run_preflight001/worker_preflight.json','processor_probe_sha256':F/'run_preflight001/probe/processor_probe.json','launch_sha256':F/'run_preflight001/launch.json','identity_sha256':F/'run_preflight001/container_identity.json','exit_sha256':F/'run_preflight001/exit.json','started_sha256':F/'monitor_preflight001/started.json','terminal_sha256':F/'monitor_preflight001/terminal.json','reviewer_sha256':pr}
if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=384 or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in pre_fields.items()):raise ValueError('independent 384 noGPU result')
score=I/'rres_k256_full_score_prelaunch_review.json';sr=I/'review_rres_k256_full_score_prelaunch.py';terminal(I/'monitor_rres_k256_full_score_prelaunch',sr,score);sq=read(score)
if sq.get('status')!='passed' or sq.get('score_plan_sha256')!=sha(A/'score_plan001.json') or sq.get('cpu_proof_sha256')!=sha(A/'cpu_proof001.json') or sq.get('reviewer_sha256')!=sha(sr):raise ValueError('independent scoring prelaunch')
source=read(F/'preflight_plan001.json')['source_sha256'].copy();source.update(cpu['source_sha256_after'])
files=[*pre_fields.values(),pre,I/'monitor_rres_k256_dev384_preflight_result/started.json',I/'monitor_rres_k256_dev384_preflight_result/terminal.json',score,sr,I/'monitor_rres_k256_full_score_prelaunch/started.json',I/'monitor_rres_k256_full_score_prelaunch/terminal.json',A/'score_plan001.json',A/'cpu_proof001.json',D/'cpu_proof001.json',D/'monitor_cpu_freeze001/started.json',D/'monitor_cpu_freeze001/terminal.json',D/'host_smoke_case004/monitor_run001/started.json',D/'host_smoke_case004/monitor_run001/terminal.json']
for p in files:source[str(p.resolve())]=sha(p)
if any(sha(k)!=v for k,v in source.items()):raise ValueError('source mismatch')
value={'scope':'rres384_runtime_import_noGPU','source_sha256':source,'pilot_review_sha256':prior['review_sha256'],'preflight_result_sha256':sha(pre),'score_prelaunch_sha256':sha(score),'GPU_authorized':False,'GPU_used':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'existing accepted DEV predictions parsed for identity or hashed as inherited sources; no new predictions','scoring_performed':False,'errors':[]}
with (D/'runtime_preflight_plan001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
