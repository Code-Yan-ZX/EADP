"""Freeze noGPU source closure and three-anchor prerequisites; no GPU authority."""
import json,os
from pathlib import Path
import pilot_gate
D=Path(__file__).resolve().parent;B=D.parents[1]
pilot=pilot_gate.require();proof=pilot_gate.read(D/'cpu_proof001.json');pilot_gate.terminal(D/'monitor_cpu_freeze001',D/'cpu_check.py')
if proof.get('status')!='passed' or proof['source_sha256_before']!=proof['source_sha256_after']:raise ValueError('CPU proof')
old=pilot_gate.read(B/'e6/rres_k256_pilot_v2/preflight_plan001.json');source=old['source_sha256'].copy()
for path,digest in proof['source_sha256_after'].items():
 if path in source and source[path]!=digest:raise ValueError('source conflict')
 source[path]=digest
files=[D/'cpu_proof001.json',D/'monitor_cpu_freeze001/started.json',D/'monitor_cpu_freeze001/terminal.json',D/'monitor_design001/started.json',D/'monitor_design001/terminal.json',B/'e6/rres_k256_execution_v2/verification001.json',B/'e6/rres_k256_execution_v2/monitor_verify001/started.json',B/'e6/rres_k256_execution_v2/monitor_verify001/terminal.json',B/'e6/rres_k256_execution_v2/monitor_run001/started.json',B/'e6/rres_k256_execution_v2/monitor_run001/terminal.json',B/'e6/rres_k256_execution_v2/source_snapshot001/manifest.json',B/'e1_independent_review/rres_k256_v2_result_review.json',B/'e1_independent_review/review_rres_k256_v2_result.py',B/'e1_independent_review/monitor_rres_k256_v2_result/started.json',B/'e1_independent_review/monitor_rres_k256_v2_result/terminal.json']
for path in files:source[str(path.resolve())]=pilot_gate.sha(path)
if any(pilot_gate.sha(k)!=v for k,v in source.items()):raise ValueError('inherited source changed')
value={'scope':'rres384_noGPU_processor_preflight','images':384,'resolution':512,'visual_tokens':256,'source_sha256':source,'design_sha256':pilot_gate.sha(D/'design001.json'),'pilot_review_sha256':pilot['review_sha256'],'GPU_authorized':False,'GPU_used':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed journal parsed for input identity; inherited source files including existing predictions hashed; no new predictions','scoring_performed':False,'errors':[]}
with (D/'preflight_plan001.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
