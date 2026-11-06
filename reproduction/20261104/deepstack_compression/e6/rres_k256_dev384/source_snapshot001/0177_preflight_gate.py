"""No-device preflight only. The GPU entry is deliberately hard locked."""
import json
from pathlib import Path
import pilot_gate
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
sha=pilot_gate.sha;read=pilot_gate.read;terminal=pilot_gate.terminal
def admit_preflight():
 prior=pilot_gate.require();plan=read(D/'preflight_plan001.json');proof=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu_freeze001',D/'cpu_check.py')
 if plan.get('scope')!='rres384_noGPU_processor_preflight' or plan.get('GPU_authorized') is not False or proof.get('status')!='passed' or proof['source_sha256_before']!=proof['source_sha256_after'] or any(sha(k)!=v for k,v in plan['source_sha256'].items()) or any(plan['source_sha256'].get(k)!=v for k,v in proof['source_sha256_after'].items()) or plan.get('pilot_review_sha256')!=prior['review_sha256']:raise ValueError('CPU preflight plan')
 qpath=I/'rres_k256_dev384_cpu_preflight_review.json';reviewer=I/'review_rres_k256_dev384_cpu_preflight.py';terminal(I/'monitor_rres_k256_dev384_cpu_preflight',reviewer,qpath);q=read(qpath)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('source_sha256')!=plan['source_sha256'] or q.get('plan_sha256')!=sha(D/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('preflight independent review')
 return {'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(qpath)}
def admit_gpu():raise RuntimeError('GPU locked: complete model worker, independent noGPU result and GPU review required')
