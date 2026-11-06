"""CPU-only import probe gate; never grants GPU authority."""
import json
from pathlib import Path
import gpu_gate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REV=ROOT/'e1_independent_review';IDENTITY=ROOT/'e6/rres_eadp_identity3_v2'
def plan():
 p=gpu_gate.read(HERE/'cpu_plan001.json')
 if p.get('scope')!='identity3_execution_cpu_only' or p.get('GPU_authorized') is not False or any(gpu_gate.sha(path)!=digest for path,digest in p['source_sha256'].items()):raise ValueError('execution CPU plan')
 q=gpu_gate.read(REV/'rres_eadp_identity3_v2_preflight_result_review.json');mon=REV/'monitor_rres_eadp_identity3_v2_preflight_result';run=IDENTITY/'preflight001';m=IDENTITY/'monitor_preflight001'
 files={'plan_sha256':IDENTITY/'preflight_plan001.json','worker_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':m/'started.json','terminal_sha256':m/'terminal.json','prelaunch_review_sha256':REV/'rres_eadp_identity3_v2_cpu_preflight_review.json','reviewer_sha256':REV/'review_rres_eadp_identity3_v2_preflight_result.py'}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GPU_authorized') is not False or q.get('errors')!=[] or any(q.get(k)!=gpu_gate.sha(v) for k,v in files.items()) or not gpu_gate.ended(mon/'started.json',mon/'terminal.json') or not gpu_gate.ended(m/'started.json',m/'terminal.json'):raise ValueError('prior diagnostic noGPU review')
 if not gpu_gate.ended(HERE/'monitor_cpu006/started.json',HERE/'monitor_cpu006/terminal.json'):raise ValueError('CPU tests')
 return p
def admit():
 p=plan();proof=REV/'rres_eadp_identity3_execution_cpu_review.json';reviewer=REV/'review_rres_eadp_identity3_execution_cpu.py';mon=REV/'monitor_rres_eadp_identity3_execution_cpu';q=gpu_gate.read(proof)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors')!=[] or q.get('plan_sha256')!=gpu_gate.sha(HERE/'cpu_plan001.json') or q.get('source_sha256')!=p['source_sha256'] or q.get('reviewer_sha256')!=gpu_gate.sha(reviewer) or not gpu_gate.ended(mon/'started.json',mon/'terminal.json'):raise ValueError('execution CPU independent review')
 return {'scope':p['scope'],'plan_sha256':gpu_gate.sha(HERE/'cpu_plan001.json'),'review_sha256':gpu_gate.sha(proof)}
def gpu():raise RuntimeError('CPU preflight never authorizes GPU')
