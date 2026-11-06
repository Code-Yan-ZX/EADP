"""Authorize only the no-device model-import preflight, never GPU execution."""
import hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';I=B/'e1_independent_review'
sys.path.insert(0,str(F));import pilot_gate
sha=pilot_gate.sha;read=pilot_gate.read;terminal=pilot_gate.terminal
def admit():
 prior=pilot_gate.require();p=read(D/'runtime_preflight_plan001.json');qpath=I/'rres_k256_dev384_runtime_preflight_review.json';reviewer=I/'review_rres_k256_dev384_runtime_preflight.py';terminal(I/'monitor_rres_k256_dev384_runtime_preflight',reviewer,qpath);q=read(qpath)
 if p.get('scope')!='rres384_runtime_import_noGPU' or p.get('GPU_authorized') is not False or p.get('pilot_review_sha256')!=prior['review_sha256'] or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('runtime plan')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('source_sha256')!=p['source_sha256'] or q.get('plan_sha256')!=sha(D/'runtime_preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('runtime review')
 return {'plan_sha256':sha(D/'runtime_preflight_plan001.json'),'review_sha256':sha(qpath)}
def admit_gpu():raise RuntimeError('runtime noGPU admission never authorizes GPU')
