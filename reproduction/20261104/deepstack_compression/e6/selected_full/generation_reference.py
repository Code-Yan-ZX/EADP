"""Accepted Div/CD smoke generation controls, no labels or scoring.

Called only after no-GPU preflight returns. The accepted smoke is complete;
the mixed records file contains outputs, but only control metadata is retained.
"""
import json
from pathlib import Path
from full_imports import HERE,SMOKE,load
sha=load('eadp_selected_full_plan',HERE/'freeze_plan.py').sha
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def controls():
 proof=json.loads((SMOKE/'smoke_001_verification.json').read_text());path=(SMOKE/'smoke_001/records.jsonl').resolve()
 if proof.get('status')!='passed' or proof.get('files_sha256',{}).get(str(path))!=sha(path):raise ValueError('accepted smoke records changed')
 seen={}
 with path.open() as f:
  for line in f:
   r=json.loads(line)
   if r.get('forced') is not False:continue
   key=(r.get('method'),r.get('task'),r.get('path'))
   if key in seen:raise ValueError('duplicate accepted smoke control')
   seen[key]=r.get('generation_controls')
 expected={(m,t,p) for m in ('div','cd') for t in TASKS for p in ('author_stock','adapted')}
 if set(seen)!=expected:raise ValueError('accepted smoke controls incomplete')
 out={}
 for method in ('div','cd'):
  out[method]={}
  for task in TASKS:
   stock=seen[(method,task,'author_stock')];adapted=seen[(method,task,'adapted')]
   if not isinstance(stock,dict) or stock!=adapted:raise ValueError('accepted smoke path controls differ')
   out[method][task]=stock
 return out
