"""Identity diagnostic CPU preflight gate. GPU is still hard locked."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4';REV=ROOT/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def ended(s,t):
 a=read(s);b=read(t);return type(a.get('worker_pid')) is int and b.get('worker_pid')==a['worker_pid'] and b.get('reason')=='completed' and b.get('worker_returncode')==0 and b.get('remaining_process_group')==[]
def prior():
 q=read(REV/'rres_eadp_timing_v4_preflight_result_review.json');run=TIMING/'preflight001';mon=TIMING/'monitor_preflight001';revmon=REV/'monitor_rres_eadp_timing_v4_preflight_result'
 files={'plan_sha256':TIMING/'preflight_plan001.json','worker_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':mon/'started.json','terminal_sha256':mon/'terminal.json','prelaunch_review_sha256':REV/'rres_eadp_timing_v4_cpu_preflight_review.json','reviewer_sha256':REV/'review_rres_eadp_timing_v4_preflight_result.py'}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(p) for k,p in files.items()) or not ended(mon/'started.json',mon/'terminal.json') or not ended(revmon/'started.json',revmon/'terminal.json'):raise ValueError('upstream noGPU review')
 return {'review_sha256':sha(REV/'rres_eadp_timing_v4_preflight_result_review.json'),'review_started_sha256':sha(revmon/'started.json'),'review_terminal_sha256':sha(revmon/'terminal.json')}
def plan():
 p=read(HERE/'preflight_plan001.json')
 if p.get('scope')!='identity3_cpu_preflight_only' or p.get('GPU_authorized') is not False or any(sha(path)!=digest for path,digest in p['source_sha256'].items()):raise ValueError('identity3 source mismatch')
 if p['prior']!=prior() or not ended(HERE/'monitor_cpu008/started.json',HERE/'monitor_cpu008/terminal.json') or not ended(HERE/'monitor_import_order001/started.json',HERE/'monitor_import_order001/terminal.json'):raise ValueError('identity3 CPU chain')
 return p
def preflight():
 p=plan();proof=REV/'rres_eadp_identity3_v2_cpu_preflight_review.json';reviewer=REV/'review_rres_eadp_identity3_v2_cpu_preflight.py';mon=REV/'monitor_rres_eadp_identity3_v2_cpu_preflight';q=read(proof)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors')!=[] or q.get('plan_sha256')!=sha(HERE/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('source_sha256')!=p['source_sha256'] or not ended(mon/'started.json',mon/'terminal.json'):raise ValueError('identity3 independent CPU review')
 return {'plan_sha256':sha(HERE/'preflight_plan001.json'),'review_sha256':sha(proof),'scope':p['scope']}
def gpu():raise RuntimeError('GPU identity diagnostic locked until GPU plan, actual noGPU preflight and independent review')
