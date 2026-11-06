"""Require accepted 16-arm lightweight identity before any latency GPU work."""
import json
from pathlib import Path
from identity_gate import sha,supervised,validate_b_identity
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;C=BASE/'e4_c_timing';REVIEW=BASE/'e1_independent_review'

def require_light_identity():
 formal=C/'light_smoke_003_verification.json';fs=C/'monitor_light_verify003/started.json';ft=C/'monitor_light_verify003/terminal.json';verifier=C/'light_verify.py'
 review=REVIEW/'e4c_light_smoke003_review.json';rs=REVIEW/'monitor_e4c_light_smoke003/started.json';rt=REVIEW/'monitor_e4c_light_smoke003/terminal.json';reviewer=REVIEW/'review_e4c_light_smoke.py'
 run=C/'light_smoke_003';run_started=C/'monitor_light_smoke003/started.json';run_terminal=C/'monitor_light_smoke003/terminal.json';plan=C/'light_gpu_smoke_launch_plan003.json'
 if not all(p.is_file() for p in (formal,fs,ft,review,rs,rt,run_started,run_terminal,plan)):raise ValueError('latency locked: light identity proof missing')
 validate_b_identity();supervised(fs,ft,verifier,formal);supervised(rs,rt,reviewer,review)
 f=json.loads(formal.read_text());r=json.loads(review.read_text());p=json.loads(plan.read_text());started=json.loads(run_started.read_text());terminal=json.loads(run_terminal.read_text())
 if (f.get('status')!='passed' or f.get('generations')!=16 or f.get('lightweight_generations')!=12 or f.get('comparisons')!=16 or
     f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(verifier) or
     r.get('status')!='passed' or r.get('independent_review') is not True or r.get('generations')!=16 or r.get('lightweight_generations')!=12 or r.get('comparisons')!=16 or
     r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('timing_claims') is not False or r.get('errors')!=[] or
     r.get('formal_proof_sha256')!=sha(formal) or r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('reviewer_sha256')!=sha(reviewer) or
     r.get('run_started_sha256')!=sha(run_started) or r.get('run_terminal_sha256')!=sha(run_terminal) or r.get('launch_plan_sha256')!=sha(plan)):
  raise ValueError('light identity proof fields mismatch')
 if terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid'):raise ValueError('light GPU supervisor incomplete')
 evidence=f.get('evidence_sha256');arrays=f.get('array_sha256');sources=f.get('source_sha256')
 required=[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','generated.jsonl','records.jsonl','comparisons.jsonl')]+[run_started,run_terminal,C/'light_reference001.json']
 if not isinstance(evidence,dict) or any(evidence.get(str(x.resolve()))!=sha(x) for x in required):raise ValueError('light evidence binding mismatch')
 if not isinstance(arrays,dict) or len(arrays)!=16 or any(Path(path).resolve().parent!=(run/'arrays').resolve() or sha(path)!=digest for path,digest in arrays.items()):raise ValueError('light archive identity mismatch')
 if not isinstance(sources,dict) or p.get('approval_bundle',{}).get('source_sha256')!=sources or set(p.get('source_snapshots',{}))!=set(sources):raise ValueError('light source snapshot inventory mismatch')
 for path,digest in sources.items():
  snap=p['source_snapshots'][path]
  if snap.get('sha256')!=digest or sha(snap['snapshot'])!=digest:raise ValueError('light source snapshot mismatch')
 if (sha(HERE/'light_adapter.py')!=sources.get(str(C/'light_adapter.py')) or
     sha(BASE/'e4/skip_deepstack.py')!=sources.get(str(BASE/'e4/skip_deepstack.py'))):raise ValueError('timing implementation differs from accepted light adapter')
 return {'formal_sha256':sha(formal),'review_sha256':sha(review),'run_started_sha256':sha(run_started),'run_terminal_sha256':sha(run_terminal),
         'launch_plan_sha256':sha(plan),'adapter_sha256':sha(HERE/'light_adapter.py'),'skip_sha256':sha(BASE/'e4/skip_deepstack.py')}
