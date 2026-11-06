"""Scoring gate; no answers or predictions before both full batch-2 proofs."""
import importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;B=ROOT/'m3_oracle_batch2';V=ROOT/'m3_oracle_batch2_verify';R=ROOT/'e1_independent_review';sys.path.insert(0,str(B));import batch_gate
runtime_gate=batch_gate
RUN=B/'runtime_001';PROOF=V/'runtime_001_verification.json';FS=V/'monitor_verify001/started.json';FT=FS.parent/'terminal.json';RS=B/'monitor_runtime001/started.json';RT=RS.parent/'terminal.json';REVIEW=R/'m3oracle_batch2_panel_review.json';IS=R/'monitor_m3batch2_panel001/started.json';IT=IS.parent/'terminal.json'
def read(p):return json.loads(Path(p).read_text())
def formal_fields(f,h):
 if (f.get('status')!='passed' or f.get('stage')!='oracle_batch2_full' or f.get('n_questions')!=128 or f.get('new_generations')!=768 or f.get('scoring_performed') is not False or f.get('reference_sources_opened_by_verifier') is not False or f.get('verifier_sha256')!=h):raise ValueError('batch2 formal acceptance required')
def review_fields(r,fh,rh):
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('n_questions')!=128 or r.get('new_generations')!=768 or r.get('scoring_performed') is not False or r.get('reference_sources_opened') is not False or r.get('errors')!=[] or r.get('formal_proof_sha256')!=fh or r.get('reviewer_sha256')!=rh):raise ValueError('batch2 independent acceptance required')
def gate(a):
 paths={a.run:RUN,a.proof:PROOF,a.verify_started:FS,a.verify_terminal:FT,a.run_started:RS,a.run_terminal:RT,a.review:REVIEW,a.review_started:IS,a.review_terminal:IT}
 if len(paths)!=9 or any(x.resolve()!=y.resolve() for x,y in paths.items()):raise ValueError('batch2 scorer paths differ')
 for p in (PROOF,FS,FT,RS,RT,REVIEW,IS,IT):
  if not p.is_file():raise ValueError('completed batch2 and both independent reviews required')
 batch_gate.supervised(FS,FT,V/'verify.py',PROOF);batch_gate.supervised(IS,IT,R/'review_m3batch2_panel.py',REVIEW);f=read(PROOF);r=read(REVIEW);formal_fields(f,batch_gate.sha(V/'verify.py'));review_fields(r,batch_gate.sha(PROOF),batch_gate.sha(R/'review_m3batch2_panel.py'))
 if any(r.get(k)!=batch_gate.sha(p) for k,p in [('formal_started_sha256',FS),('formal_terminal_sha256',FT),('frozen_sha256',RUN/'frozen.json'),('host_receipt_sha256',RUN/'host_receipt.json')]):raise ValueError('batch2 review chain changed')
 env=read(ROOT/'m3_oracle_verify_v2/environment_002_v2.json');vs=read(FS);cmd=vs['command'];cwd=Path(vs['cwd'])
 if f.get('python_version')!=env['python_version'] or f.get('executable_sha256')!=env['executable_sha256'] or batch_gate.sha((cwd/cmd[0]).resolve())!=env['executable_sha256']:raise ValueError('batch2 author Python changed')
 snap=B/'runtime_001_sources';manifest=read(snap/'manifest.json');audit=read(V/'cpu_check_003/verification.json');batch_gate.supervised(V/'monitor_cpu003/started.json',V/'monitor_cpu003/terminal.json',V/'cpu_validation.py',V/'cpu_check_003')
 if (r.get('source_manifest_sha256')!=batch_gate.sha(snap/'manifest.json') or manifest.get('acceptance_cpu_proof_sha256')!=batch_gate.sha(V/'cpu_check_003/verification.json') or manifest.get('acceptance_cpu_started_sha256')!=batch_gate.sha(V/'monitor_cpu003/started.json') or manifest.get('acceptance_cpu_terminal_sha256')!=batch_gate.sha(V/'monitor_cpu003/terminal.json') or audit.get('status')!='passed' or audit.get('source_sha256_before')!=manifest.get('acceptance_source_sha256') or audit.get('source_sha256_after')!=manifest.get('acceptance_source_sha256')):raise ValueError('batch2 acceptance source proof changed')
 approval=read(B/'runtime_approval.json');cpu=read(approval['cpu_proof']);sources=manifest['source_sha256']
 if cpu.get('status')!='passed' or cpu.get('source_sha256_before')!=sources or cpu.get('source_sha256_after')!=sources:raise ValueError('batch2 execution source proof changed')
 for mapping in (sources,manifest['acceptance_source_sha256']):
  for p,h in mapping.items():
   if batch_gate.sha(p)!=h or batch_gate.sha(snap/'files'/str(Path(p)).lstrip('/'))!=h:raise ValueError('batch2 source or snapshot changed')
 for k,p in [('run_start_sha256',RS),('run_terminal_sha256',RT),('launch_sha256',RUN/'launch.json'),('exit_sha256',RUN/'exit.json'),('panel_events_sha256',RUN/'panel/panel_events.jsonl'),('worker_complete_sha256',RUN/'panel/worker_complete.json')]:
  if f.get(k)!=batch_gate.sha(p):raise ValueError('batch2 completed evidence changed')
 files=f.get('result_files_sha256',{})
 if len(files)!=128:raise ValueError('batch2 incomplete question inventory')
 for p,h in files.items():
  if Path(p).parent.resolve()!=(RUN/'panel').resolve() or batch_gate.sha(p)!=h:raise ValueError('batch2 question evidence changed')
 accepted=batch_gate.validate(host=True,require_future=False)
 if f.get('approval_sha256')!=accepted['approval_sha256']:raise ValueError('batch2 approval changed')
 spec=importlib.util.spec_from_file_location('batch2_formal_verify_for_scoring',V/'verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.check_terminal(RUN,read(RS),read(RT),read(RUN/'launch.json'),read(RUN/'exit.json'),read(B/'execution_plan_002.json'))
 return f
