"""Preflight and provenance gate before opening any feature/score artifact."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';S=B/'e5/selector_real32';E=B/'e4_panel_v2';X=B/'e4_analysis'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('upstream supervision or command')
def prelaunch():
 f=read(D/'plan_freeze001.json');terminal(D/'monitor_freeze001',D/'freeze_plan.py')
 if f.get('status')!='frozen' or f.get('feature_or_score_values_opened') is not False or f.get('GT_opened') is not False or f.get('predictions_opened') is not False or f.get('gpu_used') is not False or f.get('plan_sha256')!=sha(D/'PLAN.md') or any(sha(k)!=v for k,v in f['source_sha256'].items()):raise ValueError('frozen analysis plan mismatch')
 p=read(D/'code_plan001.json');cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'freeze_code.py')
 if p.get('status')!='frozen' or p.get('plan_freeze_sha256')!=sha(D/'plan_freeze001.json') or p.get('GT_opened') is not False or p.get('feature_or_score_values_opened') is not False or cpu.get('status')!='passed' or cpu.get('returncode')!=0 or cpu['source_before']!=cpu['source_after'] or p['source_sha256']!=dict(cpu['source_after'],**{str((D/'cpu_proof001.json').resolve()):sha(D/'cpu_proof001.json')}) or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('code/source proof mismatch')
 proof=I/'e4_multistream_budget_prelaunch_review.json';reviewer=I/'review_e4_multistream_budget_prelaunch.py';mon=I/'monitor_e4_multistream_budget_prelaunch';terminal(mon,reviewer,proof);r=read(proof)
 fields={'plan_freeze_sha256':D/'plan_freeze001.json','code_plan_sha256':D/'code_plan001.json','cpu_proof_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('feature_or_score_values_opened') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('missing independent prelaunch')
 return f

def prior(f):
 # Only after complete independent prelaunch may mixed DEV records be opened.
 terminal(S/'monitor_run001',S/'launch.py');terminal(S/'monitor_verify_run001',S/'verify.py');terminal(I/'monitor_selector_real32_run001',I/'review_selector_real32_run001.py',I/'selector_real32_run001_review.json')
 sf=read(S/'run_001_verification.json');sr=read(I/'selector_real32_run001_review.json')
 if sf.get('status')!='passed' or sf.get('images')!=32 or sf.get('verifier_sha256')!=sha(S/'verify.py') or sr.get('status')!='passed' or sr.get('independent_review') is not True or sr.get('images')!=32 or sr.get('GT_opened') is not False or sr.get('errors')!=[] or sr.get('reviewer_sha256')!=sha(I/'review_selector_real32_run001.py') or sr.get('formal_sha256')!=sha(S/'run_001_verification.json') or sr.get('formal_started_sha256')!=sha(S/'monitor_verify_run001/started.json') or sr.get('formal_terminal_sha256')!=sha(S/'monitor_verify_run001/terminal.json') or sr.get('run_started_sha256')!=sha(S/'monitor_run001/started.json') or sr.get('run_terminal_sha256')!=sha(S/'monitor_run001/terminal.json') or sr.get('array_sha256')!=sf.get('array_sha256'):raise ValueError('selector accepted chain mismatch')
 cap={k:v for k,v in sr['array_sha256'].items() if k.endswith('/capture.pt')}
 if cap!=f['capture_inventory_sha256']:raise ValueError('capture identity changed')
 terminal(E/'monitor_panel003',E/'panel_launch.py');terminal(E/'monitor_verify_panel003',E/'panel_verify.py');terminal(I/'monitor_e4panel003',I/'review_e4panel.py',I/'e4panel003_review.json')
 ef=read(E/'panel_003_verification.json');er=read(I/'e4panel003_review.json');records=E/'panel_003/records.jsonl'
 if ef.get('status')!='passed' or ef.get('n_questions')!=384 or ef.get('all_arms_validated')!=3072 or ef.get('GT_opened') is not False or ef.get('verifier_sha256')!=sha(E/'panel_verify.py') or er.get('status')!='passed' or er.get('n_verified')!=384 or er.get('arms_verified')!=3072 or er.get('errors')!=[] or er.get('reviewer_sha256')!=sha(I/'review_e4panel.py') or er.get('formal_proof_sha256')!=sha(E/'panel_003_verification.json') or er.get('formal_verifier_started_sha256')!=sha(E/'monitor_verify_panel003/started.json') or er.get('formal_verifier_terminal_sha256')!=sha(E/'monitor_verify_panel003/terminal.json') or ef['files_sha256'].get(str(records.resolve()))!=sha(records):raise ValueError('E4 accepted chain mismatch')
 terminal(X/'monitor_score_panel003',X/'score_e4.py');terminal(X/'independent_score_review/monitor001',X/'independent_score_review/recheck.py')
 scoreproof=read(X/'independent_score_review/verification001.json');frozen=read(X/'score_panel003/frozen.json');scores=X/'score_panel003/scores.jsonl'
 if not frozen.get('source_code_sha256') or any(sha(k)!=v for k,v in frozen['source_code_sha256'].items()) or scoreproof.get('status')!='passed' or scoreproof.get('questions')!=384 or scoreproof.get('errors')!=[] or scoreproof.get('source_sha256')!=sha(X/'independent_score_review/recheck.py') or scoreproof.get('panel_proof_sha256')!=sha(E/'panel_003_verification.json') or scoreproof.get('review_proof_sha256')!=sha(I/'e4panel003_review.json') or frozen.get('panel_proof_sha256')!=sha(E/'panel_003_verification.json') or frozen.get('review_proof_sha256')!=sha(I/'e4panel003_review.json') or scoreproof.get('score_sha256')!=f['upstream_score_file_sha256'] or scoreproof['score_sha256']!=sha(scores):raise ValueError('E4 score provenance mismatch')
 return {'selector_review_sha256':sha(I/'selector_real32_run001_review.json'),'panel_proof_sha256':sha(E/'panel_003_verification.json'),'panel_review_sha256':sha(I/'e4panel003_review.json'),'score_review_sha256':sha(X/'independent_score_review/verification001.json'),'scores_sha256':sha(scores),'records_sha256':sha(records)}
