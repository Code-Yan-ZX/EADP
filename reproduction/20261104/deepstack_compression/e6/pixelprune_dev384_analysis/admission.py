"""No-prediction/no-label fail-closed admission for Pixel DEV scoring."""
import hashlib,importlib.metadata,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];R=BASE/'e1_independent_review';E1=BASE/'e1_analysis';A=BASE/'e6/pixelprune_dev384_accept';D=BASE/'e6/pixelprune_dev384'
PLAN=A/'score_plan001.json';FROZEN=BASE/'e1/e1full001_score/frozen.json';CPU=HERE/'cpu_check003/verification.json';CM=HERE/'monitor_cpu003';PRE=R/'pixel_dev384_score_prelaunch_review.json';PR=R/'review_pixel_dev384_score_prelaunch.py';PM=R/'monitor_pixel_dev384_score_prelaunch'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervisor(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if not (t.get('reason')=='completed' and t.get('worker_returncode')==0 and t.get('remaining_process_group')==[] and type(s.get('worker_pid')) is int and s['worker_pid']>0 and s['worker_pid']==t.get('worker_pid') and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]):raise ValueError('supervisor identity or exit')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
 return s,t
def sources():
 frozen=read(FROZEN);return sorted(set([PLAN,*[HERE/x for x in ('admission.py','score.py','paired.py','test_score.py','cpu_check.py')],*[Path(x) for x in frozen['code_sha256']],E1/'completion.py']),key=str)
def source_hashes():return {str(p.resolve()):sha(p) for p in sources()}
def eq(d,expected):
 if any(d.get(k)!=v for k,v in expected.items()):raise ValueError('proof mismatch')
def e1_identity():
 frozen=read(FROZEN);review=read(R/'e1full001_score_review.json');supervisor(BASE/'e1/monitorscoree1full001',E1/'score.py',BASE/'e1/e1full001_score');supervisor(R/'monitor_e1score001',R/'review_score.py')
 eq(review,{'status':'passed','n_verified':384,'arms_verified':1536,'errors':[],'score_frozen_sha256':sha(FROZEN),'score_terminal_sha256':sha(BASE/'e1/monitorscoree1full001/terminal.json'),'reviewer_sha256':sha(R/'review_score.py')})
 if len(frozen['code_sha256'])!=6 or any(sha(p)!=h for p,h in frozen['code_sha256'].items()) or sha(E1/'completion.py')!=sha(BASE/'e1/e1full001_score_sources/completion.py') or any(importlib.metadata.version(k)!=v for k,v in frozen['runtime_versions'].items()):raise ValueError('E1 source or dependency identity')
 return frozen
def admission():
 for f in (PRE,PR,PM/'started.json',PM/'terminal.json',CPU,CM/'started.json',CM/'terminal.json'):
  if not f.is_file():raise ValueError('independent scoring review missing')
 frozen=e1_identity();supervisor(CM,HERE/'cpu_check.py',CPU.parent);supervisor(PM,PR,PRE);c=read(CPU);r=read(PRE);actual=source_hashes()
 eq(c,{'status':'passed','GT_opened':False,'real_predictions_opened':False,'source_sha256_before':actual,'source_sha256_after':actual})
 eq(r,{'status':'passed','independent_review':True,'scope':'pixel_dev384_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':sha(PLAN),'cpu_proof_sha256':sha(CPU),'cpu_started_sha256':sha(CM/'started.json'),'cpu_terminal_sha256':sha(CM/'terminal.json'),'e1_score_frozen_sha256':sha(FROZEN),'e1_score_review_sha256':sha(R/'e1full001_score_review.json'),'reviewer_sha256':sha(PR)})
 formal=A/'verification001.json';review=R/'pixel_dev384_result_review.json';fm=A/'monitor_verify001';rm=R/'monitor_pixel_dev384_result';rr=R/'review_pixel_dev384_result.py'
 supervisor(fm,A/'verify.py',formal);supervisor(rm,rr,review);supervisor(D/'monitor_smoke001',D/'launch.py');f=read(formal);q=read(review)
 eq(f,{'status':'passed','questions':384,'physical_generations':384,'GT_opened':False,'scoring_performed':False,'errors':[],'verifier_sha256':sha(A/'verify.py')})
 eq(q,{'status':'passed','independent_review':True,'questions':384,'physical_generations':384,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(formal),'formal_started_sha256':sha(fm/'started.json'),'formal_terminal_sha256':sha(fm/'terminal.json'),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'run_smoke001/exit.json'),'reviewer_sha256':sha(rr)})
 if len(q.get('per_question_files_sha256',{}))!=768 or any(f['files_sha256'].get(p)!=h for p,h in q['per_question_files_sha256'].items()):raise ValueError('independent record inventory mismatch')
 # Bind all accepted byte files, including records: hashes are read but data is not parsed here.
 if len(f.get('per_question',[]))!=384 or len(f.get('files_sha256',{}))<768 or any(sha(p)!=h for p,h in f['files_sha256'].items()) or any(sha(p)!=h for p,h in f['source_sha256'].items()):raise ValueError('Pixel accepted files changed')
 ep=read(BASE/'e1/e1full001_completion.json');er=read(R/'e1full001_review.json')
 supervisor(BASE/'e1/monitorverifye1full001',E1/'completion.py');supervisor(R/'monitor_e1full001',R/'review_full.py')
 if ep.get('status')!='passed' or er.get('status')!='passed' or er.get('completion_proof_sha256')!=sha(BASE/'e1/e1full001_completion.json') or er.get('reviewer_sha256')!=sha(R/'review_full.py') or sha(ep['files']['journal']['path'])!=ep['files']['journal']['sha256']:raise ValueError('E1 panel proof mismatch')
 e4formal=read(BASE/'e4_panel_v2/panel_003_verification.json');e4review=read(R/'e4panel003_review.json')
 supervisor(BASE/'e4_panel_v2/monitor_verify_panel003',BASE/'e4_panel_v2/panel_verify.py');supervisor(R/'monitor_e4panel003',R/'review_e4panel.py')
 if e4formal.get('status')!='passed' or e4review.get('status')!='passed' or e4review.get('n_verified')!=384 or e4review.get('arms_verified')!=3072 or e4review.get('reviewer_sha256')!=sha(R/'review_e4panel.py') or e4review.get('formal_proof_sha256',e4review.get('formal_sha256'))!=sha(BASE/'e4_panel_v2/panel_003_verification.json'):raise ValueError('E4 panel proof mismatch')
 records=BASE/'e4_panel_v2/panel_003/records.jsonl'
 if e4formal.get('files_sha256',{}).get(str(records))!=sha(records):raise ValueError('E4 records changed')
 e4=read(R/'e4score003_review.json');supervisor(R/'monitor_e4score003',R/'review_e4score.py');supervisor(BASE/'e4_analysis/monitor_score_panel003',BASE/'e4_analysis/score_e4.py')
 eq(e4,{'status':'passed','questions_verified':384,'arms_verified':3072,'errors':[],'score_terminal_sha256':sha(BASE/'e4_analysis/monitor_score_panel003/terminal.json'),'reviewer_sha256':sha(R/'review_e4score.py')})
 e1r=read(R/'e1full001_score_review.json')
 if e1r['score_sha256']!=sha(BASE/'e1/e1full001_score/scores.jsonl') or e4['score_journal_sha256']!=sha(BASE/'e4_analysis/score_panel003/scores.jsonl'):raise ValueError('comparison scores changed')
 return {'source_sha256':actual,'pixel_formal_sha256':sha(formal),'pixel_review_sha256':sha(review),'e1_frozen':frozen,'e4_score_review_sha256':sha(R/'e4score003_review.json'),'prelaunch_review_sha256':sha(PRE)}
