"""Read-only, fail-closed scorer provenance. Does not open predictions or labels."""
import hashlib,importlib.metadata,json
from pathlib import Path
HERE=Path(__file__).resolve().parent; BASE=HERE.parents[1]; E1=BASE/'e1_analysis'; REVIEW=BASE/'e1_independent_review'
FROZEN=BASE/'e1/e1full001_score/frozen.json'; E1REVIEW=REVIEW/'e1full001_score_review.json'
PRE=REVIEW/'e6_selected_score_prelaunch_review.json'; REVIEWER=REVIEW/'review_e6_selected_score_prelaunch.py'; MON=REVIEW/'monitor_e6_selected_score_prelaunch'
CPU=HERE/'cpu_check005/verification.json'; CMON=HERE/'monitor_cpu005'
PLAN_SHA='f397d895e27b143e1dab1ada05f8f039e8913fcc7a651eacfac9cd13c6832b43'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervisor(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor identity or exit')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
 return s,t
def sources():
 e1=read(FROZEN)['code_sha256'];return [HERE/x for x in ('plan.json','admission.py','gate.py','inventory.py','paired.py','score_selected.py','test_selected.py','cpu_check.py')]+[Path(x) for x in e1]+[E1/'completion.py',BASE/'e1/e1full001_score_sources/completion.py']
def source_hashes():return {str(p.resolve()):sha(p) for p in sources()}
def e1_identity():
 frozen=read(FROZEN);r=read(E1REVIEW)
 supervisor(BASE/'e1/monitorscoree1full001',E1/'score.py',BASE/'e1/e1full001_score')
 supervisor(REVIEW/'monitor_e1score001',REVIEW/'review_score.py')
 if (r.get('status')!='passed' or r.get('n_expected')!=384 or r.get('n_verified')!=384 or r.get('arms_verified')!=1536 or r.get('errors')!=[] or r.get('score_frozen_sha256')!=sha(FROZEN) or r.get('score_terminal_sha256')!=sha(BASE/'e1/monitorscoree1full001/terminal.json') or r.get('reviewer_sha256')!=sha(REVIEW/'review_score.py')):raise ValueError('E1 independently reviewed scorer identity')
 expected=frozen['code_sha256'];required={str((E1/x).resolve()) for x in ('score.py','metrics.py','paired_stats.py')}
 if len(expected)!=6 or not required.issubset(expected) or any(sha(p)!=v for p,v in expected.items()):raise ValueError('E1 source differs from verified scorer')
 if sha(E1/'completion.py')!=sha(BASE/'e1/e1full001_score_sources/completion.py'):raise ValueError('E1 completion snapshot differs')
 versions=frozen['runtime_versions']
 if set(versions)!={'numpy','pandas','xlsxwriter','openpyxl'} or any(importlib.metadata.version(k)!=v for k,v in versions.items()):raise ValueError('E1 scorer dependency versions differ')
 return frozen
def admit():
 for p in (PRE,REVIEWER,MON/'started.json',MON/'terminal.json',CPU,CMON/'started.json',CMON/'terminal.json'):
  if not p.is_file():raise ValueError('independent scoring prelaunch review required')
 e1_identity();supervisor(CMON,HERE/'cpu_check.py');supervisor(MON,REVIEWER,PRE)
 c=read(CPU);p=read(PRE);actual=source_hashes()
 if sha(HERE/'plan.json')!=PLAN_SHA or c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('real_predictions_opened') is not False or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual:raise ValueError('CPU scoring proof or source changed')
 expected={'status':'passed','independent_review':True,'scope':'selected_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':PLAN_SHA,'cpu_proof_sha256':sha(CPU),'cpu_started_sha256':sha(CMON/'started.json'),'cpu_terminal_sha256':sha(CMON/'terminal.json'),'e1_score_frozen_sha256':sha(FROZEN),'e1_score_review_sha256':sha(E1REVIEW),'reviewer_sha256':sha(REVIEWER)}
 if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('independent scorer review differs')
 return {'source_sha256':actual,'prelaunch_review_sha256':sha(PRE),'cpu_proof_sha256':sha(CPU)}
