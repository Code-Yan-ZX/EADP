"""Fail-closed scoring admission: no panel data/labels before complete independent reviews."""
import hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';FULL=B/'e6/stacktok_full';OLD=B/'e6/pixelprune_t045_dev384_analysis'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervisor(m,script,out=None):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor/command mismatch')
 if out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('output mismatch')
 return s
def prior_module():
 path=OLD/'admission.py';spec=importlib.util.spec_from_file_location('_stacktok_prior_pixel_scoring',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path.resolve():raise ValueError('prior admission module mismatch')
 return m
def sources():
 own=[D/x for x in ('score_plan001.json','admission.py','score.py','paired.py','test_score.py','test_main_integration.py','test_paired.py','cpu_check.py')]
 return sorted(set(own+prior_module().sources()+[FULL/'verify.py',FULL/'full_refs.py',FULL/'full_arm.py']),key=str)
def source_hashes():return {str(p.resolve()):sha(p) for p in sources()}
def equal(d,expected):
 if any(d.get(k)!=v for k,v in expected.items()):raise ValueError('proof mismatch')
def require():
 # Own source-only proof and real exit must be checked before any panel outputs.
 proof=I/'stacktok_full_score_prelaunch_review.json';reviewer=I/'review_stacktok_full_score_prelaunch.py';monitor=I/'monitor_stacktok_full_score_prelaunch';cpu=D/'cpu_check003/verification.json';cm=D/'monitor_cpu003'
 if any(not p.exists() for p in (proof,reviewer,monitor/'started.json',monitor/'terminal.json',cpu,cm/'started.json',cm/'terminal.json')):raise ValueError('scoring source review missing')
 supervisor(cm,D/'cpu_check.py',cpu.parent);supervisor(monitor,reviewer,proof);actual=source_hashes();c=read(cpu);r=read(proof)
 equal(c,{'status':'passed','source_sha256_before':actual,'source_sha256_after':actual,'GT_opened':False,'real_predictions_opened':False})
 equal(r,{'status':'passed','independent_review':True,'scope':'stacktok_full_score_prelaunch','source_sha256':actual,'plan_sha256':sha(D/'score_plan001.json'),'cpu_sha256':sha(cpu),'cpu_started_sha256':sha(cm/'started.json'),'cpu_terminal_sha256':sha(cm/'terminal.json'),'reviewer_sha256':sha(reviewer),'GT_opened':False,'predictions_opened':False,'errors':[]})
 # Full no-GT acceptance and independent reviewer exit precede any output hash/read.
 formal=FULL/'run_001_verification.json';fm=FULL/'monitor_verify001';independent=I/'stacktok_full_run001_review.json';im=I/'monitor_stacktok_full_run001';ir=I/'review_stacktok_full_result.py'
 for x in (formal,fm/'started.json',fm/'terminal.json',independent,im/'started.json',im/'terminal.json'):
  if not x.exists():raise ValueError('full StackTok acceptance missing')
 supervisor(fm,FULL/'verify.py',formal);supervisor(im,ir,independent);supervisor(FULL/'monitor_run001',FULL/'launch.py');f=read(formal);q=read(independent)
 equal(f,{'status':'passed','questions':384,'generations':384,'GT_opened':False,'scoring_performed':False,'verifier_sha256':sha(FULL/'verify.py')})
 equal(q,{'status':'passed','independent_review':True,'questions':384,'generations':384,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(formal),'formal_started_sha256':sha(fm/'started.json'),'formal_terminal_sha256':sha(fm/'terminal.json'),'run_started_sha256':sha(FULL/'monitor_run001/started.json'),'run_terminal_sha256':sha(FULL/'monitor_run001/terminal.json'),'reviewer_sha256':sha(ir)})
 files=f.get('files_sha256');indfiles=q.get('files_sha256')
 if not isinstance(files,dict) or not files or indfiles!=files or any(sha(k)!=v for k,v in files.items()):raise ValueError('full evidence inventory changed')
 # Old score and t045 admission include E1/E4, both Pixel panels, author sources and environments.
 prior=prior_module().admission();old=OLD/'score001';op=I/'pixel_t045_dev384_scores_review.json';orr=I/'review_pixel_t045_dev384_scores.py';om=I/'monitor_pixel_t045_dev384_scores002';supervisor(OLD/'monitor_score001',OLD/'score.py',old);supervisor(om,orr,op);o=read(op)
 equal(o,{'status':'passed','independent_review':True,'questions':384,'pixel_scores_recomputed':768,'comparison_scores_checked':768,'reviewer_sha256':sha(orr),'score_sha256':sha(old/'scores.jsonl'),'analysis_sha256':sha(old/'analysis.json'),'frozen_sha256':sha(old/'frozen.json'),'score_started_sha256':sha(OLD/'monitor_score001/started.json'),'score_terminal_sha256':sha(OLD/'monitor_score001/terminal.json')})
 return {'source_sha256':actual,'full_formal_sha256':sha(formal),'full_independent_sha256':sha(independent),'prior_t045_score_review_sha256':sha(op),'prior':prior,'score_prelaunch_sha256':sha(proof)}
