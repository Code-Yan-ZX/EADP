"""Fail closed before accessing new predictions or labels for t010 DEV scoring."""
import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];R=BASE/'e1_independent_review';D=BASE/'e6/pixelprune_t010_dev384';A=BASE/'e6/pixelprune_t010_accept';OLD=BASE/'e6/pixelprune_dev384_analysis';PLAN=HERE/'score_plan001.json';CPU=HERE/'cpu_check004/verification.json';CM=HERE/'monitor_cpu004';PRE=R/'pixel_t010_dev384_score_prelaunch_review.json';PR=R/'review_pixel_t010_dev384_score_prelaunch.py';PM=R/'monitor_pixel_t010_dev384_score_prelaunch'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervisor(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if not (t.get('reason')=='completed' and t.get('worker_returncode')==0 and t.get('remaining_process_group')==[] and s.get('worker_pid')==t.get('worker_pid') and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]):raise ValueError('supervisor identity or exit')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
def eq(d,expected):
 if any(d.get(k)!=v for k,v in expected.items()):raise ValueError('proof mismatch')
def legacy():
 path=OLD/'admission.py';spec=importlib.util.spec_from_file_location('eadp_pixel_t0_scoring_admission',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if Path(m.__file__).resolve()!=path.resolve():raise ValueError('legacy scoring module mismatch')
 return m
def sources():
 files=[PLAN,*[HERE/x for x in ('admission.py','score.py','paired.py','test_score.py','test_main_integration.py','cpu_check.py')],OLD/'admission.py',A/'verify.py',A/'verification_plan001.json',D/'smoke_gate.py']
 return sorted(set(files+legacy().sources()),key=str)
def source_hashes():return {str(p.resolve()):sha(p) for p in sources()}
def admission():
 for f in (PRE,PR,PM/'started.json',PM/'terminal.json',CPU,CM/'started.json',CM/'terminal.json'):
  if not f.is_file():raise ValueError('independent scoring review missing')
 supervisor(CM,HERE/'cpu_check.py',CPU.parent);supervisor(PM,PR,PRE);actual=source_hashes();c=read(CPU);r=read(PRE)
 eq(c,{'status':'passed','GT_opened':False,'real_predictions_opened':False,'source_sha256_before':actual,'source_sha256_after':actual})
 eq(r,{'status':'passed','independent_review':True,'scope':'pixel_t010_dev384_score_prelaunch','GT_opened':False,'predictions_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':sha(PLAN),'cpu_proof_sha256':sha(CPU),'cpu_started_sha256':sha(CM/'started.json'),'cpu_terminal_sha256':sha(CM/'terminal.json'),'reviewer_sha256':sha(PR)})
 # The already-reviewed t0 gate verifies its full scoring and upstream E1/E4 evidence.
 spec=importlib.util.spec_from_file_location('eadp_pixel_t010_four_image_gate',D/'smoke_gate.py');four=importlib.util.module_from_spec(spec);spec.loader.exec_module(four)
 if Path(four.__file__).resolve()!=(D/'smoke_gate.py').resolve():raise ValueError('four-image gate module mismatch')
 four_evidence=four.validate(read(D/'plan001.json'))
 prior=legacy().admission()
 oldscore=OLD/'score001';oldproof=R/'pixel_dev384_scores_review.json';oldreviewer=R/'review_pixel_dev384_scores.py';oldmonitor=R/'monitor_pixel_dev384_scores';supervisor(OLD/'monitor_score001',OLD/'score.py',oldscore);supervisor(oldmonitor,oldreviewer,oldproof);q0=read(oldproof)
 eq(q0,{'status':'passed','independent_review':True,'questions':384,'pixel_scores_recomputed':384,'comparison_scores_checked':1152,'errors':[],'score_sha256':sha(oldscore/'scores.jsonl'),'analysis_sha256':sha(oldscore/'analysis.json'),'frozen_sha256':sha(oldscore/'frozen.json'),'score_started_sha256':sha(OLD/'monitor_score001/started.json'),'score_terminal_sha256':sha(OLD/'monitor_score001/terminal.json'),'reviewer_sha256':sha(oldreviewer)})
 formal=A/'verification001.json';review=R/'pixel_t010_dev384_result_review.json';fm=A/'monitor_verify001';rm=R/'monitor_pixel_t010_dev384_result';rr=R/'review_pixel_t010_dev384_result.py'
 supervisor(fm,A/'verify.py',formal);supervisor(rm,rr,review);supervisor(D/'monitor_smoke001',D/'launch.py');f=read(formal);q=read(review)
 eq(f,{'status':'passed','questions':384,'physical_generations':384,'GT_opened':False,'scoring_performed':False,'errors':[],'verifier_sha256':sha(A/'verify.py')})
 eq(q,{'status':'passed','independent_review':True,'questions':384,'physical_generations':384,'GT_opened':False,'scoring_performed':False,'formal_sha256':sha(formal),'formal_started_sha256':sha(fm/'started.json'),'formal_terminal_sha256':sha(fm/'terminal.json'),'run_started_sha256':sha(D/'monitor_smoke001/started.json'),'run_terminal_sha256':sha(D/'monitor_smoke001/terminal.json'),'run_exit_sha256':sha(D/'run_smoke001/exit.json'),'reviewer_sha256':sha(rr)})
 per=q.get('per_question_files_sha256',{});files=f.get('files_sha256',{});sources=f.get('source_sha256',{})
 if len(per)!=768 or any(files.get(p)!=h for p,h in per.items()) or len(f.get('per_question',[]))!=384 or len(files)<768 or not sources:raise ValueError('t010 incomplete evidence inventory')
 if any(sha(p)!=h for p,h in files.items()) or any(sha(p)!=h for p,h in sources.items()):raise ValueError('t010 evidence changed')
 return {'source_sha256':actual,'pixel_formal_sha256':sha(formal),'pixel_review_sha256':sha(review),'prior_t0':prior,'four_image_evidence':four_evidence,'prelaunch_review_sha256':sha(PRE)}
