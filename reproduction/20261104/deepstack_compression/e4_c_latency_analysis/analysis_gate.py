"""Fail-closed provenance and prelaunch gates, no timing-value parsing."""
import hashlib,importlib.metadata,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parent;F=B/'e4_c_latency_full';R=B/'e1_independent_review'
P=HERE/'analysis_plan001.json';SUP=HERE/'analysis_plan001_supplement.json';CPU=HERE/'cpu_check004/verification.json';CM=HERE/'monitor_cpu004'
PRE=R/'e4c_latency_analysis_prelaunch_review.json';PRE_SCRIPT=R/'review_e4c_latency_analysis_prelaunch.py';PRE_MON=R/'monitor_e4c_latency_analysis_prelaunch'
FORMAL=F/'full_001_verification.json';FM=F/'monitor_verify_full001';RUNM=F/'monitor_full001';IND=R/'e4c_latency_full001_review.json';IM=R/'monitor_e4c_latency_full001'
MAN=HERE/'manual_overlap001/verification.json';MANMON=HERE/'monitor_manual001';MANREV=R/'e4c_latency_manual_overlap_review.json';MANREVMON=R/'monitor_e4c_latency_manual_overlap'
PLAN_SHA='83d617c735ca12332783b92350c0448106f34b2c0eb2bf6b374006d980c30ef1';SUP_SHA='7fb05c058125f038c1303eb23f722147500c27535337dfb7ad46641d45402636'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def sources():return [HERE/x for x in ('analysis_plan001.json','analysis_plan001_supplement.json','analysis_gate.py','analysis_core.py','analyze.py','test_analysis_core.py','test_analysis_gate.py','test_manual_overlap.py','manual_overlap.py','cpu_check.py')]
def source_hashes():return {str(x.resolve()):sha(x) for x in sources()}
def supervisor(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervisor identity or terminal')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output mismatch')
 return s,t
def prelaunch():
 for x in (P,SUP,CPU,CM/'started.json',CM/'terminal.json',PRE,PRE_SCRIPT,PRE_MON/'started.json',PRE_MON/'terminal.json'):
  if not x.is_file():raise ValueError('independent analysis prelaunch review required')
 supervisor(CM,HERE/'cpu_check.py');supervisor(PRE_MON,PRE_SCRIPT,PRE)
 c=read(CPU);p=read(PRE);actual=source_hashes()
 if sha(P)!=PLAN_SHA or sha(SUP)!=SUP_SHA or read(SUP).get('base_plan_sha256')!=PLAN_SHA or c.get('status')!='passed' or c.get('full_timing_values_opened') is not False or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual or importlib.metadata.version('numpy')!=c.get('numpy_version'):raise ValueError('CPU analysis/source proof invalid')
 expected={'status':'passed','independent_review':True,'scope':'latency_analysis_prelaunch','full_timing_values_opened':False,'errors':[],'source_sha256':actual,'plan_sha256':PLAN_SHA,'supplement_sha256':SUP_SHA,'cpu_proof_sha256':sha(CPU),'cpu_started_sha256':sha(CM/'started.json'),'cpu_terminal_sha256':sha(CM/'terminal.json'),'reviewer_sha256':sha(PRE_SCRIPT)}
 if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('independent analysis prelaunch proof invalid')
 return {'source_sha256':actual,'prelaunch_sha256':sha(PRE),'cpu_sha256':sha(CPU)}
def upstream():
 supervisor(FM,F/'full_verify.py',FORMAL);supervisor(IM,R/'review_e4c_latency_full001.py',IND);supervisor(MANMON,HERE/'manual_overlap.py');supervisor(MANREVMON,R/'review_e4c_latency_manual_overlap.py')
 f=read(FORMAL);i=read(IND);m=read(MAN);mr=read(MANREV);journal=(F/'full_001/journal.jsonl').resolve();formalann=(F/'full_001_annotations.jsonl').resolve();manualann=(HERE/'manual_overlap001/annotations.jsonl').resolve()
 if (f.get('status'),f.get('n_jobs'),f.get('n_generated'),f.get('GT_opened'),f.get('scoring_performed'),f.get('errors'))!=('passed',5940,5940,False,False,[]) or f.get('verifier_sha256')!=sha(F/'full_verify.py'):raise ValueError('formal acceptance invalid')
 if (i.get('status'),i.get('independent_review'),i.get('n_jobs'),i.get('n_generated'),i.get('GT_opened'),i.get('scoring_performed'),i.get('errors'))!=('passed',True,5940,5940,False,False,[]) or any(i.get(k)!=sha(p) for k,p in [('formal_proof_sha256',FORMAL),('formal_started_sha256',FM/'started.json'),('formal_terminal_sha256',FM/'terminal.json'),('run_started_sha256',RUNM/'started.json'),('run_terminal_sha256',RUNM/'terminal.json'),('health_sha256',RUNM/'health.jsonl'),('annotations_sha256',formalann),('reviewer_sha256',R/'review_e4c_latency_full001.py')]):raise ValueError('independent full review invalid')
 if (m.get('status'),m.get('jobs'),m.get('all_samples_retained'),m.get('GT_opened'),m.get('scoring_performed'))!=('passed',5940,True,False,False) or m.get('annotations_sha256')!=sha(manualann) or m.get('annotator_sha256')!=sha(HERE/'manual_overlap.py'):raise ValueError('manual annotation invalid')
 if (mr.get('status'),mr.get('independent_review'),mr.get('jobs_verified'),mr.get('GT_opened'),mr.get('scoring_performed'),mr.get('errors'))!=('passed',True,5940,False,False,[]) or any(mr.get(k)!=sha(p) for k,p in [('formal_proof_sha256',FORMAL),('manual_proof_sha256',MAN),('manual_annotations_sha256',manualann),('manual_started_sha256',MANMON/'started.json'),('manual_terminal_sha256',MANMON/'terminal.json'),('reviewer_sha256',R/'review_e4c_latency_manual_overlap.py')]):raise ValueError('independent manual review invalid')
 evidence=f.get('evidence_sha256',{})
 if any(evidence.get(str(p.resolve()))!=sha(p) for p in (journal,formalann,F/'full_001/frozen.json',RUNM/'started.json',RUNM/'terminal.json',RUNM/'health.jsonl')):raise ValueError('formal evidence changed')
 if any(sha(p)!=h for p,h in m.get('input_sha256',{}).items()) or m.get('input_sha256',{}).get(str(journal))!=sha(journal) or mr.get('journal_sha256')!=sha(journal):raise ValueError('manual inputs changed')
 e1=B/'e1/e1full001_completion.json';e1proof=read(e1);frozen=read(F/'full_001/frozen.json');inp=Path(e1proof['files']['inputs']['path']).resolve()
 if frozen.get('e1_acceptance_sha256')!=sha(e1) or sha(inp)!=e1proof['files']['inputs']['sha256']:raise ValueError('accepted DEV inputs changed')
 return {'formal_sha256':sha(FORMAL),'independent_sha256':sha(IND),'manual_sha256':sha(MAN),'manual_review_sha256':sha(MANREV),'journal_sha256':sha(journal),'formal_annotations_sha256':sha(formalann),'manual_annotations_sha256':sha(manualann),'e1_acceptance_sha256':sha(e1),'inputs_path':str(inp),'inputs_sha256':sha(inp),'namespace':m['time_namespace']}
def admit():return {'prelaunch':prelaunch(),'upstream':upstream()}
