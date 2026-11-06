"""Two independent reviews gate full-latency CPU preflight and GPU launch."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PILOT=BASE/'e4_c_latency';REVIEW=BASE/'e1_independent_review'
sys.path.insert(0,str(PILOT))
from identity_gate import sha,supervised
from pilot_gate import pilot_gate
from latency_admission import terminal,cpu_container

def sources():
 from latency_worker import sources as pilot_sources
 names=('budget_plan.json','pilot_gate.py','full_admission.py','full_worker.py','full_launch.py','full_verify.py','container_identity.py','preflight_worker.py','preflight_import.py','import_probe.py','cpu_check.py','test_full.py')
 return list(pilot_sources())+[HERE/n for n in names]

def run_id(started,script,prefix,output):
 cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 if ([(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--run-id')!=1 or cmd.index('--run-id')+1>=len(cmd) or Path(output).resolve().parent!=HERE/(prefix+cmd[cmd.index('--run-id')+1])):raise ValueError('full preflight command mismatch')

def admit(*,preflight_only=False,host=True):
 if preflight_only:
  proof=REVIEW/'e4c_latency_full_worker_preflight_review.json';reviewer=REVIEW/'review_e4c_latency_full_worker_preflight.py';mon=REVIEW/'monitor_e4c_latency_full_worker_preflight';scope='full_latency_worker_cpu_preflight'
 else:
  proof=REVIEW/'e4c_latency_full_prelaunch_review.json';reviewer=REVIEW/'review_e4c_latency_full_prelaunch.py';mon=REVIEW/'monitor_e4c_latency_full_prelaunch';scope='full_latency_launch'
 if not proof.is_file():raise ValueError('full latency locked: independent review missing')
 supervised(mon/'started.json',mon/'terminal.json',reviewer,proof);review=json.loads(proof.read_text());pilot=pilot_gate(host=host);expected={str(p):sha(p) for p in sources()}
 if (review.get('status')!='passed' or review.get('scope')!=scope or review.get('independent_review') is not True or review.get('GT_opened') is not False or review.get('scoring_performed') is not False or review.get('errors')!=[] or
     review.get('reviewer_sha256')!=sha(reviewer) or review.get('source_sha256')!=expected or review.get('pilot_identity')!=pilot):raise ValueError('full latency review mismatch')
 names=['cpu_proof','cpu_started','cpu_terminal','import_launch','import_exit','import_started','import_terminal']
 if not preflight_only:names+=['worker_launch','worker_exit','worker_started','worker_terminal','worker_result']
 for name in names:
  p=Path(review[name+'_path']).resolve()
  if not p.is_relative_to(HERE) or review.get(name+'_sha256')!=sha(p):raise ValueError('full review evidence mismatch '+name)
 cs=terminal(review['cpu_started_path'],review['cpu_terminal_path']);is_=terminal(review['import_started_path'],review['import_terminal_path'])
 run_id(cs,HERE/'cpu_check.py','cpu_check',review['cpu_proof_path']);run_id(is_,HERE/'preflight_import.py','import_preflight',review['import_launch_path'])
 cp=json.loads(Path(review['cpu_proof_path']).read_text());il=json.loads(Path(review['import_launch_path']).read_text());ie=json.loads(Path(review['import_exit_path']).read_text());cpu_container(il,ie,is_,HERE/'import_probe.py')
 if (cp.get('status')!='passed' or cp.get('source_sha256_before')!=expected or cp.get('source_sha256_after')!=expected or il.get('source_sha256')!=expected or Path(review['import_exit_path']).resolve().parent!=Path(review['import_launch_path']).resolve().parent):raise ValueError('full CPU/import evidence mismatch')
 if not preflight_only:
  ws=terminal(review['worker_started_path'],review['worker_terminal_path']);wl=json.loads(Path(review['worker_launch_path']).read_text());we=json.loads(Path(review['worker_exit_path']).read_text());wr=json.loads(Path(review['worker_result_path']).read_text());run_id(ws,HERE/'preflight_worker.py','worker_preflight',review['worker_launch_path']);cpu_container(wl,we,ws,HERE/'full_worker.py')
  previous=admit(preflight_only=True,host=host);cmd=wl.get('command',[]);out=Path(review['worker_launch_path']).resolve().parent
  if (wl.get('approval_bundle')!=previous or wl.get('source_sha256')!=expected or cmd.count('--preflight-only')!=1 or cmd.count('--out')!=1 or Path(cmd[cmd.index('--out')+1]).resolve()!=out or
      Path(review['worker_exit_path']).resolve().parent!=out or Path(review['worker_result_path']).resolve()!=out/'worker_preflight.json' or wr.get('status')!='passed' or wr.get('cuda_initialized') is not False or wr.get('model_loaded') is not False or wr.get('predictions') is not False or wr.get('jobs')!=5940 or wr.get('full_source_sha256')!=expected or wr.get('pilot_identity')!=pilot or len(wr.get('weight_stats_before',{}))!=4):raise ValueError('full worker CPU preflight mismatch')
 return {'scope':scope,'review_sha256':sha(proof),'review_started_sha256':sha(mon/'started.json'),'review_terminal_sha256':sha(mon/'terminal.json'),'pilot_identity':pilot,'source_sha256':expected,'preflight_only':preflight_only}
