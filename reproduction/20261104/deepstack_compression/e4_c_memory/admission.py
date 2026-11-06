"""Fail-closed admission for a future retained, no-GPU memory worker preflight.

This does not authorize a memory GPU run. The independent review artifact and
its actual supervisor exit are mandatory, and missing proof fails immediately.
The controller and worker execute separately from the host sampler because of
same-named contract.py modules.
"""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;LATENCY=BASE/'e4_c_latency';REVIEW=BASE/'e1_independent_review'
sys.path.insert(0,str(LATENCY))
from identity_gate import sha,supervised
from latency_admission import terminal,cpu_container

CPU=HERE/'cpu_check002/verification.json';CS=HERE/'monitor_cpu_check002/started.json';CT=HERE/'monitor_cpu_check002/terminal.json'
IL=HERE/'import_preflight002/launch.json';IE=HERE/'import_preflight002/exit.json';IS=HERE/'monitor_import_preflight002/started.json';IT=HERE/'monitor_import_preflight002/terminal.json'
REVIEW_PROOF=REVIEW/'e4c_memory_worker_preflight_review.json';REVIEWER=REVIEW/'review_e4c_memory_worker_preflight.py';MONITOR=REVIEW/'monitor_e4c_memory_worker_preflight'

def sources():
 from latency_worker import sources as latency_sources
 names=('admission.py','preflight_entry.py','preflight_launch.py','test_admission.py','preflight_import.py','import_probe.py','sampler_probe.py','cpu_check.py','worker_preflight.py','test_worker_preflight.py','worker.py','test_worker.py','plan.py','test_plan.py','contract.py','test_contract.py','sampler.py','barrier.py','test_barrier.py','host_evidence.py','test_host_evidence.py','sampler_io.py','test_sampler_io.py','README.md')
 return list(latency_sources())+[HERE/n for n in names]

def admit():
 if not REVIEW_PROOF.is_file():raise ValueError('memory CPU container preflight locked: independent review absent')
 supervised(MONITOR/'started.json',MONITOR/'terminal.json',REVIEWER,REVIEW_PROOF)
 review=json.loads(REVIEW_PROOF.read_text());host=HERE/'host_preflight001_proof.json';host_proof=json.loads(host.read_text());cpu=HERE/'cpu_proof012.json';cpu_proof=json.loads(cpu.read_text())
 frozen={str(p):sha(p) for p in sources()}
 if (review.get('status')!='passed' or review.get('scope')!='memory_cpu_container_preflight_only' or review.get('independent_review') is not True or
     review.get('GT_opened') is not False or review.get('scoring_performed') is not False or review.get('errors')!=[] or
     review.get('reviewer_sha256')!=sha(REVIEWER) or review.get('source_sha256')!=frozen or
     review.get('host_proof_sha256')!=sha(host) or review.get('cpu_proof_sha256')!=sha(cpu) or any(review.get(name+'_sha256')!=sha(path) for name,path in (('suite_proof',CPU),('suite_started',CS),('suite_terminal',CT),('import_launch',IL),('import_exit',IE),('import_started',IS),('import_terminal',IT)))):raise ValueError('memory preflight review mismatch')
 cs=terminal(CS,CT);ins=terminal(IS,IT);suite=json.loads(CPU.read_text());launch=json.loads(IL.read_text());exit_=json.loads(IE.read_text())
 for started,script in ((cs,HERE/'cpu_check.py'),(ins,HERE/'preflight_import.py')):
  cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
  if [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!='002':raise ValueError('memory CPU supervisor command mismatch')
 cpu_container(launch,exit_,ins,HERE/'import_probe.py')
 if (suite.get('status')!='passed' or suite.get('tests')!=23 or suite.get('source_sha256_before')!=frozen or suite.get('source_sha256_after')!=frozen or launch.get('source_sha256')!=frozen):raise ValueError('memory CPU suite or import source mismatch')
 for proof in (host_proof,cpu_proof):
  if proof.get('status')!='passed' or any(sha(p)!=digest for p,digest in proof.get('files_sha256',{}).items()):raise ValueError('memory earlier CPU proof mismatch')
 if any(sha(p)!=digest for p,digest in host_proof.get('source_sha256',{}).items()):raise ValueError('memory host preflight source changed')
 return {'scope':'memory_cpu_container_preflight_only','review_sha256':sha(REVIEW_PROOF),'source_sha256':frozen,'host_proof_sha256':sha(host),'cpu_proof_sha256':sha(cpu)}
