"""Require independent CPU/source review before v2 oracle scoring."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;ORIGINAL=ROOT/'m3_oracle_analysis';V2=ROOT/'m3_oracle_verify_v2'
sys.path.insert(0,str(ROOT/'m3_oracle_exec'))
import runtime_gate
def sha(path):return runtime_gate.sha(path)
def sources():
 return [HERE/x for x in ('score_gate_v2.py','prelaunch_gate.py','score_driver.py','cpu_validation.py','test_prelaunch.py')]+[ORIGINAL/x for x in ('score_oracle.py','score_gate.py','analysis_gate.py','paired.py','plan.json')]+[V2/'verify.py',V2/'environment_002_v2.json',V2/'acceptance_binding_002.json',V2/'verify_002_sources/manifest.json']
def validate():
 cpu=HERE/'cpu_check_002/verification.json';cs=HERE/'monitor_cpu002/started.json';ct=HERE/'monitor_cpu002/terminal.json';review=HERE/'independent_review.json';reviewer=ROOT/'e1_independent_review/review_m3oracle_score_v2.py';rs=ROOT/'e1_independent_review/monitor_m3oracle_score_v2/started.json';rt=rs.parent/'terminal.json'
 runtime_gate.supervised(cs,ct,HERE/'cpu_validation.py',cpu.parent);runtime_gate.supervised(rs,rt,reviewer,review)
 c=json.loads(cpu.read_text());r=json.loads(review.read_text());actual={str(p.resolve()):sha(p) for p in sources()}
 if (c.get('status')!='passed' or c.get('returncode')!=0 or c.get('source_sha256_before')!=actual or c.get('source_sha256_after')!=actual or c.get('GT_opened') is not False or c.get('predictions_parsed') is not False or
     r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or r.get('predictions_parsed') is not False or r.get('source_sha256')!=actual or
     r.get('cpu_proof_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(cs) or r.get('cpu_terminal_sha256')!=sha(ct) or r.get('reviewer_sha256')!=sha(reviewer)):
  raise ValueError('v2 scoring CPU or independent source review missing')
 return {'review_sha256':sha(review),'cpu_sha256':sha(cpu),'source_sha256':actual}
