"""Fixed E1 and E4-smoke admission for native identity diagnosis."""
import json
import sys
from pathlib import Path
from native_contract import HERE,E4
sys.path.insert(0,str(E4))
from identity_worker import sha
from panel_gate import validated_dev,validate_smoke_gate

ROOT=HERE.parent
REVIEW=ROOT/'e1_independent_review/e4smoke001_review.json'
REVIEW_SOURCE=ROOT/'e1_independent_review/review_e4smoke.py'
REVIEW_TERMINAL=ROOT/'e1_independent_review/monitor_e4smoke001/terminal.json'
REVIEW_STARTED=ROOT/'e1_independent_review/monitor_e4smoke001/started.json'

def check_gates(a):
    e1,_=validated_dev(a.e1_acceptance,a.e1_started,a.e1_terminal)
    validate_smoke_gate(a.e1_acceptance,a.smoke_dir,a.smoke_proof,a.smoke_started,a.smoke_terminal)
    r=json.loads(REVIEW.read_text());t=json.loads(REVIEW_TERMINAL.read_text());s=json.loads(REVIEW_STARTED.read_text())
    if (r.get('status')!='passed' or r.get('natural_verified')!=256 or r.get('forced_verified')!=8 or r.get('errors')!=[]
        or r.get('proof_sha256')!=sha(a.smoke_proof) or r.get('verifier_terminal_sha256')!=sha(a.smoke_terminal)
        or r.get('reviewer_sha256')!=sha(REVIEW_SOURCE) or t.get('reason')!='completed' or t.get('worker_returncode')!=0
        or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid')):
        raise ValueError('independent E4 smoke review incomplete')
    cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    scripts=[(cwd/x).resolve() for x in cmd if Path(x).name==REVIEW_SOURCE.name]
    if scripts!=[REVIEW_SOURCE.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=REVIEW.resolve():
        raise ValueError('independent review supervisor command mismatch')
    return e1
