"""Require the independently reviewed native K1024 diagnostic and real exits."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
E4=ROOT/'e4';NATIVE=ROOT/'e4_native';REVIEW=ROOT/'e1_independent_review'
sys.path.insert(1,str(E4));sys.path.insert(1,str(NATIVE))
from identity_worker import sha
from panel_v2_imports import load_local
validate_supervisor=load_local('panel_gate').validate_supervisor

PROOF=NATIVE/'native_001_verification.json';RUN=NATIVE/'native_001'
VERIFIER_STARTED=NATIVE/'monitor_verify_native001/started.json';VERIFIER_TERMINAL=NATIVE/'monitor_verify_native001/terminal.json'
RUN_STARTED=NATIVE/'monitor_native001/started.json';RUN_TERMINAL=NATIVE/'monitor_native001/terminal.json'
REVIEW_PROOF=REVIEW/'e4native001_review.json';REVIEW_SOURCE=REVIEW/'review_e4native.py'
REVIEW_STARTED=REVIEW/'monitor_e4native001/started.json';REVIEW_TERMINAL=REVIEW/'monitor_e4native001/terminal.json'

def check_native(*, inspect_live=True):
    validate_supervisor(VERIFIER_STARTED,VERIFIER_TERMINAL,NATIVE/'native_verify.py',PROOF)
    proof=json.loads(PROOF.read_text());review=json.loads(REVIEW_PROOF.read_text())
    if (proof.get('status')!='passed' or proof.get('stage')!='stock_native_k1024' or proof.get('tasks')!=3 or
        proof.get('pairs')!=6 or proof.get('natural_pairs')!=3 or proof.get('forced_pairs')!=3 or
        proof.get('forced_decode_steps')!=32 or proof.get('kv_fingerprints_compared') is not True or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or
        proof.get('verifier_sha256')!=sha(NATIVE/'native_verify.py')):
        raise ValueError('native formal proof incomplete')
    files=proof.get('files_sha256',{})
    for path in (RUN_STARTED,RUN_TERMINAL,RUN/'launch.json',RUN/'exit.json',RUN/'frozen.json',
                 RUN/'worker_complete.json',RUN/'records.jsonl',RUN/'generated.jsonl',RUN/'comparisons.jsonl'):
        if files.get(str(path.resolve()))!=sha(path):raise ValueError('native evidence changed: '+str(path))
    for path,digest in files.items():
        if sha(path)!=digest:raise ValueError('native proof file changed: '+path)
    arrays=proof.get('array_sha256',{})
    if len(arrays)!=12:raise ValueError('native array evidence incomplete')
    for path,digest in arrays.items():
        if Path(path).parent.resolve()!=(RUN/'arrays').resolve() or sha(path)!=digest:raise ValueError('native array changed')
    if (review.get('status')!='passed' or review.get('pairs_expected')!=6 or review.get('pairs_verified')!=6 or
        review.get('errors')!=[] or review.get('formal_proof_sha256')!=sha(PROOF) or
        review.get('formal_verifier_terminal_sha256')!=sha(VERIFIER_TERMINAL) or review.get('reviewer_sha256')!=sha(REVIEW_SOURCE)):
        raise ValueError('independent native review incomplete')
    started=json.loads(REVIEW_STARTED.read_text());term=json.loads(REVIEW_TERMINAL.read_text())
    if (term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or
        term.get('worker_pid')!=started.get('worker_pid')):raise ValueError('native review supervisor failed')
    cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if [(cwd/x).resolve() for x in cmd if Path(x).name==REVIEW_SOURCE.name]!=[REVIEW_SOURCE.resolve()]:
        raise ValueError('wrong native review supervisor command')
    if inspect_live:
        from native_verify import terminal
        terminal(RUN.resolve(),json.loads(RUN_STARTED.read_text()),json.loads(RUN_TERMINAL.read_text()),
                 json.loads((RUN/'launch.json').read_text()),json.loads((RUN/'exit.json').read_text()))
    return {'proof_sha256':sha(PROOF),'proof_terminal_sha256':sha(VERIFIER_TERMINAL),
            'review_sha256':sha(REVIEW_PROOF),'review_terminal_sha256':sha(REVIEW_TERMINAL)}
