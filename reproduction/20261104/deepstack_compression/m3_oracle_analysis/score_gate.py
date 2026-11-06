"""Fail closed before decoding any oracle predictions or source answers."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXEC=ROOT/'m3_oracle_exec'
sys.path.insert(0,str(EXEC))
import runtime_gate,verify,e1_admission

def read(path):return json.loads(Path(path).read_text())
def gate(a):
    run=a.run.resolve();proof=a.proof.resolve();started=a.verify_started.resolve();terminal=a.verify_terminal.resolve()
    runtime_gate.supervised(started,terminal,EXEC/'verify.py',proof)
    result=read(proof)
    if (result.get('status')!='passed' or result.get('stage')!='oracle_full' or result.get('n_questions')!=128 or
        result.get('new_generations')!=768 or result.get('scoring_performed') is not False or
        result.get('reference_sources_opened_by_verifier') is not False or result.get('verifier_sha256')!=runtime_gate.sha(EXEC/'verify.py')):
        raise ValueError('full oracle acceptance required')
    reviewer=ROOT/'e1_independent_review/review_m3oracle_panel.py';review=read(a.review)
    runtime_gate.supervised(a.review_started,a.review_terminal,reviewer,a.review)
    if (review.get('status')!='passed' or review.get('n_questions')!=128 or review.get('new_generations')!=768 or
        review.get('errors')!=[] or review.get('scoring_performed') is not False or review.get('reference_sources_opened') is not False or
        review.get('formal_proof_sha256')!=runtime_gate.sha(proof) or review.get('formal_started_sha256')!=runtime_gate.sha(started) or
        review.get('formal_terminal_sha256')!=runtime_gate.sha(terminal) or review.get('frozen_sha256')!=runtime_gate.sha(run/'frozen.json') or
        review.get('host_receipt_sha256')!=runtime_gate.sha(run/'host_receipt.json') or review.get('reviewer_sha256')!=runtime_gate.sha(reviewer)):
        raise ValueError('independent oracle review not complete')
    expected={'run_start_sha256':a.run_started,'run_terminal_sha256':a.run_terminal,'launch_sha256':run/'launch.json',
              'exit_sha256':run/'exit.json','panel_events_sha256':run/'panel/panel_events.jsonl',
              'worker_complete_sha256':run/'panel/worker_complete.json'}
    if any(result.get(k)!=runtime_gate.sha(v) for k,v in expected.items()):raise ValueError('accepted oracle evidence changed')
    files=result.get('result_files_sha256',{})
    if len(files)!=128:raise ValueError('accepted question inventory incomplete')
    for path,digest in files.items():
        p=Path(path)
        if p.parent.resolve()!=(run/'panel').resolve() or runtime_gate.sha(p)!=digest:raise ValueError('question journal changed')
    approval=runtime_gate.validate(host=True,require_future=False)
    if result.get('approval_sha256')!=approval['approval_sha256']:raise ValueError('runtime approval changed')
    verify.check_terminal(run,read(a.run_started),read(a.run_terminal),read(run/'launch.json'),read(run/'exit.json'),read(EXEC/'runtime_plan.json'))
    return result
