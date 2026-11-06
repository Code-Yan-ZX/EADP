"""CPU-only gates for a future E4-A full DEV panel; never opens GT."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;E4=HERE.parent/'e4'
sys.path.insert(1,str(E4))
from identity_worker import check_e1,read_jsonl,select_panel,sha
REVIEW=HERE.parent/'e1_independent_review/e4smoke001_review.json'
REVIEW_SOURCE=HERE.parent/'e1_independent_review/review_e4smoke.py'
REVIEW_STARTED=HERE.parent/'e1_independent_review/monitor_e4smoke001/started.json'
REVIEW_TERMINAL=HERE.parent/'e1_independent_review/monitor_e4smoke001/terminal.json'

def validate_deadline(plan,now):
    path=Path(plan['budget_source'])
    if sha(path)!=plan['budget_source_sha256']:raise ValueError('three-day budget source changed')
    source=json.loads(path.read_text())
    if (source.get('budget_start_unix')!=plan['budget_start_unix'] or source.get('budget_end_unix')!=plan['budget_end_unix'] or
        source.get('budget_end_beijing')!=plan['budget_end_beijing'] or now<plan['budget_start_unix'] or
        now+plan['outer_timeout_seconds']>plan['budget_end_unix']):
        raise ValueError('insufficient remaining three-day budget')

def validate_smoke_review(proof_path,verify_terminal):
    review=json.loads(REVIEW.read_text());s=json.loads(REVIEW_STARTED.read_text());t=json.loads(REVIEW_TERMINAL.read_text())
    if (review.get('status')!='passed' or review.get('natural_verified')!=256 or review.get('forced_verified')!=8 or
        review.get('errors')!=[] or review.get('proof_sha256')!=sha(proof_path) or
        review.get('verifier_terminal_sha256')!=sha(verify_terminal) or review.get('reviewer_sha256')!=sha(REVIEW_SOURCE) or
        t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
        t.get('worker_pid')!=s.get('worker_pid')):raise ValueError('independent smoke review incomplete')
    cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    if ([(cwd/x).resolve() for x in cmd if Path(x).name==REVIEW_SOURCE.name]!=[REVIEW_SOURCE.resolve()] or
        cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=REVIEW.resolve()):
        raise ValueError('independent smoke review supervisor command mismatch')
    return {'proof_sha256':sha(REVIEW),'terminal_sha256':sha(REVIEW_TERMINAL)}

def validate_budget_supervisor(started,run,pid):
    plan=json.loads((HERE/'budget_plan.json').read_text())
    if (started.get('worker_pid')!=pid or started.get('process_group')!=pid or
        started.get('timeout_seconds')!=plan['outer_timeout_seconds'] or started.get('stale_seconds')!=plan['stale_seconds'] or
        not 0<started.get('snapshot_seconds',0)<=plan['snapshot_seconds'] or
        not 0<started.get('poll_seconds',0)<=30 or started.get('grace_seconds',0)<60 or
        started.get('progress_paths')!=[str((Path(run)/'generated.jsonl').resolve())]):
        raise ValueError('panel supervisor budget/progress configuration mismatch')


def validate_supervisor(started_path,terminal_path,script,output):
    started=json.loads(started_path.read_text());terminal=json.loads(terminal_path.read_text())
    if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or
        terminal.get('remaining_process_group')!=[] or not isinstance(started.get('worker_pid'),int) or
        terminal.get('worker_pid')!=started['worker_pid']):raise ValueError('supervisor did not exit cleanly')
    cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    scripts=[(cwd/x).resolve() for x in cmd if Path(x).name==script.name]
    if scripts!=[script.resolve()] or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd):
        raise ValueError('wrong supervised verifier')
    if (cwd/cmd[cmd.index('--out')+1]).resolve()!=output.resolve():raise ValueError('supervisor output mismatch')


def validate_smoke_gate(e1_path,smoke_dir,proof_path,started_path,terminal_path):
    validate_supervisor(started_path,terminal_path,E4/'verify_k256.py',proof_path)
    proof=json.loads(proof_path.read_text());run=smoke_dir.resolve()
    if (proof.get('status')!='passed' or proof.get('all_arms_validated')!=256 or proof.get('forced_arms_validated')!=8
        or proof.get('forced_decode_steps')!=32 or proof.get('GT_opened') is not False
        or proof.get('scoring_performed') is not False or proof.get('verifier_sha256')!=sha(E4/'verify_k256.py')):
        raise ValueError('complete independent E4 smoke acceptance required')
    files=proof.get('files_sha256',{});required=[e1_path,run/'launch.json',run/'exit.json',run/'frozen.json',
        run/'worker_complete.json',run/'records.jsonl',run/'forced_records.jsonl',run/'generated.jsonl']
    for path in required:
        if files.get(str(path.resolve()))!=sha(path):raise ValueError('smoke acceptance input changed or missing')
    arrays=proof.get('array_sha256',{})
    if len(arrays)!=264:raise ValueError('smoke arrays incomplete')
    for path,digest in arrays.items():
        p=Path(path)
        if p.parent.resolve()!=(run/'arrays').resolve() or sha(p)!=digest:raise ValueError('smoke array changed')
    compared=proof.get('compared',[])
    if len(compared)!=64 or len({(r['task'],str(r['qid']),r['arm']) for r in compared})!=64:
        raise ValueError('K256 smoke identity inventory incomplete')
    return proof


def validated_dev(e1_path,e1_started,e1_terminal):
    proof=check_e1(e1_path,e1_started,e1_terminal)
    from score import recheck_acceptance
    recheck_acceptance(e1_path)
    select_panel(proof)  # validates all E1 frozen images, messages and old-anchor proof
    rows=read_jsonl(proof['files']['inputs']['path'])
    counts={task:sum(r['task']==task for r in rows) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')}
    if counts!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} or len(rows)!=384:
        raise ValueError('full DEV inventory invalid')
    return proof,rows
