"""CPU-only gates for a future E4-A full DEV panel; never opens GT."""
import json
from pathlib import Path
from identity_worker import HERE,check_e1,read_jsonl,select_panel,sha


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
    validate_supervisor(started_path,terminal_path,HERE/'verify_k256.py',proof_path)
    proof=json.loads(proof_path.read_text());run=smoke_dir.resolve()
    if (proof.get('status')!='passed' or proof.get('all_arms_validated')!=256 or proof.get('forced_arms_validated')!=8
        or proof.get('forced_decode_steps')!=32 or proof.get('GT_opened') is not False
        or proof.get('scoring_performed') is not False or proof.get('verifier_sha256')!=sha(HERE/'verify_k256.py')):
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
