"""Fail-closed prospective oracle launch gate. Does not open references."""
import json,subprocess,time
from pathlib import Path
from cpu_validation import sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
APPROVAL=HERE/'runtime_approval.json'

def read(path):return json.loads(Path(path).read_text())
def supervised(started,terminal,script,out):
    s=read(started);t=read(terminal);cwd=Path(s.get('cwd','/'));cmd=s.get('command',[])
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
        not isinstance(s.get('worker_pid'),int) or s['worker_pid']<=0 or t.get('worker_pid')!=s['worker_pid'] or
        [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or
        cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):
        raise ValueError('review/CPU supervisor or output mismatch')

def check_budget(plan,now,require_future=True):
    source=Path(plan['budget_source']);original=read(source)
    if (sha(source)!=plan['budget_source_sha256'] or original['budget_start_unix']!=plan['budget_start_unix'] or
        original['budget_end_unix']!=plan['budget_end_unix'] or now<plan['budget_start_unix'] or
        (require_future and now+plan['outer_timeout_seconds']>plan['budget_end_unix']) or
        not 0<plan['question_checkpoint_seconds']<=plan['inner_timeout_seconds']<plan['outer_timeout_seconds'] or
        plan['snapshot_seconds']!=3600 or not plan['progress_stale_seconds']>plan['question_checkpoint_seconds'] or
        plan['questions']!=128 or plan['new_generations_per_question']!=6 or plan['maximum_new_generations']!=768):
        raise ValueError('invalid or exhausted user time budget')

def validate(*,host=False,require_future=True,preflight_only=False):
    if not APPROVAL.is_file():raise ValueError('oracle execution locked: independent approval absent')
    a=read(APPROVAL)
    required={'cpu_proof','cpu_started','cpu_terminal','review_proof','review_started','review_terminal'}
    if not preflight_only:required|={'preflight_proof','preflight_started','preflight_terminal'}
    if set(a)!=required:raise ValueError('incomplete oracle approval')
    planpath=HERE/'runtime_plan.json';plan=read(planpath);check_budget(plan,time.time(),require_future)
    if sha(plan['sample_path'])!=plan['sample_sha256'] or sha(plan['amendment_path'])!=plan['amendment_sha256']:
        raise ValueError('oracle sample/protocol amendment changed')
    cpu=Path(a['cpu_proof']);c=read(cpu);supervised(a['cpu_started'],a['cpu_terminal'],HERE/'cpu_validation.py',cpu.parent)
    if c.get('status')!='passed' or c.get('GT_opened') is not False or c.get('real_model_loaded') is not False or c.get('source_sha256_before')!=c.get('source_sha256_after'):
        raise ValueError('CPU proof failed')
    src=c['source_sha256_after']
    if any(sha(p)!=d for p,d in src.items()) or any(src.get(str(p.resolve()))!=sha(p) for p in HERE.glob('*.py')) or src.get(str(planpath))!=sha(planpath):
        raise ValueError('reviewed source inventory changed')
    if preflight_only:
        review=Path(a['review_proof']);expected=HERE/'independent_runtime_preflight.json'
        reviewer=ROOT/'e1_independent_review/review_m3oracle_runtime_preflight.py'
        if review.resolve()!=expected.resolve():raise ValueError('wrong provisional review')
        supervised(a['review_started'],a['review_terminal'],reviewer,review);r=read(review)
        if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or
            r.get('reviewer_sha256')!=sha(reviewer) or r.get('cpu_proof_sha256')!=sha(cpu) or
            r.get('cpu_started_sha256')!=sha(a['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
            r.get('plan_sha256')!=sha(planpath) or r.get('source_sha256')!=src or r.get('GT_opened') is not False or r.get('gpu_used') is not False):
            raise ValueError('provisional review mismatch')
        return {'approval_sha256':sha(APPROVAL),'plan_sha256':sha(planpath),'cpu_sha256':sha(cpu),'review_sha256':sha(review)}
    pp=Path(a['preflight_proof']);pre=read(pp);st=read(a['preflight_started']);te=read(a['preflight_terminal']);cmd=st.get('command',[]);cwd=Path(st.get('cwd','/'))
    pre_run=pp.stem.removeprefix('preflight_');pre_dir=HERE/('preflight_'+pre_run)
    if (te.get('reason')!='completed' or te.get('worker_returncode')!=0 or te.get('remaining_process_group')!=[] or te.get('worker_pid')!=st.get('worker_pid') or
        [(cwd/x).resolve() for x in cmd if Path(x).name=='preflight.py']!=[(HERE/'preflight.py').resolve()] or
        cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=pre_run or pre.get('status')!='passed' or
        pre.get('cpu_proof_sha256')!=sha(cpu) or pre.get('preflight_sha256')!=sha(HERE/'preflight.py') or
        pre.get('launch_sha256')!=sha(pre_dir/'launch.json') or pre.get('log_sha256')!=sha(pre_dir/'launcher.log') or
        pre.get('exit_sha256')!=sha(pre_dir/'exit.json') or pre.get('receipt_sha256')!=sha(pre_dir/'host_receipt.json') or pre.get('container_running') is not False or
        pre.get('container_pid')!=0 or pre.get('container_exit_code')!=0 or pre.get('result',{}).get('GT_opened') is not False or
        pre.get('result',{}).get('predictions_opened') is not False or pre.get('result',{}).get('cuda_initialized') is not False or
        pre.get('result',{}).get('model_loaded') is not False or pre.get('result',{}).get('inputs_verified')!=384 or
        pre.get('result',{}).get('sample_verified')!=128 or pre.get('result',{}).get('weights_sha256_verified')!=4 or
        pre.get('result',{}).get('prediction_journal_byte_hash_deferred') is not True):
        raise ValueError('real CPU container preflight incomplete')
    ex=read(pre_dir/'exit.json');launch=read(pre_dir/'launch.json');docker_cmd=launch.get('command',[])
    if (ex!=pre.get('exit') or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or
        ex.get('container_stopped_verified') is not True or ex.get('interrupted') is not None or ex.get('timed_out') is not False or
        ex.get('errors')!=[] or launch.get('host_receipt')!={'path':str((pre_dir/'host_receipt.json').resolve()),'sha256':sha(pre_dir/'host_receipt.json')} or
        launch.get('cpu_proof_sha256')!=sha(cpu) or launch.get('worker_sha256')!=sha(HERE/'runtime_worker.py') or
        docker_cmd.count(str(HERE/'runtime_worker.py'))!=1 or docker_cmd.count('--preflight-only')!=1 or 'CUDA_VISIBLE_DEVICES=' not in docker_cmd or
        any('/dev/nvidia' in str(x) for x in docker_cmd)):
        raise ValueError('preflight launch/exit not bound')
    if host:
        q=subprocess.run(['docker','inspect',pre['container_id']],capture_output=True,text=True,timeout=20)
        if q.returncode:raise ValueError('preflight container cannot be inspected')
        v=json.loads(q.stdout)[0];state=v['State']
        if v['Id']!=pre['container_id'] or state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0:
            raise ValueError('preflight container not confirmed stopped')
    review=Path(a['review_proof']);expected_review=HERE/'independent_runtime_review.json';reviewer=ROOT/'e1_independent_review/review_m3oracle_runtime.py'
    if review.resolve()!=expected_review.resolve():raise ValueError('wrong independent oracle review')
    supervised(a['review_started'],a['review_terminal'],reviewer,review)
    r=read(review)
    if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('reviewer_sha256')!=sha(reviewer) or
        r.get('cpu_proof_sha256')!=sha(cpu) or r.get('cpu_started_sha256')!=sha(a['cpu_started']) or r.get('cpu_terminal_sha256')!=sha(a['cpu_terminal']) or
        r.get('preflight_sha256')!=sha(pp) or r.get('preflight_started_sha256')!=sha(a['preflight_started']) or r.get('preflight_terminal_sha256')!=sha(a['preflight_terminal']) or
        r.get('plan_sha256')!=sha(planpath) or r.get('source_sha256')!=src or r.get('GT_opened') is not False or r.get('gpu_used') is not False):
        raise ValueError('independent runtime review mismatch')
    return {'approval_sha256':sha(APPROVAL),'plan_sha256':sha(planpath),'cpu_sha256':sha(cpu),'preflight_sha256':sha(pp),'review_sha256':sha(review)}
