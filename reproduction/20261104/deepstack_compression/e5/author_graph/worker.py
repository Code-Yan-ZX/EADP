"""Bounded no-model author-graph probe; GPU execution requires reviewed launcher."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(HERE))
import preflight as cpu

sha=cpu.sha

def save(path,obj):
    with Path(path).open('x') as f:
        json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def event(path,obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj)+'\n');f.flush();os.fsync(f.fileno())

def validate_plan(plan):
    if (plan.get('stage')!='author CUDA graph feasibility' or plan.get('budgets')!=[1,16,128,256,512]
        or plan.get('child_timeout_seconds')!=300 or not isinstance(plan.get('sha256'),dict)):
        raise ValueError('invalid fixed probe plan')
    required={str((HERE/x).resolve()) for x in ('PLAN.md','selector.py','probe.py','preflight.py','worker.py','launch.py','test_selector.py','test_worker.py')}
    if not required.issubset(plan['sha256']):raise ValueError('missing probe source')
    for path,digest in plan['sha256'].items():
        if sha(path)!=digest:raise ValueError('source or evidence changed: '+path)
    review_path=BASE/'e1_independent_review/selector_real32_run001_review.json'
    review=json.loads(review_path.read_text());mon=BASE/'e1_independent_review/monitor_selector_real32_run001'
    started=json.loads((mon/'started.json').read_text());terminal=json.loads((mon/'terminal.json').read_text())
    reviewer=BASE/'e1_independent_review/review_selector_real32_run001.py'
    required_upstream={str(x.resolve()) for x in (review_path,reviewer,mon/'started.json',mon/'terminal.json')}
    if (not required_upstream.issubset(plan['sha256']) or review.get('status')!='passed'
        or review.get('independent_review') is not True or review.get('images')!=32 or review.get('cases')!=96
        or review.get('formal_sha256')!=sha(cpu.PROOF) or review.get('array_sha256',{}).get(str(cpu.CASE))!=cpu.CASE_SHA
        or review.get('reviewer_sha256')!=sha(reviewer) or review.get('errors')!=[]
        or review.get('GT_opened') is not False or review.get('scoring_performed') is not False
        or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0
        or terminal.get('remaining_process_group')!=[] or terminal.get('worker_pid')!=started.get('worker_pid')):
        raise ValueError('independent real32 review incomplete')
    return cpu.check()

def require_review(plan_path):
    review=json.loads((HERE/'independent_prelaunch_review.json').read_text())
    terminal=Path(review['review_terminal_path']).resolve();reviewer=Path(review['reviewer_path']).resolve()
    if BASE/'e1_independent_review' not in terminal.parents or BASE/'e1_independent_review' not in reviewer.parents:
        raise ValueError('unexpected review location')
    started=terminal.parent/'started.json';s=json.loads(started.read_text());t=json.loads(terminal.read_text())
    cmd=s['command'];cwd=Path(s['cwd'])
    if (review.get('status')!='passed' or review.get('independent_review') is not True
        or review.get('plan_sha256')!=sha(plan_path) or review.get('preflight_sha256')!=sha(HERE/'preflight001.json')
        or review.get('reviewer_sha256')!=sha(reviewer) or review.get('review_started_sha256')!=sha(started)
        or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
        or t.get('worker_pid')!=s.get('worker_pid') or str(reviewer) not in [str((cwd/x).resolve()) for x in cmd]
        or '--out' not in cmd or (cwd/cmd[cmd.index('--out')+1]).resolve()!=HERE/'independent_prelaunch_review.json'):
        raise ValueError('independent review not completed')

def stop_child(p):
    if p is None or p.poll() is not None:return
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--child-k',type=int);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();plan_path=HERE/'plan001.json';plan=json.loads(plan_path.read_text());result=validate_plan(plan)
    if a.preflight_only:
        if a.child_k is not None or a.out.resolve()!=HERE/'preflight001.json':raise ValueError('bad preflight args')
        result.update(source_plan_sha256=sha(plan_path),case_count=1,model_loaded=False);save(a.out,result);return
    # This worker must be PID 1 of the retained Docker job, or its direct child.
    if (a.out.resolve()!=HERE/'run_smoke001/observations' or
        (a.child_k is None and os.getpid()!=1) or (a.child_k is not None and os.getppid()!=1)):
        raise ValueError('unexpected GPU worker parent or output path')
    launch=json.loads((HERE/'run_smoke001/launch.json').read_text())
    if launch.get('plan_sha256')!=sha(plan_path) or launch.get('stage')!='smoke':raise ValueError('launch identity')
    require_review(plan_path)
    if a.child_k is not None:
        if a.child_k not in plan['budgets']:raise ValueError('unexpected K')
        out=a.out/f'k_{a.child_k:03d}.jsonl'
        try:
            import torch,probe
            if Path(probe.__file__).resolve()!=HERE/'probe.py':raise ValueError('wrong probe import')
            if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
            case=torch.load(cpu.CASE,map_location='cpu',weights_only=True)
            author=probe.load_author(cpu.AUTHOR,cpu.AUTHOR_SHA,sha)
            okay=probe.run_one(a.child_k,case,author,lambda x:event(out,x))
            if not okay:raise ValueError('ordered selection mismatch')
        except BaseException as exc:
            event(out,{'event':'failed','k':a.child_k,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            raise
        return
    a.out.mkdir(exist_ok=False);event(a.out/'progress.jsonl',{'event':'worker_started'})
    child=None;statuses=[]
    def interrupted(sig,_frame):raise InterruptedError('controller received signal '+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        for k in plan['budgets']:
            event(a.out/'progress.jsonl',{'event':'k_started','k':k})
            command=[sys.executable,str(HERE/'worker.py'),'--child-k',str(k),'--out',str(a.out)]
            with (a.out/f'k_{k:03d}.stdout.log').open('x') as stdout,(a.out/f'k_{k:03d}.stderr.log').open('x') as stderr:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                try:rc=child.wait(timeout=plan['child_timeout_seconds']);timed_out=False
                except subprocess.TimeoutExpired:stop_child(child);rc=child.returncode;timed_out=True
                finally:stop_child(child)
            statuses.append({'k':k,'returncode':rc,'timed_out':timed_out})
            event(a.out/'progress.jsonl',{'event':'k_finished','k':k,'returncode':rc,'timed_out':timed_out})
            child=None
    except BaseException as exc:
        stop_child(child);event(a.out/'progress.jsonl',{'event':'worker_interrupted','error_type':type(exc).__name__,'error':str(exc)})
        raise
    save(a.out/'worker_complete.json',{'status':'attempts_recorded','cases':statuses,'GT_opened':False,'scoring_performed':False})
    if any(x['returncode']!=0 or x['timed_out'] for x in statuses):raise SystemExit(1)

if __name__=='__main__':main()
