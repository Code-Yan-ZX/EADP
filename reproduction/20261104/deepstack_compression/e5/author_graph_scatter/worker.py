"""Per-process, supervised CUDA capture diagnosis; review/scheduling required."""
import argparse,importlib.util,json,os,signal,subprocess,sys,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
PRIOR=BASE/'e5/author_graph'
sys.path[:0]=[str(HERE),str(PRIOR)]
import preflight as cpu
import transform
spec=importlib.util.spec_from_file_location('author_graph_frozen_controller',PRIOR/'worker.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common);common.HERE=HERE
sys.path.insert(0,str(HERE))
sha=cpu.sha;save=common.save;event=common.event;stop_child=common.stop_child
CASES=('index_assign','scatter_assign','index_read','boolean_mask_assign','k_001','k_016','k_128','k_256','k_512')

def validate_plan(plan):
    if (plan.get('stage')!='author scatter graph diagnosis' or tuple(plan.get('cases',()))!=CASES
        or plan.get('child_timeout_seconds')!=300 or plan.get('allocator_fraction')!=.6
        or plan.get('k512_min_free_bytes')!=32*1024**3 or plan.get('other_min_free_bytes')!=12*1024**3):
        raise ValueError('invalid fixed diagnostic plan')
    required={str((HERE/x).resolve()) for x in ('PLAN.md','transform.py','primitive.py','probe.py','worker.py','launch.py','test_transform.py','test_worker.py','cpu_real.py')}
    if not required.issubset(plan.get('sha256',{})):raise ValueError('missing source')
    for path,digest in plan['sha256'].items():
        if sha(path)!=digest:raise ValueError('source or evidence changed: '+path)
    result=cpu.check()
    real=json.loads((HERE/'cpu_real001.json').read_text());terminal=json.loads((HERE/'monitor_cpu_real001/terminal.json').read_text())
    if (real.get('status')!='passed' or real.get('cpu_original_matches_candidate') is not True
        or real.get('cuda_initialized') is not False or real.get('input_sha256')!=cpu.CASE_SHA
        or real.get('author_sha256')!=transform.AUTHOR_SHA or real.get('transform_sha256')!=sha(HERE/'transform.py')
        or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]):
        raise ValueError('real CPU original/candidate evidence missing')
    _,metadata=transform.build(transform.load_author())
    if metadata.get('replacement_count')!=1:raise ValueError('unexpected source transformation')
    return result

def run_child(name,out,plan):
    path=out/(name+'.jsonl')
    try:
        import torch
        if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
        free,total=torch.cuda.mem_get_info();minimum=plan['k512_min_free_bytes'] if name=='k_512' else plan['other_min_free_bytes']
        event(path,{'event':'gpu_budget','case':name,'free_bytes':free,'total_bytes':total,
                    'minimum_free_bytes':minimum,'allocator_fraction':plan['allocator_fraction'],
                    'limitation':'CUDA context/driver and graph metadata may exceed PyTorch allocator fraction'})
        if free<minimum:raise MemoryError('insufficient device free memory for planned case')
        torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'],0)
        if name.startswith('k_'):
            import probe
            if Path(probe.__file__).resolve()!=HERE/'probe.py':raise ValueError('wrong probe path')
            case=torch.load(cpu.CASE,map_location='cpu',weights_only=True)
            module=transform.load_author();candidate,metadata=transform.build(module)
            event(path,{'event':'candidate_built','case':name,'transform':metadata})
            if not probe.run_one(int(name[2:]),case,module._greed_select_impl,candidate,lambda x:event(path,x)):
                raise ValueError('ordered selection mismatch')
        else:
            import primitive
            if Path(primitive.__file__).resolve()!=HERE/'primitive.py':raise ValueError('wrong primitive path')
            primitive.run(name,lambda x:event(path,x))
    except BaseException as exc:
        event(path,{'event':'failed','case':name,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--child-case',choices=CASES);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan_path=HERE/'plan001.json';plan=json.loads(plan_path.read_text());result=validate_plan(plan)
    if a.preflight_only:
        if a.child_case or a.out.resolve()!=HERE/'preflight001.json':raise ValueError('bad preflight args')
        result.update(source_plan_sha256=sha(plan_path),case_count=1,model_loaded=False);save(a.out,result);return
    if (a.out.resolve()!=HERE/'run_smoke001/observations' or
        (a.child_case is None and os.getpid()!=1) or (a.child_case is not None and os.getppid()!=1)):
        raise ValueError('unexpected GPU worker parent or output')
    launch=json.loads((HERE/'run_smoke001/launch.json').read_text())
    if launch.get('plan_sha256')!=sha(plan_path) or launch.get('stage')!='smoke':raise ValueError('launch identity')
    common.require_review(plan_path)
    if a.child_case:run_child(a.child_case,a.out,plan);return
    a.out.mkdir(exist_ok=False);event(a.out/'progress.jsonl',{'event':'worker_started'});child=None;statuses=[]
    def interrupted(sig,_frame):raise InterruptedError('controller received signal '+str(sig))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        for name in CASES:
            event(a.out/'progress.jsonl',{'event':'case_started','case':name})
            command=[sys.executable,str(HERE/'worker.py'),'--child-case',name,'--out',str(a.out)]
            with (a.out/(name+'.stdout.log')).open('x') as stdout,(a.out/(name+'.stderr.log')).open('x') as stderr:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr)
                try:rc=child.wait(timeout=plan['child_timeout_seconds']);timed_out=False
                except subprocess.TimeoutExpired:stop_child(child);rc=child.returncode;timed_out=True
                finally:stop_child(child)
            statuses.append({'case':name,'returncode':rc,'timed_out':timed_out})
            event(a.out/'progress.jsonl',{'event':'case_finished','case':name,'returncode':rc,'timed_out':timed_out});child=None
    except BaseException as exc:
        stop_child(child);event(a.out/'progress.jsonl',{'event':'worker_interrupted','error_type':type(exc).__name__,'error':str(exc)});raise
    # This is a diagnostic: failed captures remain recorded outcomes, not passes.
    save(a.out/'worker_complete.json',{'status':'attempts_recorded','cases':statuses,'GT_opened':False,'scoring_performed':False,
         'interpretation':'inspect each case; process exit0 does not mean capture or identity passed'})

if __name__=='__main__':main()
