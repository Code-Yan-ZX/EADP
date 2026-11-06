"""Reviewed, bounded three-process 32-image component probe; no model/GT."""
import argparse,importlib.util,json,os,signal,subprocess,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];PRIOR=BASE/'e5/author_graph'
sys.path[:0]=[str(HERE),str(PRIOR)]
import inventory
if Path(inventory.__file__).resolve()!=HERE/'inventory.py':raise ValueError('wrong inventory import')
spec=importlib.util.spec_from_file_location('fixed_author_graph_controller',PRIOR/'worker.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common);common.HERE=HERE
sys.path.insert(0,str(HERE))
sha=inventory.sha;save=common.save;event=common.event;stop_child=common.stop_child
CASES=('k_128','k_256','k_512')

def validate_plan(plan,*,hash_archives=True):
    if (plan.get('stage')!='author scatter graph 32-image identity' or tuple(plan.get('cases',()))!=CASES
        or plan.get('child_timeout_seconds')!=600 or plan.get('allocator_fraction')!=.6
        or plan.get('k512_min_free_bytes')!=32*1024**3 or plan.get('other_min_free_bytes')!=12*1024**3):
        raise ValueError('invalid frozen schedule or resource budget')
    required={str((HERE/x).resolve()) for x in ('PLAN.md','inventory.py','freeze.py','probe.py','worker.py','launch.py','verify.py','test_inventory.py','test_worker.py','test_verify.py','archives002.json','monitor_freeze002/started.json','monitor_freeze002/terminal.json')}
    if not required.issubset(plan.get('sha256',{})):raise ValueError('missing source')
    for path,digest in plan['sha256'].items():
        if sha(path)!=digest:raise ValueError('source or evidence changed: '+path)
    rows=inventory.inventory(hash_files=hash_archives)
    if rows!=plan.get('case_manifest'):raise ValueError('fixed archive inventory mismatch')
    scatter=inventory.require_scatter_review()
    archive=json.loads((HERE/'archives002.json').read_text())
    started=json.loads((HERE/'monitor_freeze002/started.json').read_text());terminal=json.loads((HERE/'monitor_freeze002/terminal.json').read_text())
    command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
    if (archive.get('status')!='passed' or archive.get('cases')!=rows or archive.get('case_count')!=96
        or archive.get('inventory_sha256')!=sha(HERE/'inventory.py') or archive.get('freezer_sha256')!=sha(HERE/'freeze.py')
        or archive.get('scatter_review_sha256')!=scatter['independent_review_sha256']
        or archive.get('gpu_used') is not False or archive.get('GT_opened') is not False
        or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]
        or terminal.get('worker_pid')!=started.get('worker_pid') or len(command)<2 or (cwd/command[1]).resolve()!=HERE/'freeze.py'):
        raise ValueError('archive freeze evidence mismatch')
    return {'status':'passed','source_count':len(plan['sha256']),'case_count':len(rows),
            'scatter_formal_sha256':scatter['formal_sha256'],'scatter_review_sha256':scatter['independent_review_sha256'],
            'cuda_initialized':False,'GT_opened':False,'scoring_performed':False,'model_loaded':False,
            'archive_hashes_checked':hash_archives}

def run_child(name,out,plan):
    path=out/(name+'.jsonl')
    try:
        import torch
        if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
        free,total=torch.cuda.mem_get_info();minimum=plan['k512_min_free_bytes'] if name=='k_512' else plan['other_min_free_bytes']
        event(path,{'event':'gpu_budget','case':name,'free_bytes':free,'total_bytes':total,'minimum_free_bytes':minimum,
                    'allocator_fraction':plan['allocator_fraction'],
                    'limitation':'CUDA context/driver and graph metadata may exceed PyTorch allocator fraction'})
        if free<minimum:raise MemoryError('insufficient device free memory')
        torch.cuda.set_per_process_memory_fraction(plan['allocator_fraction'],0)
        import probe
        if Path(probe.__file__).resolve()!=HERE/'probe.py':raise ValueError('wrong probe source')
        k=int(name[2:]);cases=[x for x in inventory.inventory(hash_files=False) if x['budget']==k]
        if not probe.run_one(k,cases,lambda x:event(path,x)):raise ValueError('one or more complete-order mismatches')
    except BaseException as exc:
        event(path,{'event':'failed','case':name,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--child-case',choices=CASES);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plan_path=HERE/'plan001.json';plan=json.loads(plan_path.read_text())
    result=validate_plan(plan,hash_archives=a.child_case is None)
    if a.preflight_only:
        import torch
        if torch.cuda.is_initialized() or a.child_case or a.out.resolve()!=HERE/'preflight001.json':raise ValueError('bad CPU preflight')
        result['source_plan_sha256']=sha(plan_path);save(a.out,result);return
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
    save(a.out/'worker_complete.json',{'status':'attempts_recorded','cases':statuses,'GT_opened':False,'scoring_performed':False,
         'interpretation':'per-K process outcomes require formal independent audit; exit0 is not identity success'})

if __name__=='__main__':main()
