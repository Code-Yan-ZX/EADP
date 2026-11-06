"""Fail-closed gate for future 32x3 formal and independent identity proofs."""
import importlib.util,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
COMPONENT=BASE/'e5/author_graph_scatter_real32';AUDIT=BASE/'e1_independent_review'
FORMAL=COMPONENT/'smoke001_verification.json';FMON=COMPONENT/'monitor_verify001'
REVIEW=AUDIT/'author_graph_scatter_real32_result_review.json';REVIEWER=AUDIT/'review_author_graph_scatter_real32.py'
RMON=AUDIT/'monitor_author_graph_scatter_real32_result'

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def terminal(monitor,script,output):
    s=read(monitor/'started.json');t=read(monitor/'terminal.json');cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
        or t.get('worker_pid')!=s.get('worker_pid') or len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve()
        or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=output.resolve()):
        raise ValueError('component verifier supervisor incomplete')

def validate_summary(formal,review,bindings):
    if (formal.get('status')!='passed' or formal.get('complete_96_identity') is not True
        or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False
        or formal.get('cuda_initialized') is not False or formal.get('errors')!=[]
        or review.get('status')!='passed' or review.get('independent_review') is not True
        or review.get('complete_96_identity') is not True or review.get('GT_opened') is not False
        or review.get('scoring_performed') is not False or review.get('errors')!=[]
        or any(review.get(k)!=v for k,v in bindings.items())):
        raise ValueError('32x3 independently verified identity required')
    budgets=formal.get('budgets',[])
    if ([x.get('k') for x in budgets]!=[128,256,512] or any(x.get('outcome')!='completed' or x.get('compared_images')!=32
        or x.get('identity_mismatch_images')!=[] for x in budgets)):
        raise ValueError('component identity inventory incomplete')

def require():
    formal=read(FORMAL);review=read(REVIEW)
    terminal(FMON,COMPONENT/'verify.py',FORMAL);terminal(RMON,REVIEWER,REVIEW)
    bindings={'formal_sha256':sha(FORMAL),'formal_started_sha256':sha(FMON/'started.json'),
              'formal_terminal_sha256':sha(FMON/'terminal.json'),
              'run_started_sha256':sha(COMPONENT/'monitor_smoke001/started.json'),
              'run_terminal_sha256':sha(COMPONENT/'monitor_smoke001/terminal.json'),
              'plan_sha256':sha(COMPONENT/'plan001.json'),'reviewer_sha256':sha(REVIEWER)}
    validate_summary(formal,review,bindings)
    plan=read(COMPONENT/'plan001.json')
    if formal.get('verifier_sha256')!=sha(COMPONENT/'verify.py'):
        raise ValueError('component verifier changed')
    for path,digest in plan.get('sha256',{}).items():
        if sha(path)!=digest:raise ValueError('component source changed: '+path)
    files=formal.get('files_sha256')
    if not isinstance(files,dict) or not files or any(sha(path)!=digest for path,digest in files.items()):
        raise ValueError('component evidence changed')
    spec=importlib.util.spec_from_file_location('scatter32_fixed_inventory',COMPONENT/'inventory.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve()!=COMPONENT/'inventory.py':raise ValueError('wrong archive gate module')
    cases=module.inventory(hash_files=True)
    if cases!=plan.get('case_manifest'):raise ValueError('component archive identities changed')
    return {'status':'passed','scope':'32x3 identity gate only; GPU still needs root scheduling',
            'formal_sha256':sha(FORMAL),'review_sha256':sha(REVIEW),'archive_cases':len(cases)}

if __name__=='__main__':raise SystemExit('locked: complete component proofs and reviewed launcher required')
