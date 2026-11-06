"""CPU identity gate for the already archived 32x3 selector inputs."""
import hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
REAL=BASE/'e5/selector_real32';AUDIT=BASE/'e1_independent_review'
FORMAL=REAL/'run_001_verification.json';REVIEW=AUDIT/'selector_real32_run001_review.json'
MANIFEST=REAL/'sample_manifest_001.json'
SCATTER=BASE/'e5/author_graph_scatter';SREVIEW=AUDIT/'author_graph_scatter_result_review.json'
SREVIEWER=AUDIT/'review_author_graph_scatter.py';SMON=AUDIT/'monitor_author_graph_scatter_result'
KS=(128,256,512)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def terminal(monitor,script,output):
    s=read(monitor/'started.json');t=read(monitor/'terminal.json')
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]
        or t.get('worker_pid')!=s.get('worker_pid')):raise ValueError('supervised proof did not exit successfully')
    cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    if (len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or cmd.count('--out')!=1
        or cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=output.resolve()):
        raise ValueError('supervisor command identity mismatch')
    return s,t

def inventory(*,hash_files=True):
    formal=read(FORMAL);review=read(REVIEW);manifest=read(MANIFEST)
    terminal(REAL/'monitor_verify_run001',REAL/'verify.py',FORMAL)
    terminal(AUDIT/'monitor_selector_real32_run001',AUDIT/'review_selector_real32_run001.py',REVIEW)
    if (formal.get('status')!='passed' or formal.get('images')!=32 or formal.get('cases')!=96
        or formal.get('verifier_sha256')!=sha(REAL/'verify.py')
        or formal.get('GT_opened') is not False or formal.get('scoring_performed') is not False
        or review.get('status')!='passed' or review.get('independent_review') is not True
        or review.get('images')!=32 or review.get('cases')!=96 or review.get('formal_sha256')!=sha(FORMAL)
        or review.get('reviewer_sha256')!=sha(AUDIT/'review_selector_real32_run001.py')
        or review.get('GT_opened') is not False or review.get('scoring_performed') is not False
        or formal.get('source_sha256',{}).get(str(MANIFEST.resolve()))!=sha(MANIFEST)
        or manifest.get('status')!='frozen' or manifest.get('budgets')!=list(KS)
        or len(manifest.get('pilot',[]))!=32):raise ValueError('real32 evidence incomplete')
    for key,path in [('formal_started_sha256',REAL/'monitor_verify_run001/started.json'),
                     ('formal_terminal_sha256',REAL/'monitor_verify_run001/terminal.json'),
                     ('run_started_sha256',REAL/'monitor_run001/started.json'),
                     ('run_terminal_sha256',REAL/'monitor_run001/terminal.json')]:
        if review.get(key)!=sha(path):raise ValueError('real32 review terminal binding changed')
    cases=[]
    for index,row in enumerate(manifest['pilot']):
        for k in KS:
            path=(REAL/'run_001'/f'image_{index:03d}'/f'case_{k}.pt').resolve()
            expected=formal.get('array_sha256',{}).get(str(path))
            if (not isinstance(expected,str) or len(expected)!=64 or
                review.get('array_sha256',{}).get(str(path))!=expected or (hash_files and sha(path)!=expected)):
                raise ValueError('archive identity missing or changed: '+str(path))
            cases.append({'image_index':index,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],
                          'budget':k,'path':str(path),'sha256':expected})
    if len(cases)!=96 or len({(x['image_index'],x['budget']) for x in cases})!=96:
        raise ValueError('duplicate or incomplete cases')
    return cases

def require_scatter_review():
    proof=read(SCATTER/'smoke001_verification.json');review=read(SREVIEW)
    terminal(SCATTER/'monitor_verify001',SCATTER/'verify.py',SCATTER/'smoke001_verification.json')
    terminal(SMON,SREVIEWER,SREVIEW)
    if (proof.get('status')!='passed' or proof.get('full_k_identity_all_passed') is not True
        or review.get('status')!='passed' or review.get('independent_review') is not True
        or review.get('formal_sha256')!=sha(SCATTER/'smoke001_verification.json')
        or review.get('case_input_sha256')!='f651a859849d1e0c84119a7b708ffff794cbe8beaf60e0d9a8bf63d5ae24f95d'
        or review.get('reviewer_sha256')!=sha(SREVIEWER) or review.get('errors')!=[]
        or review.get('GT_opened') is not False or review.get('scoring_performed') is not False):
        raise ValueError('scatter single-case independent review incomplete')
    for key,path in [('formal_started_sha256',SCATTER/'monitor_verify001/started.json'),
                     ('formal_terminal_sha256',SCATTER/'monitor_verify001/terminal.json'),
                     ('run_started_sha256',SCATTER/'monitor_smoke001/started.json'),
                     ('run_terminal_sha256',SCATTER/'monitor_smoke001/terminal.json'),
                     ('plan_sha256',SCATTER/'plan001.json'),
                     ('verification_plan_sha256',SCATTER/'verification_plan001.json')]:
        if review.get(key)!=sha(path):raise ValueError('scatter review evidence changed')
    files=proof.get('files_sha256')
    if not isinstance(files,dict) or not files or any(sha(path)!=digest for path,digest in files.items()):
        raise ValueError('scatter formal evidence changed')
    for k in (1,16,128,256,512):
        cases=[x for x in review.get('cases',[]) if x.get('case')==f'k_{k:03d}']
        if len(cases)!=1 or cases[0].get('outcome')!='completed':raise ValueError('scatter K gate incomplete')
    return {'formal_sha256':sha(SCATTER/'smoke001_verification.json'),'independent_review_sha256':sha(SREVIEW)}

if __name__=='__main__':raise SystemExit('no GPU launch; reviewed admission is required')
