"""Score full accepted DEV384 only; no predictions or labels before admission."""
import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEV=HERE.parent
BASE=DEV.parents[2]
REVIEW=BASE/'e1_independent_review'
sys.path[:0]=[str(DEV),str(BASE/'e1_analysis')]
import gate,metrics,score as e1_score
import score_core as core

def validate_imports():
    expected=((gate,DEV/'gate.py'),(metrics,BASE/'e1_analysis/metrics.py'),(e1_score,BASE/'e1_analysis/score.py'),(core,HERE/'score_core.py'))
    for module,path in expected:
        if Path(module.__file__).resolve()!=path.resolve():raise ImportError('scoring module source mismatch')

def read(p):return json.loads(Path(p).read_text())
def lines(p):
    with Path(p).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]

def source_review():
    proof=HERE/'prelaunch_review.json';reviewer=HERE/'review_scoring.py'
    gate.terminal(HERE/'monitor_review_scoring/started.json',HERE/'monitor_review_scoring/terminal.json',reviewer,proof)
    r=read(proof);required=[DEV/'scoring_plan_001.json',HERE/'score_core.py',HERE/'test_score_core.py',HERE/'test_score_gate.py',HERE/'score_panel.py',DEV/'gate.py',BASE/'e1_analysis/metrics.py',BASE/'e1_analysis/score.py',BASE/'e1_analysis/paired_stats.py',metrics.VQA,metrics.MISC,metrics.MATCH]
    if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('reviewer_sha256')!=gate.sha(reviewer) or r.get('core_started_sha256')!=gate.sha(HERE/'monitor_core005/started.json') or r.get('core_terminal_sha256')!=gate.sha(HERE/'monitor_core005/terminal.json') or any(r.get('source_sha256',{}).get(str(p.resolve()))!=gate.sha(p) for p in required)):
        raise ValueError('independent score source review missing or changed')
    s=read(HERE/'monitor_core005/started.json');t=read(HERE/'monitor_core005/terminal.json');cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or '-m' not in cmd or 'unittest' not in cmd or 'discover' not in cmd or cmd.count('-s')!=1 or (cwd/cmd[cmd.index('-s')+1]).resolve()!=HERE):raise ValueError('scoring core tests not complete')
    return r

def accepted_prior(run):
    formal=DEV/'run_001_verification.json';review=DEV/'run_001_independent_review.json'
    fs=DEV/'monitor_verify001/started.json';ft=DEV/'monitor_verify001/terminal.json'
    rs=DEV/'monitor_review_run001/started.json';rt=DEV/'monitor_review_run001/terminal.json'
    gate.terminal(fs,ft,DEV/'panel_verify.py',formal);gate.terminal(rs,rt,DEV/'review_panel.py',review)
    gate.terminal(DEV/'monitor_001/started.json',DEV/'monitor_001/terminal.json',DEV/'launch.py')
    f=read(formal);r=read(review)
    if (f.get('status')!='passed' or f.get('questions')!=384 or f.get('generations')!=1152 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('verifier_sha256')!=gate.sha(DEV/'panel_verify.py') or
        r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=384 or r.get('generations')!=1152 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('formal_sha256')!=gate.sha(formal) or r.get('formal_started_sha256')!=gate.sha(fs) or r.get('formal_terminal_sha256')!=gate.sha(ft) or r.get('reviewer_sha256')!=gate.sha(DEV/'review_panel.py')):
        raise ValueError('PriorTR formal or independent full-panel acceptance absent')
    files=f.get('files_sha256',{});arrays=f.get('array_sha256',{})
    required=[DEV/'monitor_001/started.json',DEV/'monitor_001/terminal.json']+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl')]
    if len(arrays)!=9 or any(files.get(str(p.resolve()))!=gate.sha(p) for p in required):raise ValueError('accepted PriorTR artifacts missing or changed')
    for path,digest in {**files,**arrays}.items():
        if gate.sha(path)!=digest:raise ValueError('accepted PriorTR artifact changed')
    frozen=read(run/'frozen.json')
    for path,digest in frozen['source_sha256'].items():
        if gate.sha(path)!=digest:raise ValueError('PriorTR runtime source changed')
    return f,r

def accepted_e1():
    score=BASE/'e1/e1full001_score/scores.jsonl';analysis=score.parent/'analysis.json';frozen=score.parent/'frozen.json';proof=REVIEW/'e1full001_score_review.json'
    st=BASE/'e1/monitorscoree1full001/started.json';end=BASE/'e1/monitorscoree1full001/terminal.json'
    gate.terminal(st,end,BASE/'e1_analysis/score.py',score.parent)
    gate.terminal(REVIEW/'monitor_e1score001/started.json',REVIEW/'monitor_e1score001/terminal.json',REVIEW/'review_score.py')
    r=read(proof);e=read(BASE/'e1/e1full001_completion.json')
    if (r.get('status')!='passed' or r.get('n_verified')!=384 or r.get('arms_verified')!=1536 or r.get('errors')!=[] or r.get('score_sha256')!=gate.sha(score) or r.get('analysis_sha256')!=gate.sha(analysis) or r.get('score_frozen_sha256')!=gate.sha(frozen) or r.get('score_terminal_sha256')!=gate.sha(end) or r.get('reviewer_sha256')!=gate.sha(REVIEW/'review_score.py') or r.get('e1_review_sha256')!=gate.sha(REVIEW/'e1full001_review.json')):
        raise ValueError('accepted E1 score unavailable')
    if e.get('status')!='passed' or e.get('GT_opened') is not False or e.get('scoring_performed') is not False:raise ValueError('E1 generation acceptance invalid')
    frozen_score=read(frozen)
    versions={x:importlib.metadata.version(x) for x in ('numpy','pandas','xlsxwriter','openpyxl')}
    if frozen_score.get('runtime_versions')!=versions:raise ValueError('E1 scoring runtime changed')
    old=frozen_score.get('code_sha256',{})
    expected=[BASE/'e1_analysis/metrics.py',BASE/'e1_analysis/score.py',BASE/'e1_analysis/paired_stats.py',metrics.VQA,metrics.MISC,metrics.MATCH]
    if set(old)!={str(Path(p).resolve()) for p in expected} or any(old[str(Path(p).resolve())]!=gate.sha(p) for p in expected):raise ValueError('E1 scoring source changed')
    return e,score,proof

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    validate_imports()
    run=a.run.resolve()
    if run!=(DEV/'run_001').resolve():raise ValueError('wrong run')
    src=source_review();formal,review=accepted_prior(run);e1,e1score,e1review=accepted_e1()
    # Above admission checks only hashes and proofs, no prediction/reference rows.
    sample=read(DEV/'sample_001.json')['sample'];records=lines(run/'data/records.jsonl');grouped=core.inventory(sample,records)
    e1rows=lines(e1score); e1scores={(r['task'],str(r['qid'])):r for r in e1rows}
    needed={(r['task'],str(r['qid'])) for r in sample}
    if len(e1rows)!=384 or set(e1scores)!=needed:raise ValueError('E1 score inventory')
    ds=Path(e1['files']['dataset_summary']['path']);meta=read(ds)
    if gate.sha(ds)!=e1['files']['dataset_summary']['sha256'] or set(meta['sources'])!=set(core.TASKS):raise ValueError('dataset identity')
    sourcefiles=[DEV/'scoring_plan_001.json',HERE/'score_core.py',HERE/'score_panel.py',BASE/'e1_analysis/metrics.py',BASE/'e1_analysis/score.py',metrics.VQA,metrics.MISC,metrics.MATCH]
    before={str(Path(p).resolve()):gate.sha(p) for p in sourcefiles}
    a.out.mkdir(parents=True,exist_ok=False)
    frozen={'source_sha256':before,'prior_formal_sha256':gate.sha(DEV/'run_001_verification.json'),'prior_review_sha256':gate.sha(DEV/'run_001_independent_review.json'),'e1_score_sha256':gate.sha(e1score),'e1_review_sha256':gate.sha(e1review),'score_source_review_sha256':gate.sha(HERE/'prelaunch_review.json'),'reference_tsv':meta['sources'],'scope':'only DEV qids parsed; no CONFIRM/FINAL answers','runtime_versions':{x:importlib.metadata.version(x) for x in ('numpy','pandas','xlsxwriter','openpyxl')}}
    with (a.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    truths=e1_score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants();native=core.native_predictions(records)
    def scorer(task,answer,truth,native_prediction):return metrics.score(task,answer,truth,author,constants,scoring_prediction=native_prediction)
    scored=core.score_rows(sample,grouped,truths,native,e1scores,scorer);tasks={}
    for task in core.TASKS:
        part=[r for r in scored if r['task']==task];tasks[task]=core.analyze(part)
        tasks[task]['terminal']={arm:{'eos':sum(r['arms'][arm]['eos'] for r in part),'truncated':sum(r['arms'][arm]['truncated'] for r in part)} for arm in (*core.ARMS,core.REFERENCE)}
    for path,digest in before.items():
        if gate.sha(path)!=digest:raise ValueError('scoring source changed')
    accepted_prior(run);accepted_e1()
    for desc in meta['sources'].values():
        if gate.sha(desc['path'])!=desc['sha256']:raise ValueError('dataset changed')
    path=a.out/'scores.jsonl'
    with path.open('x') as f:
        for r in scored:f.write(json.dumps(r,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    report={'status':'completed','questions':384,'new_predictions':1152,'tasks':tasks,'scores_sha256':gate.sha(path),'plan_sha256':gate.sha(DEV/'scoring_plan_001.json'),'formal_sha256':gate.sha(DEV/'run_001_verification.json'),'independent_review_sha256':gate.sha(DEV/'run_001_independent_review.json'),'limits':read(DEV/'scoring_plan_001.json')['limits']}
    with (a.out/'analysis.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'completed','questions':384,'new_predictions':1152}))
if __name__=='__main__':main()
