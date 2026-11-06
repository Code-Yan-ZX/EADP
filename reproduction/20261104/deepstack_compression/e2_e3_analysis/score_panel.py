"""Score only an entirely accepted E2/E3 DEV panel; admission precedes GT."""
import argparse, importlib.metadata, importlib.util, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; BASE=HERE.parent; FULL=BASE/'e2_e3_full'; REVIEW=BASE/'e1_independent_review'
sys.path[:0]=[str(FULL),str(BASE/'e1_analysis')]
from full_imports import local
import completion, metrics, score as e1_score
import paired, score_core
panel_gate=local('panel_gate'); panel_worker=local('panel_worker')
spec=importlib.util.spec_from_file_location('_eadp_e2e3_panel_verify',FULL/'panel_verify.py')
panel_verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(panel_verify)
if Path(panel_verify.__file__).resolve()!=(FULL/'panel_verify.py').resolve():raise ImportError('unexpected panel verifier')
PLAN_SHA256='1589b540c6ddc355a4a89ba2b416642fe7c901794e87c62be826b1dc2bc00915'
sha=panel_gate.sha

def read(path):return json.loads(Path(path).read_text())
def gate(a):
    """Hash proofs and artifacts; never interpret prediction JSON or labels."""
    panel_gate.terminal(a.verify_started,a.verify_terminal,FULL/'panel_verify.py',a.panel_proof)
    proof=read(a.panel_proof)
    if (proof.get('status')!='passed' or proof.get('questions')!=384 or proof.get('physical_generations')!=3840 or
        proof.get('E2_records')!=3072 or proof.get('E3_new_records')!=768 or proof.get('E3_endpoint_aliases')!=768 or
        proof.get('GT_opened') is not False or proof.get('scoring_performed') is not False or proof.get('verifier_sha256')!=sha(FULL/'panel_verify.py')):
        raise ValueError('formal full panel acceptance missing')
    reviewer=REVIEW/'review_e2e3full_run.py'; panel_gate.terminal(a.review_started,a.review_terminal,reviewer,a.review_proof)
    review=read(a.review_proof)
    if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('questions')!=384 or
        review.get('physical_generations')!=3840 or review.get('GT_opened') is not False or review.get('scoring_performed') is not False or
        review.get('formal_sha256')!=sha(a.panel_proof) or review.get('formal_started_sha256')!=sha(a.verify_started) or
        review.get('formal_terminal_sha256')!=sha(a.verify_terminal) or review.get('reviewer_sha256')!=sha(reviewer)):
        raise ValueError('independent full panel acceptance missing')
    run=a.run.resolve();accepted=proof.get('files_sha256',{})
    required=[a.e1_acceptance,a.e1_started,a.e1_terminal,a.run_started,a.run_terminal]+[run/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','data/worker_complete.json','data/records.jsonl','data/generated.jsonl','data/e3_views.jsonl')]
    if any(accepted.get(str(Path(p).resolve()))!=sha(p) for p in required):raise ValueError('accepted artifact missing or changed')
    panel_gate.files_match(accepted); panel_gate.files_match(proof.get('array_sha256',{}))
    budget=read(FULL/'budget_plan.json')
    panel_verify.validate_exit(run,read(a.run_started),read(a.run_terminal),read(run/'launch.json'),read(run/'exit.json'),budget)
    e1=panel_verify.check_e1(a.e1_acceptance,a.e1_started,a.e1_terminal)
    if e1.get('GT_opened') is not False or e1.get('scoring_performed') is not False:raise ValueError('upstream E1 proof invalid')
    rows,_=panel_worker.inputs(e1)
    frozen=read(run/'frozen.json')
    if set(frozen['code_sha256'])!=set(local('source_inventory').sources()):raise ValueError('incomplete source inventory')
    panel_gate.files_match(frozen['code_sha256'])
    return proof,e1,rows

def validate_statistics_review():
    reviewer=REVIEW/'review_e2e3_stats.py';proof=REVIEW/'e2e3_stats_review.json';started=REVIEW/'monitor_e2e3stats/started.json';ended=REVIEW/'monitor_e2e3stats/terminal.json'
    panel_gate.terminal(started,ended,reviewer)
    r=read(proof);cs=HERE/'monitor_cpu002/started.json';ct=HERE/'monitor_cpu002/terminal.json';s=read(cs);t=read(ct);cmd=s.get('command',[]);cwd=Path(s.get('cwd','/'))
    if (r.get('status')!='passed' or r.get('contrasts_verified')!=11 or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('plan_sha256')!=sha(HERE/'plan.json') or r.get('source_sha256')!=sha(HERE/'paired.py') or r.get('test_sha256')!=sha(HERE/'test_paired.py') or r.get('reviewer_sha256')!=sha(reviewer) or r.get('prior_cpu_started_sha256')!=sha(cs) or r.get('prior_cpu_terminal_sha256')!=sha(ct) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or t.get('worker_pid')!=s.get('worker_pid') or '-m' not in cmd or 'unittest' not in cmd or 'discover' not in cmd or cmd.count('-s')!=1 or (cwd/cmd[cmd.index('-s')+1]).resolve()!=HERE):
        raise ValueError('independent statistics review invalid')
    return r

def validate_analysis_review():
    """Bind scorer and predeclared statistics to independently tested source."""
    proof=REVIEW/'e2e3_score_prelaunch_review.json';reviewer=REVIEW/'review_e2e3_score.py';start=REVIEW/'monitor_e2e3_score_prelaunch/started.json';end=REVIEW/'monitor_e2e3_score_prelaunch/terminal.json'
    panel_gate.terminal(start,end,reviewer)
    r=read(proof);sources=r.get('source_sha256',{})
    required=[HERE/x for x in ('plan.json','paired.py','score_core.py','score_panel.py','test_paired.py','test_score_core.py','test_score_gate.py')]
    if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('reviewer_sha256')!=sha(reviewer) or r.get('stats_review_sha256')!=sha(REVIEW/'e2e3_stats_review.json') or any(sources.get(str(x.resolve()))!=sha(x) for x in required)):
        raise ValueError('independent scoring source review invalid')
    panel_gate.files_match(sources)
    validate_statistics_review()
    return r

def main():
    p=argparse.ArgumentParser()
    for name in ('run','panel-proof','verify-started','verify-terminal','run-started','run-terminal','e1-acceptance','e1-started','e1-terminal','review-proof','review-started','review-terminal','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if sha(HERE/'plan.json')!=PLAN_SHA256:raise ValueError('analysis plan changed')
    validate_analysis_review()
    sources=[HERE/x for x in ('plan.json','paired.py','score_core.py','score_panel.py')]+[BASE/'e1_analysis'/x for x in ('score.py','metrics.py','completion.py')]+[metrics.VQA,metrics.MISC,metrics.MATCH,REVIEW/'review_e2e3full_run.py']
    pre={str(x.resolve()):sha(x) for x in sources}
    proof,e1,panel=gate(a)  # no prediction parsing or GT before admission
    records=completion.read_jsonl(a.run/'data/records.jsonl');grouped=score_core.inventory(panel,records)
    dataset=Path(e1['files']['dataset_summary']['path']);meta=read(dataset)
    if sha(dataset)!=e1['files']['dataset_summary']['sha256'] or set(meta['sources'])!=set(score_core.TASKS):raise ValueError('dataset source proof')
    needed={(r['task'],str(r['qid'])) for r in panel}
    a.out.mkdir(parents=True,exist_ok=False)
    frozen={'stage':'E2/E3 DEV scoring','plan_sha256':sha(HERE/'plan.json'),'panel_proof_sha256':sha(a.panel_proof),'review_proof_sha256':sha(a.review_proof),'source_sha256':pre,'dataset_summary_sha256':sha(dataset),'reference_tsv':meta['sources'],'scope':'384 DEV only; non-DEV answer values not parsed','runtime_versions':{n:importlib.metadata.version(n) for n in ('numpy','pandas','xlsxwriter','openpyxl')}}
    with (a.out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    truth=e1_score.read_truths(meta['sources'],needed);author=metrics.author_functions();constants=metrics._author_constants();native=score_core.native_predictions(records)
    def scorer(task,answer,truth_value,native_prediction):
        return metrics.score(task,answer,truth_value,author,constants,scoring_prediction=native_prediction)
    scored=score_core.score_rows(panel,grouped,truth,native,scorer);plan=read(HERE/'plan.json');tasks={}
    for task in score_core.TASKS:
        subset=[x for x in scored if x['task']==task]
        rows=[{'group_id':x['group_id'],'scores':{arm:x['arms'][arm]['score'] for arm in paired.CONFIGS}} for x in subset]
        tasks[task]=paired.analyze(rows,replicates=plan['bootstrap']['replicates'],seed=plan['bootstrap']['seed'])
        tasks[task]['terminal']={arm:{'eos':sum(x['arms'][arm]['eos'] for x in subset),'truncated':sum(x['arms'][arm]['truncated'] for x in subset),'empty':sum(not x['arms'][arm]['answer'].strip() for x in subset)} for arm in paired.CONFIGS}
    panel_gate.files_match(pre);panel_gate.files_match(proof['files_sha256']);panel_gate.files_match(proof['array_sha256'])
    for desc in meta['sources'].values():
        if sha(desc['path'])!=desc['sha256']:raise ValueError('source dataset changed')
    score_path=a.out/'scores.jsonl'
    with score_path.open('x') as f:
        for r in scored:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush();os.fsync(f.fileno())
    report={'status':'completed','n_questions':384,'n_predictions':len(records),'tasks':tasks,'scores_sha256':sha(score_path),'plan_sha256':PLAN_SHA256,'panel_proof_sha256':sha(a.panel_proof),'limitations':plan['limits']}
    with (a.out/'analysis.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'status':'completed','n_questions':384,'n_predictions':len(records)}))
if __name__=='__main__':main()
