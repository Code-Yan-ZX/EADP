"""Source-only review for StackTok DEV scoring; no predictions or GT opened."""
import argparse, ast, hashlib, importlib.util, json
from pathlib import Path

B=Path(__file__).resolve().parent.parent
D=B/'e6/stacktok_full_analysis';R=Path(__file__).resolve()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=read(D/'score_plan001.json')
    assert plan['arms']==['stacktok_k256','eadp_k256_d1','pixel_t0','pixel_t045']
    assert plan['tasks']==['DocVQA_VAL','TextVQA_VAL','AI2D_TEST'] and len(plan['contrasts'])==3
    assert plan['bootstrap']=={'unit':'image group','paired':True,'replicates':10000,'seed':20261104,'confidence':0.95,'multiplicity_adjustment':'none'}
    assert plan['status']=='frozen_during_stacktok_generation_before_observing_panel_results'
    spec=importlib.util.spec_from_file_location('_stacktok_scoring_prelaunch_admission',D/'admission.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert Path(m.__file__).resolve()==D/'admission.py'
    sources=m.source_hashes();assert len(sources)==34 and all(sha(path)==h for path,h in sources.items())
    for name in ('score.py','admission.py','paired.py','test_main_integration.py','test_score.py','test_paired.py','score_plan001.json'):
        assert str((D/name).resolve()) in sources
    cpu=read(D/'cpu_check003/verification.json');s=read(D/'monitor_cpu003/started.json');t=read(D/'monitor_cpu003/terminal.json');cwd=Path(s['cwd'])
    assert cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after']==sources
    assert cpu['GT_opened'] is False and cpu['real_predictions_opened'] is False and cpu['test_returncode']==0
    assert s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group']
    cmd=s['command'];assert [(cwd/x).resolve() for x in cmd if Path(x).name=='cpu_check.py']==[(D/'cpu_check.py').resolve()]
    log=(D/'cpu_check003/tests.stderr').read_text();assert 'Ran 6 tests' in log and '\nOK\n' in log
    assert all(x in log for x in ('test_actual_main_and_gate','test_missing_prelaunch_refuses_before_prediction_loader','test_full_inventory_and_mismatch','test_reject_missing_or_nonfinite'))
    tree=ast.parse((D/'score.py').read_text()); main_fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    calls=[n for n in ast.walk(main_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    gate=[n.lineno for n in calls if isinstance(n.func.value,ast.Name) and n.func.value.id=='admission' and n.func.attr=='require']
    truth=[n.lineno for n in calls if n.func.attr=='read_truths']
    assert len(gate)==len(truth)==1 and gate[0]<truth[0]
    proof={'status':'passed','independent_review':True,'scope':'stacktok_full_score_prelaunch','source_sha256':sources,
           'plan_sha256':sha(D/'score_plan001.json'),'cpu_sha256':sha(D/'cpu_check003/verification.json'),
           'cpu_started_sha256':sha(D/'monitor_cpu003/started.json'),'cpu_terminal_sha256':sha(D/'monitor_cpu003/terminal.json'),
           'reviewer_sha256':sha(R),'GT_opened':False,'predictions_opened':False,'errors':[],
           'limitations':['Source and synthetic tests only; scoring and its outputs still require independent result review.',
                          'DEV paired intervals are exploratory and uncorrected; equal K does not establish equal compute.']}
    with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
if __name__=='__main__':main()
