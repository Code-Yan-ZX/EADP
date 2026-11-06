"""Second-person DEV score review, using direct per-image bootstrap draws."""
import argparse,collections,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd

B=Path(__file__).resolve().parent.parent;D=B/'e6/stacktok_full_analysis';I=B/'e1_independent_review';R=Path(__file__).resolve()
sys.path.insert(0,str(B/'e1_analysis'))
import metrics,score as e1score

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def key(r):return (r['task'],str(r['qid']),r['group_id'])
def close(x,y):return bool(np.allclose(x,y,rtol=1e-11,atol=1e-10))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    root=D/'score001';s=read(D/'monitor_score001/started.json');t=read(D/'monitor_score001/terminal.json')
    assert s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group']
    assert [(Path(s['cwd'])/x).resolve() for x in s['command'] if Path(x).name=='score.py']==[(D/'score.py').resolve()]
    f=read(root/'frozen.json');analysis=read(root/'analysis.json');plan=read(D/'score_plan001.json')
    assert analysis['status']=='completed' and analysis['questions']==384 and analysis['scores_sha256']==sha(root/'scores.jsonl')
    assert analysis['plan_sha256']==f['plan_sha256']==sha(D/'score_plan001.json') and plan['bootstrap']['replicates']==10000
    formal=read(B/'e6/stacktok_full/run_001_verification.json');ind=read(I/'stacktok_full_run001_review.json')
    assert formal['status']==ind['status']=='passed' and ind['independent_review'] and ind['questions']==384 and not ind['GT_opened']
    data=rows(root/'scores.jsonl');records={key(r):r for r in rows(B/'e6/stacktok_full/run_001/records.jsonl')}
    previous={key(r):r for r in rows(B/'e6/pixelprune_t045_dev384_analysis/score001/scores.jsonl')}
    anchors={key(r):r for r in read(B/'e6/pixelprune_t045_dev384/plan001.json')['anchors']}
    assert len(data)==len(records)==len(previous)==len(anchors)==384 and {key(r) for r in data}==set(records)==set(previous)==set(anchors)
    needed={(r['task'],str(r['qid'])) for r in data};truth=e1score.read_truths(f['references'],needed);author=metrics.author_functions();constants=metrics._author_constants()
    names=('stacktok_k256','eadp_k256_d1','pixel_t0','pixel_t045');details={}
    for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
        sub=[r for r in data if r['task']==task];assert len(sub)==128 and len({r['group_id'] for r in sub})==128
        buf=io.BytesIO();pd.DataFrame({'index':range(len(sub)),'prediction':[r['arms'][names[0]]['answer'] for r in sub]}).to_excel(buf,index=False,engine='xlsxwriter');buf.seek(0);native=pd.read_excel(buf)
        assert list(native['index'])==list(range(128))
        for r,actual in zip(sub,native['prediction']):
            k=key(r);arm=r['arms'][names[0]];record=records[k]
            assert arm['answer']==record['answer'] and arm['eos']==record['eos'] and arm['truncated']==record['truncated']
            v=metrics.score(task,arm['answer'],truth[(task,str(r['qid']))],author,constants,scoring_prediction=str(actual))
            assert math.isfinite(v['official']) and math.isfinite(v['independent'])
            assert abs(v['official']-arm['score'])<=1e-12 and abs(v['independent']-arm['independent_score'])<=1e-12
            for n,old in (('eadp_k256_d1','e4_k256_d1'),('pixel_t0','pixel_t0'),('pixel_t045','pixel_t045')):
                assert r['arms'][n]==previous[k]['arms'][old]
        sub.sort(key=lambda r:r['group_id']);values=np.array([[r['arms'][n]['score'] for n in names] for r in sub]);assert np.isfinite(values).all()
        sampled=np.random.default_rng(20261104).integers(0,128,(10000,128));boot=values[sampled].mean(axis=1);report=analysis['tasks'][task]
        assert report['n_questions']==report['n_image_groups']==128
        for j,n in enumerate(names):
            assert close(report['arm_means_pct'][n],100*values[:,j].mean())
            assert close(report['arm_percentile_95_ci_pct'][n],np.percentile(100*boot[:,j],[2.5,97.5]))
        for label,j in (('stacktok_minus_eadp',1),('stacktok_minus_pixel_t0',2),('stacktok_minus_pixel_t045',3)):
            delta=values[:,0]-values[:,j];got=report['contrasts'][label]
            assert close(got['difference_pp'],100*delta.mean()) and close(got['image_cluster_percentile_95_ci_pp'],np.percentile(100*(boot[:,0]-boot[:,j]),[2.5,97.5]))
            assert (got['improved'],got['declined'],got['unchanged'])==(int((delta>1e-12).sum()),int((delta< -1e-12).sum()),int((abs(delta)<=1e-12).sum()))
        assert report['terminal']=={'eos':sum(r['arms'][names[0]]['eos'] for r in sub),'truncated':sum(r['arms'][names[0]]['truncated'] for r in sub),'empty':sum(not r['arms'][names[0]]['answer'].strip() for r in sub)}
        details[task]={'questions':128,'image_groups':128,'score_checks':128,'contrast_checks':3}
    result={'status':'passed','independent_review':True,'independent_of_scoring_driver_author':True,'questions':384,'score_checks':384,'reference_checks':1152,'contrast_checks':9,
            'tasks':details,'GT_opened_DEV_only':True,'CONFIRM_FINAL_answers_opened':False,'score_sha256':sha(root/'scores.jsonl'),'analysis_sha256':sha(root/'analysis.json'),
            'plan_sha256':sha(D/'score_plan001.json'),'score_started_sha256':sha(D/'monitor_score001/started.json'),'score_terminal_sha256':sha(D/'monitor_score001/terminal.json'),
            'reviewer_sha256':sha(R),'limitations':['The accepted author and independent E1 metric implementations are reused.',
                                                   'All comparisons are exploratory DEV, uncorrected, and not equal-compute claims.']}
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
