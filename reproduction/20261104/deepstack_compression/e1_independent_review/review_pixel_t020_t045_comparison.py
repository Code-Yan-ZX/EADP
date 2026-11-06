"""Independent reconstruction of post-hoc paired DEV score differences."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_t020_t045_comparison'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(m,script):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or [(cwd/x).resolve() for x in c if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('supervision '+str(m))
def check(tag):
 a=B/f'e6/pixelprune_{tag}_dev384_analysis';p=R/f'pixel_{tag}_dev384_scores_review.json';reviewer=R/f'review_pixel_{tag}_dev384_scores.py';proof=read(p);terminal(a/'monitor_score001',a/'score.py');terminal(R/('monitor_pixel_t045_dev384_scores002' if tag=='t045' else 'monitor_pixel_t020_dev384_scores'),reviewer)
 for key,path in (('score_sha256',a/'score001/scores.jsonl'),('analysis_sha256',a/'score001/analysis.json'),('frozen_sha256',a/'score001/frozen.json'),('score_started_sha256',a/'monitor_score001/started.json'),('score_terminal_sha256',a/'monitor_score001/terminal.json'),('reviewer_sha256',reviewer)):
  if proof[key]!=sha(path):raise ValueError('score review binding '+key)
 if proof['status']!='passed' or proof['independent_review'] is not True or proof['questions']!=384 or proof['pixel_scores_recomputed']!=768 or proof['errors']!=[]:raise ValueError('score review incomplete')
 rows=[json.loads(x) for x in (a/'score001/scores.jsonl').read_text().splitlines()];index={(x['task'],str(x['qid']),x['group_id']):x for x in rows}
 if len(rows)!=384 or len(index)!=384:return None
 return index
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();terminal(D/'monitor_compare002',D/'compare_v2.py');plan=read(D/'plan001.json');freeze=read(D/'source_freeze002.json');result=read(D/'result002.json')
if freeze['plan_sha256']!=sha(D/'plan001.json') or freeze['script_sha256']!=sha(D/'compare_v2.py') or result['plan_sha256']!=sha(D/'plan001.json') or result['script_sha256']!=sha(D/'compare_v2.py') or result['status']!='completed' or result['GT_opened'] is not False or result['score_files_containing_predictions_opened'] is not True:raise ValueError('source or completion')
new=check('t020');old=check('t045')
if new is None or old is None or set(new)!=set(old) or sha(B/'e6/pixelprune_t045_dev384_analysis/score001/scores.jsonl')!=plan['prior_t045_scores_sha256']:raise ValueError('pair inventory')
for tag in ('t020','t045'):
 if result[tag+'_review_sha256']!=sha(R/f'pixel_{tag}_dev384_scores_review.json') or result[tag+'_score_sha256']!=sha(B/f'e6/pixelprune_{tag}_dev384_analysis/score001/scores.jsonl'):raise ValueError('result evidence binding')
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 keys=sorted((k for k in new if k[0]==task),key=lambda k:k[2])
 if len(keys)!=128 or len({k[2] for k in keys})!=128:raise ValueError('group count')
 v=np.array([[new[k]['arms']['pixel_t020']['score'],old[k]['arms']['pixel_t045']['score']] for k in keys],dtype=float)
 if not np.isfinite(v).all() or (v<0).any() or (v>1).any():raise ValueError('invalid values')
 d=v[:,0]-v[:,1];rng=np.random.default_rng(20261104);indices=rng.integers(0,128,size=(10000,128));boots=v[indices].mean(axis=1);ci=np.quantile(100*(boots[:,0]-boots[:,1]),[.025,.975]);r=result['results'][task]
 if r['questions']!=128 or not math.isclose(r['mean_difference_pp'],100*d.mean(),abs_tol=1e-10) or not np.allclose(r['image_cluster_percentile_95_ci_pp'],ci,rtol=1e-11,atol=1e-10) or (r['improved'],r['declined'],r['unchanged'])!=(int((d>1e-12).sum()),int((d< -1e-12).sum()),int((abs(d)<=1e-12).sum())):raise ValueError('recomputation mismatch')
out={'status':'passed','independent_review':True,'questions':384,'tasks':3,'paired_bootstrap_replicates':10000,'GT_opened':False,'score_files_containing_predictions_opened':True,'result_sha256':sha(D/'result002.json'),'plan_sha256':sha(D/'plan001.json'),'script_sha256':sha(D/'compare_v2.py'),'comparison_started_sha256':sha(D/'monitor_compare001/started.json'),'comparison_terminal_sha256':sha(D/'monitor_compare001/terminal.json'),'reviewer_sha256':sha(__file__),'limitations':plan['limitations']+['first comparison failed after reading complete t020 scores; v2 source repair and the group ordering used for fixed-seed resampling were post-observation'],'errors':[]}
with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
