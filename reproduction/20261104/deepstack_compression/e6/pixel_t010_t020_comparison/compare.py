"""Posthoc paired DEV comparison of two independently scored complete panels."""
import hashlib,json,os
from pathlib import Path
import numpy as np

D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()


def load(p):return json.loads(Path(p).read_text())


def terminal(path):
 s=load(path/'started.json');t=load(path/'terminal.json')
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('review not completed')


def checked(tag):
 score=B/f'e6/pixelprune_{tag}_dev384_analysis/score001';proof=load(R/f'pixel_{tag}_dev384_scores_review.json')
 terminal(B/f'e6/pixelprune_{tag}_dev384_analysis/monitor_score001');terminal(R/f'monitor_pixel_{tag}_dev384_scores')
 if (proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('questions')!=384 or proof.get('errors')!=[] or proof.get('score_sha256')!=sha(score/'scores.jsonl') or proof.get('analysis_sha256')!=sha(score/'analysis.json') or proof.get('reviewer_sha256')!=sha(R/f'review_pixel_{tag}_dev384_scores.py')):raise ValueError('complete scored panel proof missing')
 rows=[json.loads(line) for line in (score/'scores.jsonl').read_text().splitlines()]
 indexed={(r['task'],str(r['qid']),r['group_id']):r for r in rows}
 if len(rows)!=384 or len(indexed)!=384:raise ValueError('duplicate or incomplete panel')
 return score,proof,indexed


def main():
 plan=load(D/'plan001.json');freeze=load(D/'source_freeze001.json')
 if freeze['plan_sha256']!=sha(D/'plan001.json') or freeze['script_sha256']!=sha(__file__):raise ValueError('analysis source changed')
 old=B/'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl'
 if plan['prior_t020_scores_sha256']!=sha(old):raise ValueError('old reference changed')
 ns,np_,new=checked('t010');os_,op_,prev=checked('t020')
 if set(new)!=set(prev):raise ValueError('paired inventory mismatch')
 results={}
 for task in plan['tasks']:
  keys=sorted((k for k in new if k[0]==task),key=lambda k:k[2])
  if len(keys)!=128 or len({k[2] for k in keys})!=128:raise ValueError('invalid task groups')
  x=np.array([new[k]['arms']['pixel_t010']['score'] for k in keys],dtype=float)
  y=np.array([prev[k]['arms']['pixel_t020']['score'] for k in keys],dtype=float)
  if not np.isfinite(x).all() or not np.isfinite(y).all() or (x<0).any() or (x>1).any() or (y<0).any() or (y>1).any():raise ValueError('invalid scores')
  delta=x-y;rng=np.random.default_rng(20261104);indices=rng.integers(0,len(keys),(10000,len(keys)));boot=100*delta[indices].mean(axis=1)
  results[task]={'questions':len(keys),'mean_difference_pp':float(100*delta.mean()),'image_cluster_percentile_95_ci_pp':[float(v) for v in np.quantile(boot,[.025,.975])],
                 'improved':int((delta>1e-12).sum()),'declined':int((delta< -1e-12).sum()),'unchanged':int((abs(delta)<=1e-12).sum())}
 out={'status':'completed','plan_sha256':sha(D/'plan001.json'),'script_sha256':sha(__file__),
      't010_score_sha256':sha(ns/'scores.jsonl'),'t020_score_sha256':sha(os_/'scores.jsonl'),
      't010_review_sha256':sha(R/'pixel_t010_dev384_scores_review.json'),'t020_review_sha256':sha(R/'pixel_t020_dev384_scores_review.json'),
      'results':results,'limitations':plan['limitations'],'GT_opened':False,'score_files_containing_predictions_opened':True}
 with (D/'result001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())


if __name__=='__main__':main()
