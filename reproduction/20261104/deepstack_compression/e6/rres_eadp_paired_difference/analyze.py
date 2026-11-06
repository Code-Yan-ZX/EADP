"""Read accepted precomputed scores only after frozen independent admission."""
import hashlib,json,os
from pathlib import Path
from paired import compute
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ended(s,t):
 a=read(s);b=read(t);return type(a.get('worker_pid')) is int and b.get('worker_pid')==a['worker_pid'] and b.get('reason')=='completed' and b.get('worker_returncode')==0 and b.get('remaining_process_group')==[]
def main():
 plan=read(HERE/'analysis_plan001.json')
 if plan.get('status')!='frozen' or any(sha(path)!=digest for path,digest in plan['source_sha256'].items()):raise ValueError('source changed')
 independent=ROOT/'e1_independent_review/rres_eadp_paired_difference_prelaunch_review.json';reviewer=ROOT/'e1_independent_review/review_rres_eadp_paired_difference_prelaunch.py';s=ROOT/'e1_independent_review/monitor_rres_eadp_paired_difference_prelaunch/started.json';t=s.parent/'terminal.json';q=read(independent)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('prelaunch_only') is not True or q.get('plan_sha256')!=sha(HERE/'analysis_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('GT_opened') is not False or q.get('scores_opened') is not False or q.get('errors')!=[] or not ended(s,t):raise ValueError('analysis prelaunch review')
 prior=ROOT/'e1_independent_review/rres_k256_dev384_scores_review.json';r=read(prior);score=ROOT/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl';s=ROOT/'e1_independent_review/monitor_rres_k256_dev384_scores/started.json';t=s.parent/'terminal.json'
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=384 or r.get('errors')!=[] or not ended(s,t) or sha(score)!=r.get('score_sha256'):raise ValueError('accepted score identity')
 anchors=read(ROOT/'e6/rres_k256_dev384/design001.json')['anchors']
 with score.open() as f:rows=[json.loads(line) for line in f]
 result=compute(rows,anchors,draws=10000,seed=20261104);result.update(plan_sha256=sha(HERE/'analysis_plan001.json'),prelaunch_review_sha256=sha(independent),accepted_scores_review_sha256=sha(prior),scores_sha256=sha(score))
 with (HERE/'difference001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
