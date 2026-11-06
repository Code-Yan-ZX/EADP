"""Summarize accepted 32-image selector pilot after independent failure review."""
import argparse,json,os,sys
from pathlib import Path
from timing_summary import summarize
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 rr=BASE/'e1_independent_review';proof=rr/'selector_real32_failure_review.json';reviewer=rr/'review_selector_real32_failure.py'
 supervised(rr/'monitor_selector_real32_failure/started.json',rr/'monitor_selector_real32_failure/terminal.json',reviewer,proof)
 x=json.loads(proof.read_text());analysis=HERE/'result_001.json';prior=rr/'selector_real32_run001_review.json'
 if x.get('status')!='passed' or x.get('independent_review') is not True or x.get('cases')!=2 or x.get('GT_opened') is not False or x.get('reviewer_sha256')!=sha(reviewer) or x.get('result_sha256')!=sha(analysis) or x.get('prior_review_sha256')!=sha(prior):raise ValueError('independent failure review rejected')
 z=json.loads(analysis.read_text());journal=BASE/'e5/selector_real32/run_001/journal.jsonl'
 if z.get('status')!='completed' or z.get('journal_sha256')!=sha(journal) or z.get('independent_review_sha256')!=sha(prior):raise ValueError('journal or upstream review changed')
 rows=[json.loads(s) for s in journal.read_text().splitlines() if s.strip()]
 out={'status':'completed','scope':'32 image exploratory real-input selector isolated CUDA event timings','budget_stats':summarize(rows),'source_sha256':sha(HERE/'timing_summary.py'),'runner_sha256':sha(__file__),'journal_sha256':sha(journal),'failure_review_sha256':sha(proof),'limitations':['Pilot expanded after single-image result; not the frozen 128 image full plan.','Each image median uses five interleaved measurements; distribution summarizes image medians.','K512 candidates omitted one of 32 images for exact order mismatch; 31-image paired subset is selected and may be biased.','CUDA graph replays the same fused kernel, not an independent selection algorithm; graph setup and persistent allocations are separate from event timing.','GPU event component timings are not end-to-end TTFT; background process and thermal effects are not independently eliminated.','No causal claim about GPU arithmetic or exact equivalence beyond recorded cases.']}
 with a.out.open('x') as h:json.dump(out,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
if __name__=='__main__':main()
