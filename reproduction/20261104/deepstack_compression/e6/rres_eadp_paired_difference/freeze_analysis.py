"""Freeze the posthoc descriptive analysis before reading score rows."""
import json,os
from pathlib import Path
from analyze import sha,ended
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 if not ended(HERE/'monitor_cpu002/started.json',HERE/'monitor_cpu002/terminal.json'):raise ValueError('CPU proof')
 paths=[HERE/x for x in ('PLAN.md','paired.py','test_paired.py','analyze.py','freeze_analysis.py','monitor_cpu002/started.json','monitor_cpu002/terminal.json')]
 paths.extend([ROOT/'e1_independent_review/rres_k256_dev384_scores_review.json',ROOT/'e1_independent_review/review_rres_k256_dev384_scores.py',ROOT/'e1_independent_review/monitor_rres_k256_dev384_scores/started.json',ROOT/'e1_independent_review/monitor_rres_k256_dev384_scores/terminal.json',ROOT/'e6/rres_k256_dev384/design001.json'])
 result={'status':'frozen','scope':'posthoc paired DEV score descriptions; no training','draws':10000,'seed':20261104,'source_sha256':{str(p.resolve()):sha(p) for p in paths},'GT_opened':False,'score_rows_opened':False,'GPU_authorized':False}
 with (HERE/'analysis_plan001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
