import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[D/'PLAN.md',D/'compare.py',D/'freeze.py',B/'e6/pixelprune_dev384/plan001.json',B/'e6/rres_k256_dev384/design001.json',B/'e6/pixelprune_dev384_accept/verification001.json',B/'e6/rres_k256_dev384/verification001.json',I/'pixel_dev384_result_review.json',I/'rres_k256_dev384_result_review.json',I/'pixel_dev384_scores_review.json',I/'rres_k256_dev384_scores_review.json',B/'e6/pixelprune_dev384_analysis/score001/scores.jsonl',B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl']
for mon in ('monitor_pixel_dev384_result','monitor_pixel_dev384_scores','monitor_rres_k256_dev384_result','monitor_rres_k256_dev384_scores'):
 files.extend([I/mon/'started.json',I/mon/'terminal.json'])
out={'status':'frozen','GT_opened':False,'target_predictions_parsed':False,'existing_prediction_files_opened_for_hash':True,'source_sha256':{str(p):sha(p) for p in files},'questions':384,'replicates':10000,'seed':20261104}
with (D/'plan001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
