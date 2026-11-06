import hashlib,importlib.metadata,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';E=B/'e1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=[B/'initial_plan.json',D/'PLAN.md',D/'core.py',D/'run.py',D/'freeze.py',D/'cpu_check.py',D/'test_core.py',D/'test_main_synthetic.py',E/'e1full001_completion.json',E/'input_preparation_002/inputs.jsonl',E/'input_preparation_002/independent_verification.json',B/'e6/rres_k256_dev384/design001.json',B/'e6/rres_k256_dev384/verification001.json',B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl',I/'e1full001_review.json',I/'rres_k256_dev384_result_review.json',I/'rres_k256_dev384_scores_review.json',I/'review_full.py',I/'review_rres_k256_dev384_result.py',I/'review_rres_k256_dev384_scores.py']
for mon in (E/'monitorverifye1full001',I/'monitor_e1full001',I/'monitor_rres_k256_dev384_result',I/'monitor_rres_k256_dev384_scores',B/'e6/rres_k256_full_analysis_v2/monitor_score001'):
 f.extend([mon/'started.json',mon/'terminal.json'])
value={'status':'frozen','source_sha256':{str(p):sha(p) for p in f},'pillow_version':importlib.metadata.version('Pillow'),'GT_opened':False,'individual_dimensions_read':False,'scores_parsed':False,'existing_prediction_files_opened_for_hash':True,'images':384}
with (D/'plan001.json').open('x') as fp:json.dump(value,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
