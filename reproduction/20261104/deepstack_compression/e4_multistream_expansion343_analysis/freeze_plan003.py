import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[B/'e5/selector_real32/sample_manifest_001.json',B/'e4_multistream_expansion352/selection001.json',B/'e4_multistream_expansion343/plan_freeze001.json',B/'e1_independent_review/e4_multistream_budget_v2_result_review.json',B/'e1_independent_review/e4_multistream_expansion352_pilot_v4_result_review.json']
out={'status':'frozen','plan_sha256':sha(D/'PLAN.md'),'identity_and_prior_sha256':{str(p):sha(p) for p in paths},'previous_32_and_9_results_already_observed':True,'343_completed_before_this_repair':True,'independent_identity_review_processed_343_norms':True,'repair_scope':'mechanical prior path and module cache; analysis values not opened by this freeze','new_343_values_opened':False,'GT_opened':False,'scoring_performed':False,'GPU_used':False}
with (D/'plan_freeze003.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
