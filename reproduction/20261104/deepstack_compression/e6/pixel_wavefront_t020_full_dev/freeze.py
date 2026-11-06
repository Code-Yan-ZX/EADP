"""Freeze exploratory sources and already-accepted baseline before GPU work."""
import json
import os
from pathlib import Path
import baseline

D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
def main():
    paths=[D/x for x in ('admission.py','baseline.py','compare.py','worker.py','launch.py','finish_guard.py',
                         'snapshot.py','verify.py','freeze.py','cpu_validation.py','test_compare.py','test_verify.py','test_selector.py','test_worker.py')]
    paths += [B/'e6/pixel_wavefront_cpu/candidate.py', B/'e6/pixelprune_t020_accept/verify.py',
              ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py',
              ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/diagnose.py',
              ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json',
              B/'e4_c_latency/timing_launch.py',B/'initial_plan.json']
    source={str(x):baseline.sha(x) for x in paths}
    old=baseline.read(baseline.ORIGINAL/'plan001.json')
    for path,digest in old['sha256'].items():
        if baseline.sha(path)!=digest:raise ValueError('old source changed: '+path)
        source[path]=digest
    p={'stage':'wavefront_t020_full_dev_proposal','questions':384,'GT_opened':False,'scoring_performed':False,
       'predictions_opened':True,'prediction_scope':'accepted DEV predictions for source hashes and later output comparison',
       'source_sha256':source,'budget_source_sha256':baseline.sha(B/'initial_plan.json'),
       'baseline':baseline.validate(),'wavefront_prior':baseline.validate_wavefront_prior(),
       'inner_timeout_seconds':10500,'outer_timeout_seconds':10920,'stale_seconds':1200,'grace_seconds':90,
       'minimum_free_bytes':32*1024**3,'limitations':['post-DEV exploratory','full historical high-dimensional KV/logits unavailable','not a timing comparison']}
    with (D/'plan002.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print('frozen 384-question plan,',len(source),'sources')
if __name__=='__main__':main()
