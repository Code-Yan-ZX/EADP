"""Freeze exactly the accepted E1 DEV 384 inputs and per-input processor references.

This reads accepted E1 DEV mixed prediction journal solely for input identity,
no ground truth or new predictions. No GPU or model weights are loaded.
"""
import hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;B=HERE.parent.parent;ROOT=B.parents[2];sys.path.insert(0,str(B/'e4_c_latency'))
from input_preflight import load_accepted_e1_references

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def main():
 e1=B/'e1';rows,refs=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 if len(rows)!=384 or len(refs)!=384:raise ValueError('DEV denominator mismatch')
 natural=B/'e6/pixelprune_natural3_v3';plan=json.loads((natural/'plan001.json').read_text());recipe=json.loads((ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json').read_text())['recipe'];expected={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}
 if any(recipe[k]!=expected[k] for k in ('max_new_tokens','top_p','top_k','temperature','repetition_penalty')):raise ValueError('E1 recipe mismatch')
 sample=[]
 for r in rows:
  key=(r['task'],str(r['qid']));ref=refs[key]
  if r.get('split')!='DEV':raise ValueError('non-DEV row')
  sample.append({'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'message_sha256':r['message_sha256'],'image_sha256':r['image_sha256'],'processor':ref['processor'],'image_start':ref['image_start']})
 if len({(r['task'],r['qid']) for r in sample})!=384 or len({r['group_id'] for r in sample})!=384:raise ValueError('duplicates in frozen panel')
 paths=[HERE/'worker.py',HERE/'launch.py',HERE/'test_worker.py',HERE/'freeze_plan.py',HERE/'monitor_test001/started.json',HERE/'monitor_test001/terminal.json',HERE/'monitor_test001/stderr.log',natural/'plan001.json',natural/'preflight001.json',natural/'run_smoke001/observations/worker_complete.json',B/'e1_independent_review/pixel_natural3_result_review.json',B/'e1_independent_review/review_pixel_natural3_result.py',B/'e1_independent_review/monitor_pixel_natural3_result/started.json',B/'e1_independent_review/monitor_pixel_natural3_result/terminal.json',e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json',B/'e4_c_latency/input_preflight.py',B/'e4_c_latency/e1_reference_gate.py',B/'e4_panel_v2/panel_v2_imports.py',B/'e4_panel_v2/panel_gate.py',B/'e4/identity_worker.py',B/'e1_analysis/completion.py']
 source=dict(plan['sha256'])
 for f in paths:source[str(f)]=sha(f)
 out={'stage':'full 384 DEV PixelPrune natural generation, author patch before model loading; no labels/scoring','anchors':sample,'sha256':source,'budget_source_sha256':plan['budget_source_sha256'],'source_commit':plan['source_commit'],'method':'pred_2d','metric':'max','threshold':0.0,'vit_layer':0,'generation':expected,'stale_seconds':1200,'GT_opened':False,'predictions_opened_for_identity':True,'limitations':['384 DEV generations; no held-out predictions or labels','threshold zero gives dynamic K; not equal-compute K256','hooks and hashing prohibit timing claims','original E1 DEV mixed prediction journal parsed for processor/input identity','author class patch applied before model construction','full raw sequence stored immediately after generation; high-dimensional tensors retained only as metadata/runtime checks']}
 with (HERE/'plan001.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print('frozen 384 DEV inputs',sha(HERE/'plan001.json'))
if __name__=='__main__':main()
