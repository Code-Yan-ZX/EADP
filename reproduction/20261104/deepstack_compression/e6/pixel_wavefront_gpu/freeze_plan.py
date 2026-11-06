"""Freeze the four previously verified examples and every source/data dependency."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
old=read(B/'e6/pixelprune_threshold32/plan001.json');s=read(B/'e6/pixelprune_t045_smoke/plan001.json');review=read(B/'e1_independent_review/pixel_t045_result_review.json');wave=read(B/'e1_independent_review/pixel_wavefront_cpu_review.json');assert review['status']=='passed' and review['images']==4 and wave['status']=='passed' and wave['ordered_matches']==96
anchors=[]
for x in s['anchors']:
 match=[z for z in old['samples'] if z['identity']['task']==x['task'] and str(z['identity']['qid'])==x['qid']];assert len(match)==1
 capture=Path(match[0]['capture_path']);assert sha(capture)==match[0]['capture_sha256'];anchors.append({'task':x['task'],'qid':x['qid'],'group_id':x['group_id'],'capture_path':str(capture),'expected_indices':x['cpu_selected_indices']})
assert {(x['task'],x['qid']) for x in anchors}=={('DocVQA_VAL','16539'),('DocVQA_VAL','31078'),('TextVQA_VAL','39421'),('AI2D_TEST','368100')}
files=[*D.glob('*.py'),B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_launch.py',B/'e6/pixelprune_t045_smoke/plan001.json',B/'e1_independent_review/pixel_t045_result_review.json',B/'e1_independent_review/monitor_pixel_t045_result/started.json',B/'e1_independent_review/monitor_pixel_t045_result/terminal.json',B/'e1_independent_review/review_pixel_t045_result.py',B/'e1_independent_review/pixel_wavefront_cpu_review.json',B/'e1_independent_review/review_pixel_wavefront_cpu.py',B/'e1_independent_review/monitor_pixel_wavefront_cpu/started.json',B/'e1_independent_review/monitor_pixel_wavefront_cpu/terminal.json',*[Path(x['capture_path']) for x in anchors]]
files += [Path(x) for x in read(B/'e6/pixel_wavefront_cpu/plan001.json')['source_sha256']]
root=B.parents[2];ap=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';files += [ap/'run_plan.json',ap/'run_panel.py',root/'reproduction/20261028/qwen/container_runtime.py']
plan={'scope':'post-hoc four-image author anchored selector versus antidiagonal CPU/GPU diagnostic','threshold':.45,'anchors':anchors,'source_sha256':{str(x.resolve()):sha(x) for x in sorted(set(files),key=str)},'budget_source_sha256':sha(B/'initial_plan.json'),'model_loaded':False,'GT_opened':False,'new_predictions_opened':False,'limitations':['Engineering variant of author anchored max, not a new pruning method.','One synchronized host wall sample per arm/image; no model or end-to-end inference conclusion.','Same four post-hoc inputs; no extrapolation or GPU selection guarantee.']}
(D/'plan001.json').write_text(json.dumps(plan,indent=2)+'\n');print('sources',len(plan['source_sha256']))
