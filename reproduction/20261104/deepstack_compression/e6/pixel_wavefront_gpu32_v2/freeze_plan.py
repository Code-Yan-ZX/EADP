"""Freeze post-pilot expansion to the same 32 previously examined DEV images."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
old=B/'e6/pixelprune_threshold32';cpu=B/'e6/pixel_wavefront_cpu';review=B/'e1_independent_review/pixel_wavefront_cpu_review.json';gpu_review=B/'e1_independent_review/pixel_wavefront_gpu_result_review.json'
samples=read(old/'plan001.json')['samples'];result=read(cpu/'result001.json');proof=read(review);pilot=read(gpu_review)
assert len(samples)==32 and result['matches']==96 and proof['status']=='passed' and proof['ordered_matches']==96 and pilot['status']=='passed'
anchors=[]
for i,sample in enumerate(samples):
 values=[x for x in result['rows'] if x['index']==i and x['threshold']==.45]
 assert len(values)==1 and values[0]['ordered_equal'] and sha(sample['capture_path'])==sample['capture_sha256']
 identity=sample['identity'];anchors.append({'task':identity['task'],'qid':str(identity['qid']),'group_id':sample['group_id'],'capture_path':sample['capture_path'],'expected_indices':values[0]['selected_indices']})
assert len({x['group_id'] for x in anchors})==32
files=[*D.glob('*.py'),B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_launch.py',cpu/'plan001.json',cpu/'result001.json',review,B/'e1_independent_review/review_pixel_wavefront_cpu.py',B/'e1_independent_review/monitor_pixel_wavefront_cpu/started.json',B/'e1_independent_review/monitor_pixel_wavefront_cpu/terminal.json',gpu_review,B/'e1_independent_review/review_pixel_wavefront_gpu_result.py',B/'e1_independent_review/monitor_pixel_wavefront_gpu_result/started.json',B/'e1_independent_review/monitor_pixel_wavefront_gpu_result/terminal.json',*[Path(x['capture_path']) for x in anchors]]
files += [Path(x) for x in read(cpu/'plan001.json')['source_sha256']]
root=B.parents[2];ap=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';files += [ap/'run_plan.json',ap/'run_panel.py',root/'reproduction/20261028/qwen/container_runtime.py']
plan={'scope':'post-pilot expansion to 32 DEV images: author vs antidiagonal GPU selector, threshold 0.45','threshold':.45,'anchors':anchors,'source_sha256':{str(x.resolve()):sha(x) for x in sorted(set(files),key=str)},'budget_source_sha256':sha(B/'initial_plan.json'),'model_loaded':False,'GT_opened':False,'new_predictions_opened':False,'limitations':['Expansion selected after seeing four-image pilot; not a fresh independent test.','One synchronized host wall sample per arm/image; no model or end-to-end performance conclusion.','Finite comparison does not prove equality on other inputs or hardware.']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
print('sources',len(plan['source_sha256']),'anchors',len(anchors))
