"""Freeze 32 existing DEV captures for posthoc t=.2 component diagnostic."""
import hashlib,json,struct
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
cpu=B/'e6/pixel_low_threshold_cpu';old=B/'e6/pixelprune_threshold32';r=B/'e1_independent_review';review=r/'pixel_low_threshold_cpu_review.json';mon=r/'monitor_pixel_low_threshold_cpu'
proof=read(review);started=read(mon/'started.json');terminal=read(mon/'terminal.json');result=read(cpu/'result001.json');samples=read(old/'plan001.json')['samples']
if (proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('case_count')!=96 or proof.get('ordered_matches')!=96 or proof.get('result_sha256')!=sha(cpu/'result001.json') or proof.get('plan_sha256')!=sha(cpu/'plan001.json') or proof.get('reviewer_sha256')!=sha(r/'review_pixel_low_threshold_cpu.py') or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or started.get('worker_pid')!=terminal.get('worker_pid')):raise ValueError('independent CPU proof incomplete')
anchors=[]
for i,s in enumerate(samples):
 rows=[x for x in result['rows'] if x['index']==i and x['threshold']==0.2];checked=[x for x in proof['cases'] if x['index']==i and x['threshold']==0.2]
 if len(rows)!=1 or len(checked)!=1 or not rows[0]['ordered_equal'] or sha(s['capture_path'])!=s['capture_sha256']:raise ValueError('CPU input mismatch')
 ids=rows[0]['selected_indices'];case=checked[0]
 if (case['selected_count']!=len(ids) or case['actual_author_anchored_dispatch'] is not True or case['ordered_indices_sha256']!=hashlib.sha256(struct.pack('<'+'q'*len(ids),*ids)).hexdigest()):raise ValueError('independent CPU selection mismatch')
 identity=s['identity'];anchors.append({'task':identity['task'],'qid':str(identity['qid']),'group_id':s['group_id'],'capture_path':s['capture_path'],'expected_indices':ids})
if len(anchors)!=32 or len({x['group_id'] for x in anchors})!=32:raise ValueError('invalid inventory')
files=[*D.glob('*.py'),B/'initial_plan.json',B/'monitor/supervise.py',B/'e4_c_latency/timing_launch.py',cpu/'plan001.json',cpu/'result001.json',review,r/'review_pixel_low_threshold_cpu.py',mon/'started.json',mon/'terminal.json',*[Path(x['capture_path']) for x in anchors]]
files += [Path(x) for x in read(cpu/'plan001.json')['source_sha256']]
root=B.parents[2];ap=root/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02';files += [ap/'run_plan.json',ap/'run_panel.py',root/'reproduction/20261028/qwen/container_runtime.py']
plan={'scope':'posthoc 32 DEV images: author vs wavefront GPU selector, threshold 0.2','threshold':0.2,'anchors':anchors,'source_sha256':{str(x.resolve()):sha(x) for x in sorted(set(files),key=str)},'budget_source_sha256':sha(B/'initial_plan.json'),'model_loaded':False,'GT_opened':False,'new_predictions_opened':False,'limitations':['Same previously inspected 32 DEV captures, not an independent new test.','One synchronized host wall sample per arm/image; no model or end-to-end speed claim.','Finite component comparison does not prove equality for other inputs or hardware.']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
print('sources',len(plan['source_sha256']),'anchors',len(anchors))
