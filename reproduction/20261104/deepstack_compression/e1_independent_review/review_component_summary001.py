"""Recompute all component summaries at image level, preserving all measurements."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
B=Path(__file__).resolve().parent.parent;D=B/'e4_c_microbench_summary';A=B/'e1_independent_review';F=B/'e4_c_microbench_full'
def get(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(d):
 s=get(d/'started.json');t=get(d/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid']
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();term(D/'monitor_summary001');term(A/'monitor_component_full001');proof=get(A/'component_full001_review.json');assert proof['status']=='passed' and proof['images']==32 and proof['jobs']==1920
r=get(D/'result_001.json');schedule=get(B/'e4_c_microbench/schedule001.json');groups=collections.defaultdict(list)
for line in (F/'full_001/journal.jsonl').open():
 e=json.loads(line)
 if e['status']=='measured':groups[(e['image_index'],e['job']['component'])].append(e['gpu_event_ms'])
assert len(groups)==96 and all(len(v)==20 for v in groups.values());comps=['selector','ds_merger_three_streams','ds_injection'];table=[]
for i,id in enumerate(schedule['input_identities']):table.append({'task':id['task'],'qid':str(id['qid']),'group_id':id['group_id'],'median_gpu_ms':{c:statistics.median(groups[(i,c)]) for c in comps}})
assert r['per_image']==table
for subset,res in [(table,r['overall'])]+[([v for v in table if v['task']==task],d) for task,d in r['by_task'].items()]:
 assert res['n_images']==len(subset)
 for c in comps:
  vals=[x['median_gpu_ms'][c] for x in subset];assert res['components'][c]=={'median_of_image_medians_ms':statistics.median(vals),'min_image_median_ms':min(vals),'max_image_median_ms':max(vals)}
assert {k:v['n_images'] for k,v in r['by_task'].items()}=={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}
out={'status':'passed','independent_review':True,'images':32,'jobs':1920,'comparisons':'per image median of 20 repetitions then median and range of 32 image medians','GT_opened':False,'scoring_performed':False,'summary_sha256':sha(D/'result_001.json'),'run_journal_sha256':sha(F/'full_001/journal.jsonl'),'run_review_sha256':sha(A/'component_full001_review.json'),'reviewer_sha256':sha(__file__),'limitations':['32-image expansion was post hoc','known brief Python stat window cannot be mapped to specific measurement; all retained','DS staging excludes decoder addition and component measurements are not TTFT'],'errors':[]};a.out.write_text(json.dumps(out,indent=2)+'\n');print('passed')
