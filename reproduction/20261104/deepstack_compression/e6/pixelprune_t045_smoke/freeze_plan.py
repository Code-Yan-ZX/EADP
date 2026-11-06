"""Freeze four post-hoc anchors and CPU expected indices without answers."""
import hashlib,json,os
from pathlib import Path
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixelprune_t045_smoke';R=B/'e1_independent_review';C=B/'e6/pixelprune_threshold32';P=B/'e6/pixelprune_dev384'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
review=read(R/'pixel_threshold32_review.json');s=read(R/'monitor_pixel_threshold32/started.json');t=read(R/'monitor_pixel_threshold32/terminal.json')
if review['status']!='passed' or review['independent_review'] is not True or review['images']!=32 or review['threshold_cases']!=320 or review['GT_opened'] is not False or review['predictions_opened'] is not False or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('calibration review not complete')
rows=[x for x in review['per_image'] if x['threshold']==.45];chosen=[next(x for x in rows if x['task']==task and x['K']>1) for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')];chosen.append(next(x for x in rows if x['K']==1 and x['group_id'] not in {r['group_id'] for r in chosen}))
original=read(P/'plan001.json');omap={(x['task'],str(x['qid'])):x for x in original['anchors']};cal=read(C/'plan001.json');anchors=[]
for x in chosen:
 i=next(i for i,s in enumerate(cal['samples']) if s['identity']['task']==x['task'] and str(s['identity']['qid'])==str(x['qid']) and s['group_id']==x['group_id']);journal=(C/'run001'/('image_%03d'%i)/'journal.jsonl').resolve()
 if review['image_files_sha256'].get(str(journal))!=sha(journal):raise ValueError('calibration journal changed')
 found=[json.loads(line) for line in journal.read_text().splitlines() if json.loads(line).get('event')=='completed' and json.loads(line).get('threshold')==.45]
 if len(found)!=1 or found[0]['selected_count']!=x['K']:raise ValueError('calibration selected count')
 a=dict(omap[(x['task'],str(x['qid']))]);a.update(cpu_journal=str(journal),cpu_K=x['K'],cpu_selected_indices=found[0]['selected_indices']);anchors.append(a)
assert len(anchors)==4 and len({x['group_id'] for x in anchors})==4 and any(x['cpu_K']==1 for x in anchors)
source=dict(original['sha256']);files=[D/'worker.py',D/'launch.py',D/'test_worker.py',D/'freeze_plan.py',P/'plan001.json',C/'plan001.json',R/'pixel_threshold32_review.json',R/'review_pixel_threshold32.py',R/'monitor_pixel_threshold32/started.json',R/'monitor_pixel_threshold32/terminal.json',D/'monitor_test001/started.json',D/'monitor_test001/terminal.json',D/'monitor_test001/stderr.log']+[Path(x['cpu_journal']) for x in anchors]
for f in files:source[str(f.resolve())]=sha(f)
plan={'stage':'post-hoc threshold 0.45 four-DEV natural diagnostic; CPU/GPU selector comparison; no labels/scoring','anchors':anchors,'threshold':.45,'method':'pred_2d','metric':'max','anchored':True,'vit_layer':0,'generation':original['generation'],'budget_source_sha256':original['budget_source_sha256'],'source_commit':original['source_commit'],'calibration_review_sha256':sha(R/'pixel_threshold32_review.json'),'sha256':source,'stale_seconds':1200,'GT_opened':False,'predictions_opened_for_identity':True,'selection_rule':'first CPU calibration row per task with K>1 plus first distinct CPU K=1; post-hoc after DEV results','limitations':['four deliberately chosen examples are not representative','CPU/GPU indices may differ and differences are outcomes, not omitted','no accuracy, equal-K, latency, CONFIRM or FINAL claim','runtime tensors are checked but full high-dimensional visual/DS/KV arrays are not archived']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('frozen',[(x['task'],x['qid'],x['cpu_K']) for x in anchors],sha(D/'plan001.json'))
