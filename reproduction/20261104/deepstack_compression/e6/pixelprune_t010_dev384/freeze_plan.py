"""Freeze post-hoc threshold .1 full DEV contract after four-image review."""
import hashlib,json,os
from pathlib import Path
import smoke_gate
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixelprune_t010_dev384';OLD=B/'e6/pixelprune_dev384';S=B/'e6/pixelprune_t010_smoke';R=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=read(OLD/'plan001.json');review=R/'pixel_t010_result_review.json';smoke=S/'plan001.json';candidate={'smoke_review_sha256':sha(review),'smoke_plan_sha256':sha(smoke)};smoke_gate.validate(candidate)
if len(old['anchors'])!=384 or len({x['group_id'] for x in old['anchors']})!=384:raise ValueError('DEV inventory changed')
t=read(D/'monitor_test002/terminal.json');s=read(D/'monitor_test002/started.json')
if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('CPU tests incomplete')
source=dict(old['sha256']);extra=[D/'worker.py',D/'launch.py',D/'smoke_gate.py',D/'test_worker.py',D/'test_smoke_gate.py',D/'freeze_plan.py',OLD/'plan001.json',smoke,review,R/'review_pixel_t010_result.py',R/'monitor_pixel_t010_result/started.json',R/'monitor_pixel_t010_result/terminal.json',D/'monitor_test002/started.json',D/'monitor_test002/terminal.json',D/'monitor_test002/stderr.log']
for p in extra:source[str(p.resolve())]=sha(p)
plan={'stage':'post-hoc threshold .1 full 384 DEV natural generation; no scoring','anchors':old['anchors'],'sha256':source,'budget_source_sha256':old['budget_source_sha256'],'source_commit':old['source_commit'],'method':'pred_2d','metric':'max','threshold':.1,'anchored':True,'vit_layer':0,'generation':old['generation'],'stale_seconds':1200,'GT_opened':False,'predictions_opened_for_identity':True,'previous_smoke_predictions_hashed':True,'smoke_review_sha256':sha(review),'smoke_plan_sha256':sha(smoke),'limitations':['threshold selected after prior DEV studies and 32-image CPU calibration; quality is post-hoc exploratory','dynamic K is not equal-K or equal-compute','full high-dimensional visual/DS/KV tensors are not retained','four-sample CPU/GPU identity does not guarantee all 384 inputs']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('frozen',len(plan['anchors']),sha(D/'plan001.json'))
