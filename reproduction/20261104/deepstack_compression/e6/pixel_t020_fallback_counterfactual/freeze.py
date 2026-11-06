"""Freeze code/plan/proof metadata without opening per-question score records."""
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
s=read(D/'monitor_cpu001/started.json');t=read(D/'monitor_cpu001/terminal.json')
if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('CPU tests incomplete')
files=[D/x for x in ('plan001.json','core.py','run.py','test_core.py','freeze.py','monitor_cpu001/started.json','monitor_cpu001/terminal.json','monitor_cpu001/stderr.log')]
for tag in ('t020','t0'):
 if tag=='t020':A=B/'e6/pixelprune_t020_dev384_analysis';proof=R/'pixel_t020_dev384_scores_review.json';review=R/'review_pixel_t020_dev384_scores.py';mon=R/'monitor_pixel_t020_dev384_scores'
 else:A=B/'e6/pixelprune_dev384_analysis';proof=R/'pixel_dev384_scores_review.json';review=R/'review_pixel_dev384_scores.py';mon=R/'monitor_pixel_dev384_scores'
 p=read(proof)
 if p['status']!='passed' or p['independent_review'] is not True or p['questions']!=384 or p['errors']!=[]:raise ValueError('unaccepted score source')
 files.extend((proof,review,A/'monitor_score001/started.json',A/'monitor_score001/terminal.json',mon/'started.json',mon/'terminal.json'))
out={'status':'frozen','utc':datetime.now(timezone.utc).isoformat(),'scope':'source and prior proof metadata only, before this counterfactual parses item records','source_sha256':{str(p.resolve()):sha(p) for p in files},'score_records_opened':False,'GT_opened':False,'generation_performed':False}
with (D/'source_freeze001.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
