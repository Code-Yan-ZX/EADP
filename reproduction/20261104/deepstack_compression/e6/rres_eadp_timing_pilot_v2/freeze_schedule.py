"""Freeze the existing 32+30 input cohort and the complete process schedule."""
import hashlib,json,os
from pathlib import Path
import schedule
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PILOT=ROOT/'e4_c_multibudget_expansion/pilot_plan001.json'
RRES=ROOT/'e6/rres_k256_dev384/design001.json'
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def construct(pilot,design):
 ids=pilot['input_identities'];groups={r['group_id']:r for r in ids}
 if len(groups)!=62 or len(ids)!=62:raise ValueError('pilot inventory')
 anchors=[];warmups=[]
 for task in schedule.TASKS:
  for name,target in [('measured',anchors),('warmup',warmups)]:
   for gid in pilot['cohort'][task][name]:
    row=groups[gid]
    if row['task']!=task:raise ValueError('cross-task group')
    target.append(row)
 d={(r['task'],str(r['qid'])):r for r in design['anchors']}
 if len(d)!=384:raise ValueError('384-input inventory')
 for row in anchors+warmups:
  reference=d[(row['task'],str(row['qid']))]
  for key in ('group_id','image_sha256','message_sha256'):
   if reference[key]!=row[key]:raise ValueError('identity mismatch '+key)
 jobs=schedule.build(anchors,warmups);schedule.validate(jobs,anchors,warmups)
 return {'status':'frozen','scope':'post-hoc paired timing design only','measured_inputs':anchors,'warmup_inputs':warmups,'processes':jobs,'process_count':12,'warmup_requests':60,'measured_requests':320,'GT_opened':False,'predictions_opened':False,'GPU_authorized':False}
def main():
 out=HERE/'schedule001.json'
 data=construct(json.loads(PILOT.read_text()),json.loads(RRES.read_text()))
 data['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [PILOT,RRES,HERE/'schedule.py',Path(__file__).resolve()]}
 with open(out,'x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
