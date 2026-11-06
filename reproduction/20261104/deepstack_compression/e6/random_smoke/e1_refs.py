"""Extract only E1 natural P3-D1 identity fields after external E1 admission.

The source is a mixed DEV prediction journal: it is parsed; answer text is
not returned here. Caller must perform full source and reviewer admission first.
"""
import hashlib,json
from pathlib import Path
FIELDS=('task','qid','group_id','message_sha256','arm','forced','processor','visual_sha256','deepstack_full_sha256','original_position_ids','full_rope_delta','ordered_ids','generation_scalars','eos_ids')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(path,receipt,anchors):
 if sha(path)!=receipt.journal_sha:raise ValueError('accepted E1 journal changed')
 required={(r['task'],str(r['qid'])):r for r in anchors}
 if len(required)!=3:raise ValueError('expected unique anchors')
 result={k:[] for k in required}
 with Path(path).open() as f:
  for line in f:
   row=json.loads(line);key=(row.get('task'),str(row.get('qid')))
   if key in required and row.get('arm')=='P3-D1' and row.get('forced') is False:
    anchor=required[key]
    if any(row.get(k)!=anchor.get(k) for k in ('task','group_id','message_sha256')):raise ValueError('E1 row binding')
    if any(k not in row for k in FIELDS):raise ValueError('E1 identity fields missing')
    result[key].append({k:row[k] for k in FIELDS})
 if sha(path)!=receipt.journal_sha or any(len(v)!=1 for v in result.values()):raise ValueError('E1 ambiguous or changed')
 return result
