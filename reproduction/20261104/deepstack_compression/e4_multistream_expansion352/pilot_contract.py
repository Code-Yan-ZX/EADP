"""Pure validation contract for the later 9-image diagnostic pilot.

This is not an execution entry point, launcher, authorization, or model test.
"""
import math
IDENTITY=('task','qid','group_id','image_sha256','message_sha256')
TIMES=('processor_seconds','visual_and_selection_seconds','feature_seconds','serialization_seconds')
def check_identity(target,actual,reference):
 for x in (target,actual,reference):
  if any(k not in x or not str(x[k]) for k in IDENTITY):raise ValueError('missing identity')
 for k in IDENTITY:
  if str(actual[k])!=str(target[k]) or str(reference[k])!=str(target[k]):raise ValueError('identity mismatch: '+k)
 if reference.get('budget')!=256 or reference.get('ds_on') is not True or reference.get('selection_mode')!='author_greedy' or reference.get('actual_injection_verified') is not True:raise ValueError('reference is not accepted K256 D1')
 for k in ('processor','main_visual','full_deepstack'):
  if k not in actual or k not in reference or actual[k]!=reference[k]:raise ValueError('actual mismatch: '+k)
 if not isinstance(actual['full_deepstack'],list) or len(actual['full_deepstack'])!=3:raise ValueError('three full DS required')
 return True
def check_elapsed(record):
 # null denotes unavailable; missing required phases is not a zero-cost claim.
 if any(k not in record for k in TIMES):raise ValueError('phase absent')
 for k in TIMES:
  v=record[k]
  if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0):raise ValueError('invalid duration')
 return True
def check_complete(selected,records):
 ids=[(r['task'],str(r['qid'])) for r in selected]
 if len(ids)!=9 or len(set(ids))!=9 or len(records)!=9:raise ValueError('pilot denominator')
 for item,identity in zip(records,ids):
  if (item.get('task'),str(item.get('qid')))!=identity or item.get('status') not in ('passed','failed','not_evaluated'):raise ValueError('ordered result identity/status')
 # A failure never becomes a passed pilot, even when all cases are accounted for.
 return all(item['status']=='passed' for item in records)
