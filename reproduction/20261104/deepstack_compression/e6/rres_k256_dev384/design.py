"""Deterministic copy of the already frozen DEV input identities; no answers."""
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
FIELDS=('task','qid','group_id','message_sha256','image_sha256','processor','image_start')
def panel(anchors):
 if not isinstance(anchors,list) or len(anchors)!=384:raise ValueError('384 frozen identities required')
 result=[]
 for raw in anchors:
  if not isinstance(raw,dict) or any(k not in raw for k in FIELDS):raise ValueError('missing identity field')
  a={k:raw[k] for k in FIELDS};a['qid']=str(a['qid'])
  if a['task'] not in TASKS or not a['qid'] or not a['group_id'] or type(a['image_start']) is not int or a['image_start']<0:raise ValueError('invalid identity')
  if any(not isinstance(a[k],str) or len(a[k])!=64 or any(x not in '0123456789abcdef' for x in a[k]) for k in ('message_sha256','image_sha256')):raise ValueError('invalid digest')
  if not isinstance(a['processor'],dict) or not all(k in a['processor'] for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')):raise ValueError('processor identity')
  result.append(a)
 if Counter(a['task'] for a in result)!={t:128 for t in TASKS} or len({(a['task'],a['qid']) for a in result})!=384 or len({a['group_id'] for a in result})!=384:raise ValueError('denominator or duplicate')
 return result
