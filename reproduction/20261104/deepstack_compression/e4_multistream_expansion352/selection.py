"""Frozen-cohort selection logic: identity metadata only, no tensor or score I/O."""
import hashlib
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
COUNTS={'DocVQA_VAL':117,'TextVQA_VAL':117,'AI2D_TEST':118}
FIELDS=('task','qid','group_id','image_sha256','message_sha256')
def clean(row):
 d={k:str(row[k]) for k in FIELDS}
 if any(not x for x in d.values()):raise ValueError('empty identity')
 return d
def key(row):return row['task'],row['qid']
def sort_key(row):
 h=hashlib.sha256(('multistream-expansion352-pilot-v1|'+row['task']+'|'+row['group_id']+'|'+row['qid']).encode()).hexdigest()
 return h,row['qid']
def select(inputs,pilot):
 rows=[clean(r) for r in inputs];old=[clean(r) for r in pilot]
 if len(rows)!=384 or Counter(r['task'] for r in rows)!={t:128 for t in TASKS} or len({key(r) for r in rows})!=384:raise ValueError('input denominator or duplicate')
 if any(len({r['group_id'] for r in rows if r['task']==t})!=128 for t in TASKS):raise ValueError('within-task group duplicate')
 if len(old)!=32 or Counter(r['task'] for r in old)!={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10} or len({key(r) for r in old})!=32:raise ValueError('anchor denominator or duplicate')
 mapping={key(r):r for r in rows}
 if any(mapping.get(key(r))!=r for r in old):raise ValueError('anchor identity mismatch')
 excluded={key(r) for r in old};remain=[r for t in TASKS for r in sorted((r for r in rows if r['task']==t and key(r) not in excluded),key=sort_key)]
 if Counter(r['task'] for r in remain)!=COUNTS:raise ValueError('residual denominator')
 chosen=[r for t in TASKS for r in [x for x in remain if x['task']==t][:3]]
 membership={}
 for r in rows:membership.setdefault(r['group_id'],set()).add(r['task'])
 shared=sorted(g for g,ts in membership.items() if len(ts)>1)
 oldgroups={r['group_id'] for r in old}
 return {'remaining':remain,'pilot':chosen,'counts':COUNTS,'pilot_counts':{t:3 for t in TASKS},'cross_task_shared_group_count':len(shared),'remaining_rows_in_pilot_groups':sum(r['group_id'] in oldgroups for r in remain)}
