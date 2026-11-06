"""Combine disjoint accepted pilot/expansion outcomes; no data access on import."""
from collections import Counter
BUDGETS=(128,256,512)
def key(row):return (row['task'],str(row['qid']),row['group_id'])
def inventory(pilot,additional,pilot_cases,additional_cases):
 if len(pilot)!=32 or len(additional)!=96 or len({key(x) for x in pilot+additional})!=128:raise ValueError('image partition mismatch')
 expected=lambda rows:{(key(r),k) for r in rows for k in BUDGETS}
 def convert(cases,rows):
  out={}
  for c in cases:
   k=(key(c),c['budget'])
   if k in out:raise ValueError('duplicate case')
   out[k]=c
  if set(out)!=expected(rows):raise ValueError('missing or unexpected cases')
  return out
 left=convert(pilot_cases,pilot);right=convert(additional_cases,additional)
 if set(left)&set(right):raise ValueError('overlapping source cases')
 return left,right

def summarize(pilot,additional,pilot_cases,additional_cases):
 old,new=inventory(pilot,additional,pilot_cases,additional_cases);out={}
 for label,rows in [('observed_pilot_32',old),('added_96',new),('combined_128',{**old,**new})]:
  out[label]={}
  for budget in BUDGETS:
   cases=[v for (_,k),v in rows.items() if k==budget];methods={}
   for method in ('fused','graph'):
    groups=Counter()
    for c in cases:
     v=c.get(method)
     if v is None:groups['unavailable']+=1;continue
     if not isinstance(v.get('ordered_equal'),bool) or not isinstance(v.get('spatial_set_equal'),bool) or not isinstance(v.get('all_repeats_ordered_equal'),bool):raise ValueError('invalid candidate evidence')
     if v['ordered_equal'] and not v['spatial_set_equal']:raise ValueError('order implies equal set')
     groups['strict_order_pass' if v['ordered_equal'] and v['all_repeats_ordered_equal'] else 'strict_order_fail']+=1
     groups['initial_spatial_equal_posthoc' if v['spatial_set_equal'] else 'initial_spatial_different_posthoc']+=1
    expected_count=32 if label=='observed_pilot_32' else 96 if label=='added_96' else 128
    if len(cases)!=expected_count or sum(groups[x] for x in ('strict_order_pass','strict_order_fail','unavailable'))!=expected_count:raise ValueError('denominator')
    methods[method]={'planned':expected_count,**{k:groups[k] for k in ('strict_order_pass','strict_order_fail','unavailable','initial_spatial_equal_posthoc','initial_spatial_different_posthoc')}}
   out[label][str(budget)]=methods
 return out
