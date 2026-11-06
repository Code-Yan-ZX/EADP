"""Full-denominator image-paired descriptive statistics; no file/GPU access."""
import math,statistics
import numpy as np
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');SEED=20261105;DRAWS=10000
def calculate(design,records,draws=DRAWS,seed=SEED,expected_per_task=64):
 if type(draws) is not int or draws<1 or type(seed) is not int:raise ValueError('bootstrap configuration')
 expected={}
 for block in design['blocks']:
  for x in block:
   k=(x['task'],str(x['qid']))
   if k in expected:raise ValueError('duplicate designed identity')
   expected[k]=x
 if len(expected)!=3*expected_per_task or len(records)!=2*len(expected):raise ValueError('full denominator required')
 pairs={}
 for r in records:
  key=(r.get('task'),str(r.get('qid')));arm=r.get('arm')
  if key not in expected or arm not in ('author','wavefront') or (key,arm) in pairs or r.get('group_id')!=expected[key]['group_id']:raise ValueError('unknown or duplicate paired observation')
  b=r.get('boundaries',{});start=b.get('t0_before_image_read_ns');end=b.get('first_token_selected_ns')
  if type(start) is not int or type(end) is not int or not 0<start<end or type(r.get('first_token_id')) is not int or not isinstance(r.get('selected_indices'),list):raise ValueError('invalid observation')
  pairs[(key,arm)]=r
 rng=np.random.default_rng(seed);tasks={};all_rows=[]
 for task in TASKS:
  keys=sorted((k for k in expected if k[0]==task),key=lambda k:(expected[k]['group_id'],k[1]))
  if len(keys)!=expected_per_task or len({expected[k]['group_id'] for k in keys})!=len(keys):raise ValueError('task group denominator')
  entries=[]
  for key in keys:
   a=pairs[(key,'author')];b=pairs[(key,'wavefront')]
   a_ns=a['boundaries']['first_token_selected_ns']-a['boundaries']['t0_before_image_read_ns'];b_ns=b['boundaries']['first_token_selected_ns']-b['boundaries']['t0_before_image_read_ns']
   entries.append({'task':task,'qid':key[1],'group_id':expected[key]['group_id'],'author_ns':a_ns,'wavefront_ns':b_ns,'difference_ns':b_ns-a_ns,'ratio':b_ns/a_ns,'first_token_equal':a['first_token_id']==b['first_token_id'],'eos_ids_equal':a.get('eos_ids')==b.get('eos_ids'),'first_eos_equal':a.get('first_token_is_eos')==b.get('first_token_is_eos'),'ordered_selection_equal':a['selected_indices']==b['selected_indices'],'author_selected_count':len(a['selected_indices']),'wavefront_selected_count':len(b['selected_indices'])})
  diff=np.asarray([x['difference_ns'] for x in entries],dtype=np.float64);rat=np.asarray([x['ratio'] for x in entries],dtype=np.float64);sample=rng.integers(0,len(entries),size=(draws,len(entries)));dm=np.median(diff[sample],axis=1);rm=np.median(rat[sample],axis=1)
  tasks[task]={'images':len(entries),'author_median_ns':statistics.median(x['author_ns'] for x in entries),'wavefront_median_ns':statistics.median(x['wavefront_ns'] for x in entries),'paired_difference_median_ns':float(np.median(diff)),'paired_difference_bootstrap_95_ns':[float(x) for x in np.quantile(dm,[.025,.975])],'paired_ratio_median':float(np.median(rat)),'paired_ratio_bootstrap_95':[float(x) for x in np.quantile(rm,[.025,.975])],'first_token_mismatches':sum(not x['first_token_equal'] for x in entries),'eos_ids_mismatches':sum(not x['eos_ids_equal'] for x in entries),'first_eos_mismatches':sum(not x['first_eos_equal'] for x in entries),'ordered_selection_mismatches':sum(not x['ordered_selection_equal'] for x in entries)}
  all_rows.extend(entries)
 return {'status':'passed','tasks':tasks,'pairs':len(expected),'observations':len(records),'draws':draws,'seed':seed,'bootstrap':'within-task image-group paired resampling, PCG64, median and linear 2.5/97.5% quantiles; uncorrected and exploratory','per_question':all_rows}
