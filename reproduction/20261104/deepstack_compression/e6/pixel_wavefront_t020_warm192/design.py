"""Frozen deterministic 3x64 cohort and ABBA process schedule. No GPU/GT."""
import hashlib,json
from pathlib import Path
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');ARMS=(('author','wavefront'),('wavefront','author'),('wavefront','author'),('author','wavefront'))
FIELDS=('task','qid','group_id','image_path','image_sha256','message_sha256','image_start','processor')
def rank(seed,purpose,row):
 data=[seed,purpose,row['task'],str(row['qid']),row['group_id'],row['image_sha256']]
 return hashlib.sha256(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def build(rule,cohort,rows,references):
 if rule.get('tasks')!=list(TASKS) or rule.get('process_arm_order')!=[list(a) for a in ARMS] or rule.get('GPU_authorized') is not False or rule.get('GT_opened') is not False:raise ValueError('rule contract')
 bygroup={(r['task'],r['group_id']):r for r in rows}
 if len(bygroup)!=len(rows):raise ValueError('ambiguous task/group')
 selected=[];seen_groups=set()
 for task in TASKS:
  groups=cohort[task]['measured'];warm=cohort[task]['warmup']
  if len(groups)!=64 or len(set(groups))!=64 or len(warm)!=10 or len(set(warm))!=10 or set(groups)&set(warm):raise ValueError('cohort inventory')
  for group in groups:
   r=bygroup[(task,group)];key=(task,str(r['qid']));ref=references[key]
   if r.get('split')!='DEV' or any(str(ref.get(k))!=str(r.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')) or group in seen_groups:raise ValueError('input identity or shared group')
   seen_groups.add(group);selected.append({**{k:r[k] for k in FIELDS[:6]},'qid':str(r['qid']),'image_start':ref['image_start'],'processor':ref['processor']})
 warm_group=cohort[TASKS[0]]['warmup'][0];w=bygroup[(TASKS[0],warm_group)];ref=references[(w['task'],str(w['qid']))]
 if warm_group in seen_groups or any(str(ref.get(k))!=str(w.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('warmup identity or overlap')
 warmup={**{k:w[k] for k in FIELDS[:6]},'qid':str(w['qid']),'image_start':ref['image_start'],'processor':ref['processor']}
 blocks=[]
 for block in range(4):
  items=[]
  for task in TASKS:
   ordered=sorted((x for x in selected if x['task']==task),key=lambda x:(rank(rule['seed'],'partition',x),str(x['qid']),x['group_id']))
   items.extend(ordered[block*16:(block+1)*16])
  items.sort(key=lambda x:(rank(rule['seed'],'block-'+str(block),x),x['task'],str(x['qid']),x['group_id']))
  blocks.append(items)
 processes=[]
 for b,arms in enumerate(ARMS):
  for arm in arms:processes.append({'order':len(processes),'block':b,'arm':arm,'warmup':warmup,'measured':blocks[b]})
 if len(selected)!=192 or any(len(b)!=48 for b in blocks) or len(processes)!=8:raise ValueError('schedule inventory')
 return {'blocks':blocks,'processes':processes,'selected_questions':len(selected),'measured_records':sum(len(p['measured']) for p in processes),'warmup_records':len(processes),'warmup_overlap_measured':False}
