"""Bind four accepted prediction records to every frozen DEV question."""
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
ARMS=('div:author_stock','div:adapted','cd:author_stock','cd:adapted')

def build_dev_panel(panel,inputs):
 """Bind the DEV subset of the accepted split to the accepted author inputs."""
 dev=[r for r in panel if r.get('split')=='DEV']
 rows={}
 for row in inputs:
  key=(row.get('task'),str(row.get('qid')))
  if key in rows:raise ValueError('duplicate accepted input')
  rows[key]=row
 if len(dev)!=384 or len(rows)!=384:raise ValueError('DEV input denominator mismatch')
 out=[];seen=set()
 for row in dev:
  key=(row.get('task'),str(row.get('qid')))
  if key in seen or key not in rows or row.get('group_id')!=rows[key].get('group_id'):raise ValueError('DEV split and input identity differ')
  seen.add(key);inp=rows[key]
  if any(not isinstance(inp.get(k),str) or len(inp[k])!=64 for k in ('image_sha256','message_sha256')):raise ValueError('accepted input digest missing')
  out.append({**row,'image_sha256':inp['image_sha256'],'message_sha256':inp['message_sha256']})
 if seen!=set(rows):raise ValueError('accepted input outside DEV')
 return out

def inventory(panel,records):
 if len(panel)!=384 or Counter(x.get('task') for x in panel)!={x:128 for x in TASKS} or len(records)!=1536:raise ValueError('DEV denominator mismatch')
 byquestion={};bygroup=set()
 for row in panel:
  key=(row['task'],str(row['qid']))
  if row.get('split')!='DEV' or key in byquestion or row.get('group_id') in bygroup:raise ValueError('DEV identity/duplicate mismatch')
  byquestion[key]={};bygroup.add(row['group_id'])
 rows={(x['task'],str(x['qid'])):x for x in panel}
 for rec in records:
  key=(rec.get('task'),str(rec.get('qid')));arm=str(rec.get('method'))+':'+str(rec.get('path'))
  if key not in byquestion or arm not in ARMS or arm in byquestion[key] or rec.get('forced') is not False:raise ValueError('missing/duplicate/extra natural arm')
  row=rows[key]
  if any(rec.get(k)!=row.get(k) for k in ('group_id','image_sha256','message_sha256')):raise ValueError('prediction input identity mismatch')
  if not isinstance(rec.get('answer'),str) or type(rec.get('eos')) is not bool or type(rec.get('truncated')) is not bool or not isinstance(rec.get('output_ids'),list) or any(type(i) is not int for i in rec['output_ids']):raise ValueError('output metadata invalid')
  byquestion[key][arm]=rec
 if any(set(arms)!=set(ARMS) for arms in byquestion.values()):raise ValueError('incomplete four-arm question')
 return byquestion
