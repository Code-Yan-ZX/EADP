"""Accepted E1/E4 DEV identity only; these mixed journals contain predictions."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import supervised,sha

def e4_records(rows):
 panel=BASE/'e4_panel_v2';review=BASE/'e1_independent_review';formal=panel/'panel_003_verification.json';independent=review/'e4panel003_review.json';fs=panel/'monitor_verify_panel003/started.json';ft=panel/'monitor_verify_panel003/terminal.json';rs=review/'monitor_e4panel003/started.json';rt=review/'monitor_e4panel003/terminal.json';reviewer=review/'review_e4panel.py'
 supervised(fs,ft,panel/'panel_verify.py',formal);supervised(rs,rt,reviewer,independent);p=json.loads(formal.read_text());q=json.loads(independent.read_text());path=(panel/'panel_003/records.jsonl').resolve()
 if p.get('status')!='passed' or p.get('n_questions')!=384 or p.get('all_arms_validated')!=3072 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('verifier_sha256')!=sha(panel/'panel_verify.py') or p.get('files_sha256',{}).get(str(path))!=sha(path):raise ValueError('E4 formal identity changed')
 if q.get('status')!='passed' or q.get('n_verified')!=384 or q.get('arms_verified')!=3072 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('formal_proof_sha256')!=sha(formal) or q.get('formal_verifier_started_sha256')!=sha(fs) or q.get('formal_verifier_terminal_sha256')!=sha(ft) or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('E4 independent identity changed')
 expected={(r['task'],str(r['qid'])):r for r in rows};found={}
 if len(expected)!=384:raise ValueError('DEV input duplicates')
 with path.open() as f:
  for line in f:
   r=json.loads(line)
   if r.get('budget')!=256 or r.get('ds_on') is not True or r.get('forced') is not False:continue
   key=(r.get('task'),str(r.get('qid')));row=expected.get(key)
   if row is None or key in found or any(row.get(k)!=r.get(k) for k in ('group_id','image_sha256','message_sha256')):raise ValueError('E4 record input mismatch')
   fields=('processor','main_visual','full_deepstack','full_position_ids','full_delta','original_length','image_start','generation_scalars','eos_ids')
   if any(k not in r for k in fields):raise ValueError('E4 reference field missing')
   found[key]={k:r[k] for k in fields}
 if set(found)!=set(expected):raise ValueError('E4 reference inventory incomplete')
 return found

def validate_reference(r,ref):
 for a,b in (('processor','processor'),('main_visual','main_visual'),('full_deepstack','full_deepstack'),('rope_delta','full_delta'),('generation_scalars','generation_scalars'),('eos_ids','eos_ids')):
  if r[a]!=ref[b]:raise ValueError('real E4 input/recipe identity mismatch '+a)
 start=ref['image_start'];length=ref['original_length'];spatial=r['spatial_ids']
 if len(spatial)!=256 or spatial!=sorted(set(spatial)) or any(type(i) is not int or i<0 or i>=1024 for i in spatial):raise ValueError('invalid StackTok visual index')
 keep=list(range(start))+[start+i for i in spatial]+list(range(start+1024,length));pos=ref['full_position_ids']
 expected=[[ [axis[0][i] for i in keep] ] for axis in pos]
 if r['prefill_position_ids']!=expected or r['decode_trace'][0]['position_ids']!=expected:raise ValueError('actual P3 position mismatch')
 mask=[[False]*start+[True]*256+[False]*(length-start-1024)]
 if r['injected_mask']!=mask or len(r['injected_deepstack'])!=3:raise ValueError('actual DS mask mismatch')
 return True

def validate_before_selection(inp,main,ds,positions,delta,start,ref):
 from identity_records import tensor_meta
 actual={k:tensor_meta(inp[k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
 if actual!=ref['processor'] or tensor_meta(main)!=ref['main_visual'] or [tensor_meta(x) for x in ds]!=ref['full_deepstack'] or positions.detach().cpu().tolist()!=ref['full_position_ids'] or delta.detach().cpu().tolist()!=ref['full_delta'] or inp['input_ids'].shape[1]!=ref['original_length'] or start!=ref['image_start']:
  raise ValueError('pre-generation processor/vision/P3 reference mismatch')
 return True
