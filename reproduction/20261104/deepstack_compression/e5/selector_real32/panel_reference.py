"""Read only identity fields from the accepted E4 DEV mixed record journal."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;sys.path.insert(0,str(BASE/'e4_c_latency'))
from identity_gate import sha,supervised
from latency_admission import terminal
FIELDS=('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','processor','main_visual','pruned_visual','full_deepstack','ordered_ids','spatial_ids','actual_injection_verified')

def accepted():
 d=BASE/'e4_panel_v2';r=BASE/'e1_independent_review';proof=d/'panel_003_verification.json';fs=d/'monitor_verify_panel003/started.json';ft=d/'monitor_verify_panel003/terminal.json';review=r/'e4panel003_review.json';rs=r/'monitor_e4panel003/started.json';rt=r/'monitor_e4panel003/terminal.json';reviewer=r/'review_e4panel.py'
 supervised(fs,ft,d/'panel_verify.py',proof);supervised(rs,rt,reviewer,review);terminal(d/'monitor_panel003/started.json',d/'monitor_panel003/terminal.json');f=json.loads(proof.read_text());v=json.loads(review.read_text());path=d/'panel_003/records.jsonl'
 if (f.get('status')!='passed' or f.get('n_questions')!=384 or f.get('all_arms_validated')!=3072 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('full_tensor_reconstruction') is not False or f.get('verifier_sha256')!=sha(d/'panel_verify.py') or f.get('files_sha256',{}).get(str(path.resolve()))!=sha(path) or v.get('status')!='passed' or v.get('arms_verified')!=3072 or v.get('n_verified')!=384 or v.get('reviewer_sha256')!=sha(reviewer) or v.get('formal_proof_sha256')!=sha(proof) or v.get('formal_verifier_started_sha256')!=sha(fs) or v.get('formal_verifier_terminal_sha256')!=sha(ft)):raise ValueError('E4 proof mismatch')
 return {'formal_sha256':sha(proof),'review_sha256':sha(review),'review_terminal_sha256':sha(rt),'records_sha256':sha(path)}

def get_records(pilot,identity):
 path=BASE/'e4_panel_v2/panel_003/records.jsonl'
 if sha(path)!=identity['records_sha256']:raise ValueError('E4 records changed')
 keys={(r['task'],str(r['qid']),k):r for r in pilot for k in (128,256,512)};found={}
 with path.open() as f:
  for line in f:
   row=json.loads(line);key=(row.get('task'),str(row.get('qid')),row.get('budget'))
   if key not in keys or row.get('ds_on') is not True:continue
   expected=keys[key]
   if key in found or any(row.get(k)!=expected.get(k) for k in ('task','group_id','image_sha256','message_sha256')) or row.get('selection_mode')!='author_greedy' or row.get('actual_injection_verified') is not True or any(k not in row for k in FIELDS):raise ValueError('invalid E4 question reference')
   found[key]={k:row[k] for k in FIELDS}
 if len(found)!=96 or sha(path)!=identity['records_sha256']:raise ValueError('incomplete/changed E4 reference')
 return found

def compare_common(meta,record):
 return meta['processor']==record['processor'] and meta['main_visual']==record['main_visual'] and meta['full_deepstack']==record['full_deepstack']
