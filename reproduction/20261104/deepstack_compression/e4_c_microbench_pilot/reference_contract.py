"""Extract only same-question E1 identity metadata from an accepted DEV journal."""
FIELDS=('task','qid','group_id','message_sha256','processor','image_start','ordered_ids','spatial_ids','visual_sha256','deepstack_full_sha256','prefill_embed_sha256','attention_mask_sha256','prefill_position_ids','rope_delta','injected_mask','injected_deepstack')
def choose(records,row):
 found=[]
 for r in records:
  if r.get('arm')=='P3-D1' and r.get('forced') is False and (r.get('task'),str(r.get('qid')))==(row['task'],str(row['qid'])):
   if r.get('group_id')!=row['group_id'] or r.get('message_sha256')!=row['message_sha256'] or any(k not in r for k in FIELDS):raise ValueError('E1 reference input mismatch')
   found.append({k:r[k] for k in FIELDS})
 if len(found)!=1:raise ValueError('expected unique P3-D1 reference')
 return found[0]
