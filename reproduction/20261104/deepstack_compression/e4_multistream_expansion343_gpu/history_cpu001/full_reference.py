"""Extract frozen nine complete E1 P3-D1 identity records, never answers."""
FIELDS=('task','qid','group_id','message_sha256','processor','image_start','ordered_ids','spatial_ids','visual_sha256','deepstack_full_sha256','prefill_embed_sha256','attention_mask_sha256','prefill_position_ids','rope_delta','injected_mask','injected_deepstack')
E4_MAP={'processor':'processor','image_start':'image_start','ordered_ids':'ordered_ids','spatial_ids':'spatial_ids','visual_sha256':'main_visual','deepstack_full_sha256':'full_deepstack','prefill_embed_sha256':'prefill_embeds','attention_mask_sha256':'prefill_mask','prefill_position_ids':'prefill_position_ids','rope_delta':'adjusted_delta','injected_mask':'injected_mask','injected_deepstack':'injected_deepstack'}
def collect(selected,records,slim,e4):
 wanted={(r['task'],str(r['qid'])):r for r in selected}
 if len(wanted)!=343 or len(slim)!=384 or len(e4)!=343:raise ValueError('reference inventory')
 out={}
 for r in records:
  key=(r.get('task'),str(r.get('qid')))
  if key not in wanted or r.get('arm')!='P3-D1' or r.get('forced') is not False:continue
  if key in out or any(k not in r for k in FIELDS):raise ValueError('duplicate or incomplete P3-D1')
  row=wanted[key];ref=slim.get(key);q=e4.get(key)
  if ref is None or q is None or any(str(row.get(k))!=str(ref.get(k)) or str(row.get(k))!=str(q.get(k)) for k in ('task','qid','group_id','message_sha256','image_sha256')) or r.get('group_id')!=row['group_id'] or r.get('message_sha256')!=row['message_sha256'] or ref.get('processor')!=r['processor'] or ref.get('image_start')!=r['image_start'] or q.get('budget')!=256 or q.get('ds_on') is not True or q.get('actual_injection_verified') is not True or any(r[a]!=q.get(b) for a,b in E4_MAP.items()):raise ValueError('P3-D1 / E4 identity mismatch')
  out[key]={k:r[k] for k in FIELDS}
 if len(out)!=343:raise ValueError('missing P3-D1')
 return out
