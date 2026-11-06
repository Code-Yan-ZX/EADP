"""Same author processor for nine identities, without model or CUDA."""
def check(torch,wrapper,config,selected,rows,refs,e4,validate):
 idx={(r['task'],str(r['qid'])):r for r in rows};checked=[]
 if len(idx)!=384 or len(selected)!=343:raise ValueError('processor sample denominator')
 for identity in selected:
  key=(identity['task'],str(identity['qid']));row=idx[key];ref=refs[key];record=e4[key]
  if any(str(row.get(k))!=str(identity.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')) or record['processor']!=ref['processor']:raise ValueError('processor reference identity')
  actual=wrapper._processor_inputs(wrapper._build_messages(row['message'],dataset=row['task']))
  validate(torch,actual,ref['processor'],config.image_token_id,ref['image_start'])
  if not bool(torch.all(actual['attention_mask']==1)):raise ValueError('unexpected attention padding')
  checked.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'image_start':ref['image_start'],'processor':ref['processor']});del actual
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
 return checked
