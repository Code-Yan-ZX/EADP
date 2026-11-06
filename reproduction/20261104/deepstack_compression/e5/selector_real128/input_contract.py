"""Author instruction-ID to compact prefill mapping; real hook inputs remain authority."""
def compact_indices(original,instruction,image_start,compact_length):
 if not instruction or image_start<0 or len(original)<image_start+1024 or compact_length!=len(original)-768:raise ValueError('invalid original/compact lengths')
 matches=[i for i in range(len(original)-len(instruction)+1) if original[i:i+len(instruction)]==instruction]
 if len(matches)!=1:raise ValueError('instruction IDs missing or nonunique')
 positions=list(range(matches[0],matches[0]+len(instruction)))
 if any(image_start<=i<image_start+1024 for i in positions):raise ValueError('instruction crosses visual span')
 mapped=[i if i<image_start else i-768 for i in positions]
 if any(i<0 or i>=compact_length for i in mapped):raise ValueError('compact position out of range')
 return mapped

def compare_real(torch,actual,args,instruction_ids,image_start):
 if len(args)!=4:raise ValueError('expected four real selector arguments')
 ids=actual['processor']['input_ids'];prefill=actual['prefill']['inputs_embeds'];mask=actual['prefill']['visual_pos_masks']
 if ids.ndim!=2 or ids.shape[0]!=1 or prefill.ndim!=3 or prefill.shape[0]!=1 or mask.shape!=prefill.shape[:2]:raise ValueError('unexpected processor/prefill batch')
 index=compact_indices(ids[0].detach().cpu().tolist(),instruction_ids,image_start,prefill.shape[1])
 if any(bool(mask[0,i]) for i in index):raise ValueError('mapped text overlaps actual visual mask')
 actual_seq=args[2];actual_mean=args[1];mapped=prefill[:,index,:]
 return {'instruction_tokens':len(index),'mapped_indices':index,'sequence_equal':bool(torch.equal(mapped,actual_seq)),'mean_equal':bool(torch.equal(mapped.mean(dim=1),actual_mean)),'selector_sequence_shape':list(actual_seq.shape),'selector_mean_shape':list(actual_mean.shape)}

def bind_rows(identities,prepared):
 rows=[];index={(r['task'],str(r['qid'])):r for r in prepared['rows']}
 if len(index)!=384 or len(identities)!=96:raise ValueError('full DEV input inventory')
 for identity in identities:
  row=index.get((identity['task'],str(identity['qid'])))
  if row is None or row.get('split')!='DEV' or any(row.get(k)!=identity.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('frozen input mismatch')
  if (row['task'],str(row['qid'])) not in prepared['refs']:raise ValueError('E1 reference missing')
  rows.append(row)
 if len({r['group_id'] for r in rows})!=96:raise ValueError('group inventory')
 return rows
