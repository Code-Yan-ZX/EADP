"""Verify the frozen 18-arm inventory and durable raw generation journal."""
SEEDS=(20261104,20261105,20261106)
def expected(anchors):
 if len(anchors)!=3 or len({x['task'] for x in anchors})!=3:raise ValueError('invalid task anchors')
 return [{'task':a['task'],'qid':str(a['qid']),'group_id':a['group_id'],'seed':seed,'forced':forced} for a in anchors for seed in SEEDS for forced in (False,True)]
def validate(anchors,journal,generated,records,complete):
 jobs=expected(anchors)
 if len(jobs)!=18 or len(journal)!=36 or len(generated)!=18 or len(records)!=18:raise ValueError('incomplete denominator')
 if complete.get('status')!='completed' or complete.get('arms')!=18 or complete.get('natural')!=9 or complete.get('forced33')!=9 or complete.get('GT_opened') is not False or complete.get('scoring_performed') is not False or complete.get('timing_eligible') is not False:raise ValueError('invalid completion')
 for i,job in enumerate(jobs):
  started,done=journal[2*i:2*i+2];exposure=generated[i];rec=records[i]
  if started!={'event':'started',**job} or done!={'event':'recorded',**job} or exposure.get('scheduled')!=job:raise ValueError('journal order or identity')
  raw=exposure.get('raw')
  if not isinstance(raw,dict) or any(rec.get(k)!=v or raw.get(k)!=v for k,v in job.items()):raise ValueError('actual arm identity')
  if any(rec.get(k)!=raw.get(k) for k in ('output_ids','eos_ids','generation_scalars')) or rec.get('protocol')!=raw.get('protocol'):raise ValueError('generated exposure differs from result')
 return True
