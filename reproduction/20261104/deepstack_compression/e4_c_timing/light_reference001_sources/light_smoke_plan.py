"""Fixed three-DEV-anchor light-adapter identity plan; no GPU or labels."""
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')

def make_plan(anchors):
 if len(anchors)!=3 or [x.get('task') for x in anchors]!=list(TASKS):raise ValueError('need ordered B anchors')
 if len({x.get('group_id') for x in anchors})!=3 or any(not x.get('group_id') or not x.get('qid') or not x.get('message_sha256') or not x.get('image_sha256') for x in anchors):raise ValueError('invalid B anchor identity')
 jobs=[]
 for index,row in enumerate(anchors):
  for forced in ((False,True) if index==0 else (False,)):
   for path in ('fresh_on_reference','C-ON','C-DROP','C-SKIP'):
    jobs.append({'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'forced':forced,'path':path,'order':len(jobs)})
 if len(jobs)!=16:raise ValueError('wrong diagnostic count')
 return {'stage':'light_adapter_identity','anchors':anchors,'jobs':jobs,'natural_anchors':3,'forced_anchors':1,
         'fresh_on_references':4,'lightweight_generations':12,'total_generations':16,'forced_decode_steps':32,
         'comparison':'fresh ON vs accepted E4 endpoints, lightweight ON vs fresh ON, lightweight DROP/SKIP vs accepted B',
         'scope':'DEV diagnostic only; no GT, scoring or timing claims'}
