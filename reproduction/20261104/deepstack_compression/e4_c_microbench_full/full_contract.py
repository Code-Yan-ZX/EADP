"""Predeclared one-image slicing of the untouched 1920-job schedule."""
COMPONENTS=('selector','ds_merger_three_streams','ds_injection')
def split_schedule(payload):
 s=payload['schedule'];images=s['images'];jobs=s['jobs'];inputs=payload['input_identities']
 if len(images)!=32 or len({tuple(x) for x in images})!=32 or len(inputs)!=32 or len(jobs)!=1920 or tuple(s['components'])!=COMPONENTS:raise ValueError('full source inventory mismatch')
 result=[]
 for i,(image,identity) in enumerate(zip(images,inputs)):
  if [identity['task'],identity['group_id']]!=image:raise ValueError('input order mismatch')
  chunk=jobs[60*i:60*(i+1)]
  if any(x.get('order')!=60*i+j or [x.get('task'),x.get('group_id')]!=image for j,x in enumerate(chunk)):raise ValueError('full job order mismatch')
  if any(sum(x['component']==c for x in chunk)!=20 for c in COMPONENTS):raise ValueError('component count mismatch')
  normalized=[dict(x,order=j,global_order=x['order']) for j,x in enumerate(chunk)]
  result.append({'stage':'pilot','images':[image],'components':list(COMPONENTS),'jobs':normalized})
 return result
