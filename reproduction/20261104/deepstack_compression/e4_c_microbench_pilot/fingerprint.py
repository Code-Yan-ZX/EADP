"""Canonical complete tensor-tree fingerprints, outside measured intervals."""
import hashlib,json,math
def tree(torch,value):
 if torch.is_tensor(value):
  x=value.detach().contiguous().cpu()
  return {'type':'tensor','dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
 if isinstance(value,(tuple,list)):return {'type':'tuple' if isinstance(value,tuple) else 'list','items':[tree(torch,x) for x in value]}
 if isinstance(value,dict):
  if any(not isinstance(k,str) for k in value):raise TypeError('dictionary keys must be strings')
  return {'type':'dict','items':{k:tree(torch,value[k]) for k in sorted(value)}}
 if value is None or type(value) in (str,int,bool):return {'type':type(value).__name__,'value':value}
 if type(value) is float and math.isfinite(value):return {'type':'float','value':value}
 raise TypeError('unsupported or nonfinite output')
def fingerprint(torch,value):
 encoded=json.dumps(tree(torch,value),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
 return hashlib.sha256(encoded).hexdigest()
