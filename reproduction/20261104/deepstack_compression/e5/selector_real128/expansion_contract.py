"""Frozen 128 partition and strict candidate identity contracts; CPU-only."""
import hashlib,json
from pathlib import Path
EXPECTED_MANIFEST_SHA256='83171eb8298602706f9a1bad2e523cef0a833092b7b7c4673bef8bc984d92bf8'
BUDGETS=(128,256,512)
FIELDS=('task','qid','group_id','message_sha256','image_sha256')
def key(row):return (row['task'],str(row['qid']),row['group_id'])
def partition(manifest):
 if manifest.get('status')!='frozen' or manifest.get('budgets')!=list(BUDGETS):raise ValueError('invalid frozen sample')
 allrows=manifest['full_128'];pilot=manifest['pilot']
 if len(allrows)!=128 or len(pilot)!=32 or len({key(x) for x in allrows})!=128 or len({key(x) for x in pilot})!=32:raise ValueError('invalid image inventory')
 full={key(x):x for x in allrows};seen={key(x):x for x in pilot}
 if not set(seen)<=set(full) or any(any(full[k].get(f)!=v.get(f) for f in FIELDS) for k,v in seen.items()):raise ValueError('pilot identity changed')
 if len({x['group_id'] for x in allrows})!=128:raise ValueError('duplicate group')
 for x in allrows:
  if any(not isinstance(x.get(f),str) or not x[f] for f in FIELDS):raise ValueError('missing sample field')
 return [x for x in allrows if key(x) not in seen]
def load_manifest(path):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=EXPECTED_MANIFEST_SHA256:raise ValueError('manifest SHA mismatch')
 sample=json.loads(raw);out=partition(sample)
 if len(out)!=96:raise ValueError('expected 96 unobserved pilot groups')
 return sample,out
def selected(author,candidate,k):
 if k not in BUDGETS:raise ValueError('budget')
 for seq in (author,candidate):
  if len(seq)!=k or len(set(seq))!=k or any(type(x) is not int or not 0<=x<1024 for x in seq):raise ValueError('invalid indices')
 aset,bset=set(author),set(candidate)
 return {'ordered_equal':author==candidate,'spatial_set_equal':aset==bset,'ordered_positions_differing':sum(a!=b for a,b in zip(author,candidate)),'first_difference_zero_based':next((i for i,(a,b) in enumerate(zip(author,candidate)) if a!=b),None),'intersection':len(aset&bset),'union':len(aset|bset)}
def performance_eligible(identity,synthetic,author_endpoint,selection):
 return identity is True and synthetic is True and author_endpoint is True and selection.get('ordered_equal') is True
