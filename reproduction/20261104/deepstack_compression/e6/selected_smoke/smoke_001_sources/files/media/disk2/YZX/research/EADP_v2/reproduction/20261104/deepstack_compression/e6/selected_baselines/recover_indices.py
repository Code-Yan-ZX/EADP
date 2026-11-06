"""Recover author pruner IDs from exact unique rows; fail on ambiguity.

Diagnostic preparation only; no model hooks, execution authority or labels.
Matching is bitwise, including BF16 storage, and does not re-run selection.
"""
import hashlib

def recover(torch,full,selected):
 if full.ndim!=2 or selected.ndim!=2 or full.shape[1]!=selected.shape[1] or full.dtype!=selected.dtype or len(selected)>len(full):raise ValueError('incompatible token matrices')
 f=full.detach().contiguous().cpu().view(torch.uint8).numpy();s=selected.detach().contiguous().cpu().view(torch.uint8).numpy();index={}
 for i,row in enumerate(f):index.setdefault(hashlib.sha256(row.tobytes()).digest(),[]).append(i)
 ids=[]
 for row in s:
  candidates=[i for i in index.get(hashlib.sha256(row.tobytes()).digest(),[]) if (f[i]==row).all()]
  if len(candidates)!=1:raise ValueError('selected feature is missing or ambiguous')
  ids.append(candidates[0])
 if len(set(ids))!=len(ids) or ids!=sorted(ids):raise ValueError('non-unique or non-spatial selection')
 return ids
