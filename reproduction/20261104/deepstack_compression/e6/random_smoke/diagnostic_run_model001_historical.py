"""Compose the 18-arm random model diagnostic without granting GPU authority.

Inputs must already be independently admitted: E1 source journal and frozen
input rows, selection manifest, and the exact model/runtime/dependencies. The
caller also owns launcher supervision, the GPU lock and retained exit proof.
"""
import hashlib
from pathlib import Path

def run(torch,wrapper,native,rows,references,frozen,anchors,out,generate,selection,budget_runner,random_adapter,cache_factory,identity,run_arm,inventory):
 out=Path(out);arrays=out/'arrays'
 if not out.is_dir() or arrays.exists():raise ValueError('new output directory required')
 inventory.validate_inventory(rows,references,frozen,anchors)
 if wrapper.visual_token_num!=256 or wrapper.pruner.visual_token_num!=256:raise ValueError('expected K256 pruner')
 arrays.mkdir()
 def one(row,reference,seed,forced,on_generated):
  key=(row['task'],str(row['qid']),seed);choice=frozen[key]
  selected=selection(row['image_sha256'],seed)
  if selected['spatial_ids']!=choice['spatial_ids'] or selected['derived_seed64']!=choice['derived_seed64']:raise ValueError('runtime selector differs from frozen manifest')
  filename=hashlib.sha256(str((key,forced)).encode()).hexdigest()+'.npz'
  def archive(tensors):
   import numpy as np
   target=arrays/filename
   data={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
   with target.open('xb') as f:
    np.savez(f,**data);f.flush()
    import os;os.fsync(f.fileno())
   digest=hashlib.sha256(target.read_bytes()).hexdigest()
   return {'kind':'full_arrays','path':str(target),'sha256':digest}
  result=run_arm(torch,wrapper,native,row,seed,reference,selection,budget_runner,random_adapter,generate,on_generated,archive,identity,cache_factory,forced)
  if result.get('derived_seed64')!=choice['derived_seed64'] or result.get('actual_injection_verified') is not True:raise ValueError('random seed or injection identity')
  return result
 return inventory.execute(rows,references,frozen,anchors,out,one)
