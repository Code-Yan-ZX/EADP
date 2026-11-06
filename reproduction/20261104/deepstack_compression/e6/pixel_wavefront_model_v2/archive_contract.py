"""CPU-only reconstruction of the actual archived prefill tensors.

The archive contains actual tensor bytes, unlike the per-step KV fingerprints;
this does not recreate or independently verify GPU execution or full KV bytes.
"""
import hashlib
from pathlib import Path

NAMES=('prefill_embeddings','prefill_mask','prefill_positions','visual_mask','prefill_logits','injected_ds0','injected_ds1','injected_ds2')
def digest(b):return hashlib.sha256(b).hexdigest()
def validate_archive(record):
 import numpy as np,torch
 path=Path(record['archive_path']);raw=path.read_bytes()
 if digest(raw)!=record['archive_sha256']:raise ValueError('archive SHA mismatch')
 metadata=record.get('prefill_tensors',{})
 if set(metadata)!=set(NAMES):raise ValueError('archive tensor inventory')
 tensors={}
 with np.load(path,allow_pickle=False) as loaded:
  if set(loaded.files)!=set(NAMES):raise ValueError('archive data inventory')
  for name in NAMES:
   meta=metadata[name];array=loaded[name]
   if array.dtype!=np.uint8 or digest(array.tobytes())!=meta['sha256']:raise ValueError('archive tensor SHA '+name)
   dtype_name=meta['dtype'].removeprefix('torch.');dtype=getattr(torch,dtype_name,None)
   if not isinstance(dtype,torch.dtype):raise ValueError('unsupported tensor dtype')
   try:tensors[name]=torch.from_numpy(array.copy()).view(dtype).reshape(meta['shape'])
   except (RuntimeError,ValueError) as e:raise ValueError('tensor shape/dtype mismatch') from e
 if tensors['prefill_mask'].tolist()!=record['prefill_mask'] or tensors['prefill_positions'].tolist()!=record['prefill_position_ids'] or tensors['visual_mask'].tolist()!=record['visual_mask']:raise ValueError('archived positions/masks mismatch')
 if metadata['prefill_embeddings']!=record['prefill_embeddings'] or metadata['prefill_logits']!=record['prefill_logits'] or [metadata['injected_ds'+str(i)] for i in range(3)]!=record['injected_deepstack']:raise ValueError('archive metadata mismatch')
 main=tensors['prefill_embeddings'][tensors['visual_mask'].bool()].contiguous();m=record['main_visual']
 if str(main.dtype)!=m['dtype'] or list(main.shape)!=m['shape'] or digest(main.view(torch.uint8).numpy().tobytes())!=m['sha256']:raise ValueError('actual selected main visual mismatch')
 return True
