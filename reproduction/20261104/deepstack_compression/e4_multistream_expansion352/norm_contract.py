"""Lossless 4x1024 float64 norms for independently recomputable features.

Offline recomputation is limited to norms; full original tensor identity is
established online against accepted summaries, not by these stored vectors.
"""
import hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent.parent/'e4_multistream_budget_diagnostic_v2/features.py'
SHA='0a27e3b35deb0c6f8b8a493e820d8504e9100382f5e8f27a197bf0c9c67d3a6f'
def original():
 if hashlib.sha256(P.read_bytes()).hexdigest()!=SHA:raise ValueError('prior feature implementation changed')
 name='_multistream_prior_features_352'
 if name in sys.modules:raise ValueError('cached feature module forbidden')
 s=importlib.util.spec_from_file_location(name,P);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def norms(streams):
 if len(streams)!=4:raise ValueError('four streams required')
 out=[]
 for x in streams:
  x=np.asarray(x,dtype=np.float64)
  if x.ndim==3 and x.shape[0]==1:x=x[0]
  if x.ndim!=2 or x.shape[0]!=1024 or x.shape[1]<1 or not np.isfinite(x).all():raise ValueError('stream shape or values')
  n=np.linalg.norm(x,axis=1)
  if not np.isfinite(n).all():raise ValueError('nonfinite norms')
  out.append(n)
 return np.stack(out)
def features_from_norms(n,f):
 n=np.asarray(n)
 if n.shape!=(4,1024) or n.dtype!=np.float64 or not np.isfinite(n).all() or (n<0).any():raise ValueError('invalid float64 norms')
 ranks=[(f.rank(v)-1)/1023 for v in n];v={}
 for i,name in enumerate(f.FEATURES[:4]):
  mean=float(n[i].mean());v[name]=None if mean==0 else float(n[i].std(ddof=0)/mean)
 for i,name in enumerate(f.FEATURES[4:7]):v[name]=f.pearson(ranks[0],ranks[i+1])
 v[f.FEATURES[7]]=float(np.var(np.stack(ranks),axis=0,ddof=0).mean())
 return v
