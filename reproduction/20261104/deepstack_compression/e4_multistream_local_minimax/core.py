"""Synthetic reference for fixed 2x2, four-stream local minimax."""
import hashlib
import numpy as np
SEEDS=(20261104,20261105,20261106)
def select(streams,image_index):
 if len(streams)!=4 or type(image_index) is not int or image_index<0:raise ValueError('four streams and image index required')
 x=[np.asarray(v,dtype=np.float64) for v in streams]
 if any(a.ndim!=3 or a.shape[:2]!=(32,32) or a.shape[2]<1 or not np.isfinite(a).all() for a in x):raise ValueError('grid or finiteness')
 scales=[np.sqrt(np.mean(np.sum(a*a,axis=-1)))+1e-6 for a in x]
 if not np.isfinite(scales).all():raise ValueError('scale overflow')
 out={'minimax':[],'top_left':[],'main_minimax':[],**{f'random_{s}':[] for s in SEEDS}};radii={k:[] for k in out};ties=0
 for r in range(0,32,2):
  for c in range(0,32,2):
   ids=[r*32+c,r*32+c+1,(r+1)*32+c,(r+1)*32+c+1];points=[a.reshape(1024,-1)[ids] for a in x];dist=np.stack([np.linalg.norm(z[:,None,:]-z[None,:,:],axis=-1)/scale for z,scale in zip(points,scales)])
   joint=dist.max(axis=(0,1));main=dist[0].max(axis=0);minval=joint.min();ties+=int(np.count_nonzero(joint==minval)>1)
   choices={'minimax':int(np.argmin(joint)),'top_left':0,'main_minimax':int(np.argmin(main))}
   for seed in SEEDS:
    raw=f'local-minimax-random-v1|{seed}|{image_index}|{r//2}|{c//2}'.encode();choices[f'random_{seed}']=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%4
   for k,j in choices.items():out[k].append(ids[j]);radii[k].append(float(joint[j]))
   if any(joint[choices['minimax']]>joint[j] for j in choices.values()):raise ValueError('minimax objective violation')
 return {'ids':out,'radii':radii,'tie_blocks':ties,'zero_streams':sum(bool(np.count_nonzero(a)==0) for a in x),'scales':scales}
