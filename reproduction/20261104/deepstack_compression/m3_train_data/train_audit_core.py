"""Label-free image graph primitives; exhaustive Hamming <=4 candidates."""
import hashlib
from collections import defaultdict

class UnionFind:
 def __init__(self): self.p=[]
 def add(self):
  i=len(self.p);self.p.append(i);return i
 def find(self,i):
  while self.p[i]!=i:
   self.p[i]=self.p[self.p[i]];i=self.p[i]
  return i
 def union(self,a,b):
  a=self.find(a);b=self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)

def bucket_keys(v):
 if not 0<=v<2**64:raise ValueError('pHash outside 64 bits')
 return tuple((j,(v>>start)&((1<<(12 if j==4 else 13))-1)) for j,start in enumerate((0,13,26,39,52)))

class Graph:
 def __init__(self):
  self.uf=UnionFind();self.items=[];self.raw={};self.rgb={};self.semantic={};self.eval_groups={};self.buckets=defaultdict(list);self.near_edges=0
 def add(self,raw,rgb,phash,semantic=None,eval_group=None,is_eval=False):
  if raw in self.raw:
   i=self.raw[raw];item=self.items[i]
   if item['rgb']!=rgb or item['phash']!=phash:raise ValueError('same raw hash, different fingerprints')
   item['is_eval']|=is_eval
  else:
   i=self.uf.add();self.raw[raw]=i
   self.items.append({'raw':raw,'rgb':rgb,'phash':phash,'is_eval':is_eval})
   if rgb in self.rgb:self.uf.union(i,self.rgb[rgb])
   else:self.rgb[rgb]=i
   candidates=set()
   for key in bucket_keys(phash):candidates.update(self.buckets[key])
   for j in candidates:
    if (phash^self.items[j]['phash']).bit_count()<=4:self.uf.union(i,j);self.near_edges+=1
   for key in bucket_keys(phash):self.buckets[key].append(i)
  if semantic is not None:
   if semantic in self.semantic:self.uf.union(i,self.semantic[semantic])
   else:self.semantic[semantic]=i
  if eval_group is not None:
   if eval_group in self.eval_groups:self.uf.union(i,self.eval_groups[eval_group])
   else:self.eval_groups[eval_group]=i
  return i
 def components(self):
  groups=defaultdict(list)
  for i in range(len(self.items)):groups[self.uf.find(i)].append(i)
  return groups

def group_id(items):
 return hashlib.sha256(('group|'+'|'.join(sorted(x['raw'] for x in items))).encode()).hexdigest()

def fit_val(groups):
 ordered=sorted(groups,key=lambda g:(hashlib.sha256(('20261105|TRAIN_FIT_VAL|'+g).encode()).hexdigest(),g))
 n=len(ordered)*4//5
 return {g:('fit' if i<n else 'validation') for i,g in enumerate(ordered)}
