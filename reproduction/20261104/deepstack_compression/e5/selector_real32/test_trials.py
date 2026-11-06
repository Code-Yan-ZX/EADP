import types,unittest
from unittest.mock import patch
import torch
from trials import author_case,bench

def order(w,k):
 x=torch.arange(k,dtype=torch.long)
 return x.flip(0) if w.reshape(-1)[0].item()==0 else x
class Event:
 def __init__(self,*args,**kwargs):pass
 def record(self):pass
 def synchronize(self):pass
 def elapsed_time(self,_):return 1.0
class FakeGraph:
 made=[]
 def __init__(self,sim,importance,k):self.k=k;self.seen=[];self.made.append(self)
 def __call__(self,sim,importance):self.seen.append(float(importance[0]));return order(importance,self.k)
class Tests(unittest.TestCase):
 def test_actual_author_wrapper_cleanup_and_budget(self):
  native=types.SimpleNamespace()
  def greedy(importance,similarity,k):return (order(importance,k)[None,:],torch.tensor([k]))
  native._greed_select_impl=greedy
  class Pruner:
   def __init__(self,visual_token_num,alpha,beta,visual_dim,spatial_merge_size,device):self.visual_token_num=visual_token_num;self.alpha=alpha;self.beta=beta;self.visual_dim=visual_dim;self.spatial_merge_size=spatial_merge_size
   def to(self,_):return self
   def eval(self):return self
   def __call__(self,image,mean,seq,grid):
    selected,_=native._greed_select_impl(torch.ones(1,1024),torch.ones(1,1024,1024),self.visual_token_num);return (image[selected[0].sort().values],[self.visual_token_num])
  native.VisualTokenPruner=Pruner;image=torch.arange(4096,dtype=torch.float32).reshape(1024,4);args=(image,torch.ones(1,4),torch.ones(1,2,4),torch.tensor([[1,64,64]]));meta={'processor':'p','main_visual':'m','full_deepstack':'d'};record={'processor':'p','main_visual':'m','full_deepstack':'d','ordered_ids':list(range(128)),'spatial_ids':list(range(128)),'pruned_visual':('meta',(128,4))};tensor_meta=lambda x:('meta',tuple(x.shape))
  r,importance,sim,ids,output=author_case(torch,native,args,128,meta,record,tensor_meta);self.assertTrue(r['e4_order_equal']);self.assertTrue(r['e4_pruned_hash_equal']);self.assertIs(native._greed_select_impl,greedy)
  def broken(*args):raise ValueError('failure')
  native._greed_select_impl=broken
  with self.assertRaisesRegex(ValueError,'failure'):author_case(torch,native,args,128,meta,record,tensor_meta)
  self.assertIs(native._greed_select_impl,broken)
 def test_one_graph_per_k_reuses_changed_inputs(self):
  FakeGraph.made=[];native=types.SimpleNamespace(_greed_select_impl=lambda w,s,k:(order(w,k)[None,:],None));kernel=types.SimpleNamespace(fused_greedy=lambda s,w,k:(order(w,k),None,None));sim=torch.ones(1,1024,1024);w=torch.ones(1,1024);graphs={};events=[]
  with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'memory_reserved',return_value=0):
   first,m,setup=bench(torch,native,kernel,FakeGraph,graphs,sim,w,128,order(w,128),events.append,{'fused':True,'graph':True});w2=w.clone();w2[0,0]=0;second,m2,setup2=bench(torch,native,kernel,FakeGraph,graphs,sim,w2,128,order(w2,128),events.append,{'fused':True,'graph':True})
  self.assertEqual(len(FakeGraph.made),1);self.assertEqual(set(FakeGraph.made[0].seen),{0.,1.});self.assertTrue(first['graph_equals_author']);self.assertTrue(second['graph_equals_author']);self.assertIsNotNone(setup);self.assertIsNone(setup2);self.assertEqual(len(m),15);self.assertEqual(len(m2),15)
 def test_graph_failure_does_not_drop_other_methods(self):
  class BadGraph:
   def __init__(self,*args):raise ValueError('unsupported graph')
  native=types.SimpleNamespace(_greed_select_impl=lambda w,s,k:(order(w,k)[None,:],None));kernel=types.SimpleNamespace(fused_greedy=lambda s,w,k:(order(w,k),None,None));sim=torch.ones(1,1024,1024);w=torch.ones(1,1024);graphs={};events=[]
  with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'memory_reserved',return_value=0):first,m,setup=bench(torch,native,kernel,BadGraph,graphs,sim,w,128,order(w,128),events.append,{'fused':True,'graph':True})
  self.assertEqual(setup['status'],'failed');self.assertIsNone(first['graph_equals_author']);self.assertEqual(sum(x['status']=='unavailable' for x in m),5);self.assertEqual(sum(x['status']=='measured' for x in m),10)
 def test_synthetic_or_candidate_mismatch_never_timed(self):
  native=types.SimpleNamespace(_greed_select_impl=lambda w,s,k:(order(w,k)[None,:],None));calls={'fused':0}
  def wrong(s,w,k):calls['fused']+=1;return (order(w,k).flip(0),None,None)
  kernel=types.SimpleNamespace(fused_greedy=wrong);sim=torch.ones(1,1024,1024);w=torch.ones(1,1024);events=[]
  with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'memory_reserved',return_value=0):
   initial,m,_=bench(torch,native,kernel,FakeGraph,{},sim,w,128,order(w,128),events.append,{'fused':True,'graph':True})
   self.assertFalse(initial['fused_equals_author']);self.assertEqual(calls['fused'],1);self.assertTrue(all(x['status']=='unavailable' for x in m if x['implementation']=='fused'))
   calls['fused']=0;initial,m,_=bench(torch,native,kernel,FakeGraph,{},sim,w,128,order(w,128),events.append,{'fused':False,'graph':False});self.assertEqual(calls['fused'],0);self.assertTrue(all(x['status']=='unavailable' for x in m if x['implementation'] in ('fused','graph')))
 def test_changed_author_endpoint_stops_all_timing(self):
  native=types.SimpleNamespace(_greed_select_impl=lambda w,s,k:(order(w,k)[None,:],None));calls={'fused':0}
  def candidate(s,w,k):calls['fused']+=1;return (order(w,k),None,None)
  kernel=types.SimpleNamespace(fused_greedy=candidate);sim=torch.ones(1,1024,1024);w=torch.ones(1,1024);events=[]
  with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'memory_reserved',return_value=0):initial,m,_=bench(torch,native,kernel,FakeGraph,{},sim,w,128,order(w,128).flip(0),events.append,{'fused':True,'graph':True})
  self.assertFalse(initial['author_equals_endpoint']);self.assertEqual(m,[]);self.assertEqual(calls['fused'],0);self.assertEqual(events,[])
if __name__=='__main__':unittest.main()
