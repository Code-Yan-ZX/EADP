import contextlib,unittest
from types import SimpleNamespace
from bridge import make_measure
class FakeTokens:
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return [[1]*32]
class BridgeTest(unittest.TestCase):
 def setUp(self):
  self.events=[];self.torch=SimpleNamespace(inference_mode=contextlib.nullcontext,cuda=SimpleNamespace(synchronize=lambda:None));self.wrapper=SimpleNamespace(model=SimpleNamespace(generate=lambda:FakeTokens()))
  self.job={'mode':'C-ON','protocol':'fixed32'};self.row={'image_start':4}
 def execute(self,wrapper,native,torch,row,mode,protocol,sync):
  self.events.append('generate');wrapper.model.generate();return {'mode':mode,'protocol':protocol,'output_ids':[1]*32}
 def memory(self,torch,model,run):
  result=run();self.events.append('probe_exit');return result,{'allocated_peak_bytes':10,'reserved_peak_bytes':12,'kv_cache_bytes_final':4,'kv_cache_bytes_peak_logical':5,'kv_observed_forwards':33}
 def test_partial_output_precedes_memory_probe_exit(self):
  f=make_measure(self.torch,self.wrapper,None,self.execute,self.memory,'C-ON');out=f(self.row,self.job,lambda v:self.events.append('partial'))
  self.assertEqual(self.events,['generate','partial','probe_exit']);self.assertEqual(out['memory_only']['kv_observed_forwards'],33)
 def test_nested_generate_wrapper_restores_original(self):
  def nested(wrapper,native,torch,row,mode,protocol,sync):
   previous=wrapper.model.generate;captured=[]
   def inner(*a,**kw):
    result=previous(*a,**kw);captured.append(result.detach().cpu().tolist()[0]);return result
   wrapper.model.generate=inner
   try:wrapper.model.generate()
   finally:wrapper.model.generate=previous
   return {'mode':mode,'protocol':protocol,'output_ids':captured[0]}
  original=self.wrapper.model.generate;events=[]
  out=make_measure(self.torch,self.wrapper,None,nested,self.memory,'C-ON')(self.row,self.job,events.append)
  self.assertEqual(events[0]['output_ids'],out['output_ids']);self.assertIs(self.wrapper.model.generate,original)
 def test_output_saved_even_if_wrapper_fails_after_generate(self):
  def failed(wrapper,native,torch,row,mode,protocol,sync):
   wrapper.model.generate();raise RuntimeError('post-generation failure')
  f=make_measure(self.torch,self.wrapper,None,failed,self.memory,'C-ON');original=self.wrapper.model.generate
  with self.assertRaises(RuntimeError):f(self.row,self.job,lambda v:self.events.append(v['phase']))
  self.assertEqual(self.events,['generate_return']);self.assertIs(self.wrapper.model.generate,original)
 def test_disagreeing_final_tokens_are_rejected(self):
  def bad(wrapper,native,torch,row,mode,protocol,sync):
   wrapper.model.generate();return {'mode':mode,'protocol':protocol,'output_ids':[2]}
  f=make_measure(self.torch,self.wrapper,None,bad,self.memory,'C-ON')
  with self.assertRaisesRegex(ValueError,'differs'):f(self.row,self.job,lambda v:self.events.append(v['phase']))
  self.assertEqual(self.events,['generate_return'])
 def test_rejects_other_mode_before_generation(self):
  f=make_measure(self.torch,self.wrapper,None,self.execute,self.memory,'C-SKIP')
  with self.assertRaises(ValueError):f(self.row,self.job,lambda v:None)
  self.assertEqual(self.events,[])
 def test_inconsistent_stats_rejected_after_partial(self):
  def bad(t,m,run):
   value=run();return value,{'allocated_peak_bytes':10,'reserved_peak_bytes':1,'kv_cache_bytes_final':4,'kv_cache_bytes_peak_logical':5,'kv_observed_forwards':1}
  f=make_measure(self.torch,self.wrapper,None,self.execute,bad,'C-ON')
  with self.assertRaises(ValueError):f(self.row,self.job,lambda v:self.events.append('partial'))
  self.assertEqual(self.events,['generate','partial'])
if __name__=='__main__':unittest.main()
