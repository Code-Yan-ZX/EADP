import copy,itertools,unittest
from contract import admit_release,audit

class MemoryHandshakeTests(unittest.TestCase):
 def setUp(self):
  self.e={'mode':'C-ON','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-1234-abcd','host_pid':123,'start_ticks':456,'boot_id':'boot','max_sample_gap_ns':15}
  self.r={'mode':self.e['mode'],'nonce':self.e['nonce'],'container_id':self.e['container_id'],'cuda_initialized':False,'namespace_pid':1}
  self.s=dict(self.e,status='sampling',monotonic_ns=10);self.first={'type':'sample','query_start_ns':10,'query_end_ns':11,'monotonic_ns':11,'other_gpu_processes':[],'used_bytes':None,**{k:self.e[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')}}
  self.samples=[self.first,dict(self.first,query_start_ns=19,query_end_ns=20,monotonic_ns=20,used_bytes=1048576),dict(self.first,query_start_ns=29,query_end_ns=30,monotonic_ns=30)]
  self.terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
 def test_valid_lifetime_and_peak(self):
  release=admit_release(self.e,self.r,self.s,self.first);proof=audit(self.e,release,self.samples,self.terminal,29)
  self.assertEqual(proof['nvml_process_sampled_peak_bytes'],1048576)
 def test_identity_and_premature_release_rejected(self):
  for kind in ('nonce','cuda','pid','container','sampler','target_visible'):
   r=copy.deepcopy(self.r);s=copy.deepcopy(self.s);first=copy.deepcopy(self.first)
   if kind=='nonce':r['nonce']='c'*32
   if kind=='cuda':r['cuda_initialized']=True
   if kind=='pid':r['namespace_pid']=3
   if kind=='container':r['container_id']='c'*64
   if kind=='sampler':s['start_ticks']=457
   if kind=='target_visible':first['used_bytes']=1048576
   with self.subTest(kind=kind),self.assertRaises(ValueError):admit_release(self.e,r,s,first)
 def test_concurrency_missing_target_incomplete_coverage_and_failure_rejected(self):
  release=admit_release(self.e,self.r,self.s,self.first)
  for kind in ('other','target','coverage','terminal','truncated','gap','wrong_pid','crossing_exit','slow_query'):
   samples=copy.deepcopy(self.samples);terminal=copy.deepcopy(self.terminal);end=29
   if kind=='other':samples[1]['other_gpu_processes']=[{'pid':999}]
   if kind=='target':samples[1]['used_bytes']=None
   if kind=='coverage':end=31
   if kind=='terminal':terminal['worker_returncode']=1
   if kind=='truncated':samples.pop(0)
   if kind=='gap':samples[1].update(query_start_ns=27,query_end_ns=28,monotonic_ns=28)
   if kind=='wrong_pid':samples[1]['host_pid']=999
   if kind=='crossing_exit':samples[-1]['query_start_ns']=28
   if kind=='slow_query':samples[1].update(query_start_ns=19,query_end_ns=29,monotonic_ns=29)
   with self.subTest(kind=kind),self.assertRaises(ValueError):audit(self.e,release,samples,terminal,end)
if __name__=='__main__':unittest.main()

class SamplerTests(unittest.TestCase):
 def test_process_lifetime_and_first_sample_handshake(self):
  from sampler import sample_process
  e=MemoryHandshakeTests();e.setUp();events=[];ready=[];ticks=itertools.count(10);alive_count=[True,True,True,False]
  def alive(_):return alive_count.pop(0)
  queries=iter(['','GPU-1234-abcd, 123, 10',''])
  result=sample_process(e.e,alive,lambda:next(queries),lambda:next(ticks),lambda _:None,events.append,lambda s,f:ready.append((s,f)),lambda _:18,1,100)
  self.assertEqual(result['samples'],3);self.assertEqual(len(ready),1);self.assertEqual(events[-1]['status'],'completed')
 def test_other_process_fails_and_preserves_failure(self):
  from sampler import sample_process
  e=MemoryHandshakeTests();e.setUp();events=[];ticks=itertools.count(10)
  with self.assertRaisesRegex(ValueError,'other GPU process'):
   sample_process(e.e,lambda _:True,lambda:'GPU-1234-abcd, 999, 10',lambda:next(ticks),lambda _:None,events.append,lambda *_:None,lambda _:18,1,100)
  self.assertEqual(events[-1]['type'],'failure')

class ReusedPIDTests(unittest.TestCase):
 def test_target_pid_after_original_process_exit_fails_closed(self):
  from sampler import sample_process
  e=MemoryHandshakeTests();e.setUp();events=[];liveness=iter((True,False));ticks=itertools.count(10)
  with self.assertRaisesRegex(ValueError,'identity ended'):
   sample_process(e.e,lambda _:next(liveness),lambda:'GPU-1234-abcd, 123, 10',lambda:next(ticks),lambda _:None,events.append,lambda *_:None,lambda _:18,1,100)
  self.assertEqual(events[-1]['type'],'failure')

class StoppedObservationTests(unittest.TestCase):
 def test_query_crossing_stopped_event_is_followed_by_new_query(self):
  from sampler import sample_process
  e=MemoryHandshakeTests();e.setUp();events=[];ticks=itertools.count(10);alive=iter((True,True,True,False,False))
  queries=iter(('', 'GPU-1234-abcd, 123, 10', '', ''))
  result=sample_process(e.e,lambda _:next(alive),lambda:next(queries),lambda:next(ticks),lambda _:None,events.append,lambda *_:None,lambda _:20,1,100)
  samples=[x for x in events if x.get('type')=='sample'];self.assertEqual(result['samples'],4)
  self.assertLess(samples[-2]['query_start_ns'],20);self.assertGreaterEqual(samples[-1]['query_start_ns'],20)
