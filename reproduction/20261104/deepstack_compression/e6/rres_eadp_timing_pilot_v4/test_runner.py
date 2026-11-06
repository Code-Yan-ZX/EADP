import unittest
from types import SimpleNamespace
import paired_runner as runner
class Tests(unittest.TestCase):
 def test_boundary_before_durable_write(self):
  events=[];clock=iter([10,20,30,40]);sync=lambda:events.append('sync')
  def invoke(b):
   b(SimpleNamespace(shape=(1,4)),None);self.assertIs(b.generation_returned([1,2]),b.raw);events.append('decode');return 'text'
  def persist(x):events.append(('persist',x))
  response,b=runner.measure(invoke,sync,clock=lambda:next(clock),persist=persist)
  self.assertEqual(response,'text');self.assertEqual((b['t0_ns'],b['first_ns'],b['generate_end_ns'],b['response_end_ns']),(10,20,30,40));self.assertEqual(events[-1],('persist',[1,2]));self.assertLess(events.index('decode'),len(events)-1)
 def test_sync_or_first_failure_preserves_returned_raw(self):
  for missing in (False,True):
   stored=[];calls=[]
   def sync():
    calls.append(1)
    if not missing and len(calls)==3:raise RuntimeError('synchronization failed')
   def invoke(b):
    if not missing:b(SimpleNamespace(shape=(1,4)),None)
    b.generation_returned([8,9])
   with self.assertRaises((RuntimeError,ValueError)):runner.measure(invoke,sync,clock=lambda:10+len(calls),persist=stored.append)
   self.assertEqual(stored,[[8,9]])
 def test_decode_failure_preserves_returned_raw(self):
  saved=[];clock=iter([10,20,30])
  def invoke(b):b(SimpleNamespace(shape=(1,4)),None);b.generation_returned([3]);raise RuntimeError('decode')
  with self.assertRaises(RuntimeError):runner.measure(invoke,lambda:None,clock=lambda:next(clock),persist=saved.append)
  self.assertEqual(saved,[[3]])
 def test_return_formats(self):
  self.assertEqual(runner.normalize('rres',[1,2,7,8],[1,2]),[7,8]);self.assertEqual(runner.normalize('eadp',[7,8],[1,2]),[7,8])
  with self.assertRaises(ValueError):runner.normalize('rres',[1,3,7],[1,2])
  with self.assertRaises(ValueError):runner.normalize('eadp',[],[1,2])
if __name__=='__main__':unittest.main()
