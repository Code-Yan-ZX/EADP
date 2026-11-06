import unittest
from types import SimpleNamespace
from pilot_contract import ARMS,make_schedule,validate_schedule,validate_generation,verify_expected
from observer import count_calls
class Tests(unittest.TestCase):
 def setUp(self):self.anchors=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)]
 def test_order_and_pairing(self):
  x=make_schedule(self.anchors);self.assertEqual(len(x),9);self.assertTrue(validate_schedule(x,self.anchors));self.assertEqual(x,make_schedule(self.anchors))
  x[0],x[1]=x[1],x[0]
  with self.assertRaises(ValueError):validate_schedule(x,self.anchors)
 def test_refuses_non_dev_and_duplicate_group(self):
  self.anchors[0]['split']='FINAL'
  with self.assertRaises(ValueError):make_schedule(self.anchors)
  self.anchors[0]['split']='DEV';self.anchors[0]['group_id']=self.anchors[1]['group_id']
  with self.assertRaises(ValueError):make_schedule(self.anchors)
 def test_pixel_counts_real_return_and_restores(self):
  data=[object(),object(),object()];m=SimpleNamespace(fn=lambda:[data]);old=m.fn
  with count_calls(m,'fn','pixel',1) as counts:self.assertIs(m.fn()[0],data)
  self.assertEqual(counts,[3]);self.assertIs(m.fn,old)
 def test_eadp_count_and_exception_restore(self):
  class T:shape=(1,256)
  m=SimpleNamespace(fn=lambda:(T(),None));old=m.fn
  with self.assertRaisesRegex(RuntimeError,'boom'):
   with count_calls(m,'fn','eadp',1) as counts:
    m.fn();self.assertEqual(counts,[256]);raise RuntimeError('boom')
  self.assertIs(m.fn,old)
 def test_wrong_shape_and_extra_calls_fail_closed(self):
  m=SimpleNamespace(fn=lambda:[[1]])
  with self.assertRaises(ValueError):
   with count_calls(m,'fn','pixel',1):m.fn();m.fn()
  m.fn=lambda:[[1],[2]]
  with self.assertRaises(ValueError):
   with count_calls(m,'fn','pixel',1):m.fn()
 def test_natural_boundary_and_exact_same_arm_reference(self):
  self.assertTrue(validate_generation([3,9],[9],101,100,True)['eos'])
  self.assertTrue(validate_generation([3]*2048,[9],1,100,False)['truncated'])
  for args in [([3],[9],100,100,True),([3],[9],1,100,False),([9],[],1,1,False)]:
   with self.assertRaises(ValueError):validate_generation(*args)
  self.assertTrue(verify_expected([1,9],[1,9]))
  with self.assertRaises(ValueError):verify_expected([1,9],[2,9])
if __name__=='__main__':unittest.main(verbosity=2)
