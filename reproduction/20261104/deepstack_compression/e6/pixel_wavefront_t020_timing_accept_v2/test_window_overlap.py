import unittest
import window_overlap
class Tests(unittest.TestCase):
 def test_unknowns_not_clean(self):
  self.assertEqual(window_overlap.classify(10,20,'ns',None)['status'],'unknown')
  self.assertEqual(window_overlap.classify(10,20,'ns',[])['status'],'unknown')
  self.assertEqual(window_overlap.classify(10,20,'ns',[{'start_ns':10,'end_ns':11,'time_namespace':'other'}])['status'],'unknown')
 def test_closed_windows_and_malformed(self):
  q=[{'start_ns':1,'end_ns':10,'time_namespace':'ns'},{'start_ns':20,'end_ns':21,'time_namespace':'ns'},{'start_ns':21,'end_ns':25,'time_namespace':'ns'}]
  self.assertEqual(window_overlap.classify(10,20,'ns',q)['overlap_indices'],[0,1])
  with self.assertRaises(ValueError):window_overlap.classify(10,20,'ns',[{'start_ns':10,'end_ns':1,'time_namespace':'ns'}])
if __name__=='__main__':unittest.main()
