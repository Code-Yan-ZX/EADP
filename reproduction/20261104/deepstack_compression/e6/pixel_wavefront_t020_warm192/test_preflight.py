import unittest
import preflight_impl
class PreflightTest(unittest.TestCase):
 def test_inventory_and_overlap(self):
  blocks=[[{'task':'fake','qid':str(i)} for i in range(j*48,(j+1)*48)] for j in range(4)];w={'task':'fake','qid':'warm'};d={'selected_questions':192,'measured_records':384,'warmup_records':8,'blocks':blocks,'processes':[{'warmup':w} for _ in range(8)]}
  self.assertEqual(len(preflight_impl.inventory(d)),193)
  d['processes'][0]['warmup']=blocks[0][0]
  with self.assertRaises(ValueError):preflight_impl.inventory(d)
if __name__=='__main__':unittest.main()
