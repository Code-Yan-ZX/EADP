import unittest
from core import calculate
class TestCore(unittest.TestCase):
 def rows(self):return [{'group_id':f'g{i:03}','k020':128 if i<32 else 256 if i<64 else 512 if i<96 else 513,'k0':700,'s020':.4,'s0':.7,'e4':.5} for i in range(128)]
 def test_threshold_edges_full_denominator(self):
  a=calculate(self.rows());self.assertEqual([a[str(k)]['fallback_count'] for k in (128,256,512)],[32,64,96]);self.assertAlmostEqual(a['256']['mean_selected_k'],(64*700+32*512+32*513)/128);self.assertAlmostEqual(a['256']['contrasts']['versus_t0']['difference_pp'],-15)
 def test_invalid_denominator_and_scores(self):
  a=self.rows();a[1]['group_id']=a[0]['group_id']
  with self.assertRaises(ValueError):calculate(a)
  a=self.rows();a[0]['s020']=float('nan')
  with self.assertRaises(ValueError):calculate(a)
 def test_order_does_not_change_resampling(self):self.assertEqual(calculate(self.rows()),calculate(list(reversed(self.rows()))))
if __name__=='__main__':unittest.main(verbosity=2)
