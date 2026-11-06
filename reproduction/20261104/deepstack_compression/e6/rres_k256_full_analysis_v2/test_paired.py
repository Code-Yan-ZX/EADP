import unittest
import paired

class BootstrapTests(unittest.TestCase):
 def row(self,q,g,a,b,random):return {'qid':q,'group_id':g,'scores':{'rres':a,'eadp':b,'e1_full':b,**{s:random for s in paired.SEEDS}}}
 def test_pair_directions_group_weight_and_reproducibility(self):
  rows=[self.row('1','a',1,0,.5),self.row('2','a',0,0,.5),self.row('3','b',0,1,.5)]
  x=paired.analyze(rows,replicates=500,seed=9);self.assertEqual(x,paired.analyze(rows,replicates=500,seed=9));self.assertEqual(x['questions'],3);self.assertEqual(x['image_groups'],2);self.assertAlmostEqual(x['rres_minus']['eadp']['difference_pp'],0);self.assertAlmostEqual(x['rres_minus']['random_three_seed_mean']['difference_pp'],-100/6)
 def test_missing_duplicate_and_nonfinite(self):
  a=self.row('1','a',.5,.4,.3);b=self.row('1','b',.4,.2,.1)
  with self.assertRaises(ValueError):paired.analyze([a,b])
  a['scores'][paired.SEEDS[0]]=float('nan')
  with self.assertRaises(ValueError):paired.analyze([a])
if __name__=='__main__':unittest.main()
