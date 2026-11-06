import sys,unittest
from pathlib import Path
import numpy as np
import review_random_full_v2_scores as r
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e6/random_full_analysis_v3'))
import paired
class Tests(unittest.TestCase):
 def test_metrics(self):
  self.assertEqual(r.doc('cat',['cat']),1);self.assertEqual(r.doc('cat',['dog']),0);self.assertEqual(r.option('The answer is B',{'A':'dog','B':'cat'}),'B');self.assertAlmostEqual(r.text_score('yes',['yes']*10,lambda x:x),1)
 def test_independent_bootstrap_weighted_group(self):
  raw=[('1','a',.2,[.3,.4,.5]),('2','a',.5,[.6,.7,.8]),('3','b',.1,[.2,.3,.4])];rows=[];p=[]
  for q,g,e,vals in raw:
   rows.append({'group_id':g,'arms':{'eadp':{'score':e},**{str(s):{'score':x} for s,x in zip(r.SEEDS,vals)}}});p.append({'qid':q,'group_id':g,'scores':{'eadp':e,**dict(zip(r.SEEDS,vals))}})
  ids,mean,boot=r.bootstrap(rows);x=paired.analyze(p);self.assertEqual(ids,['a','b'])
  for i,a in enumerate(r.ARMS):self.assertAlmostEqual(mean[i],x['arm_mean_pct'][a]);self.assertTrue(np.allclose(np.quantile(boot[:,i],[.025,.975]),x['arm_ci95_pct'][a]))
if __name__=='__main__':unittest.main()
