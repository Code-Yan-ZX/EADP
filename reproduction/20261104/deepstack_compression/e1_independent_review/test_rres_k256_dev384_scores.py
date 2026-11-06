import importlib.util,sys,unittest
from pathlib import Path
from unittest.mock import patch
I=Path(__file__).resolve().parent;B=I.parent
spec=importlib.util.spec_from_file_location('_rres_score_review_test',I/'review_rres_k256_dev384_scores.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
sys.path.insert(0,str(B/'e6/rres_k256_full_analysis_v2'));import paired
class ReviewTests(unittest.TestCase):
 def test_independent_bootstrap_matches_grouped_multiseed(self):
  rows=[];main=[]
  for i in range(128):
   v={'rres':(i%4)/4,'eadp':(i%5)/5,'e1_full':(i%6)/6,**{s:((i+j)%7)/7 for j,s in enumerate(r.SEEDS)}}
   rows.append({'qid':str(i),'group_id':str(i//2),'scores':v});main.append({'group_id':str(i//2),'rres':{'score':v['rres']},'eadp':{'score':v['eadp']},'e1_full':{'score':v['e1_full']},'random':{str(s):{'score':v[s]} for s in r.SEEDS}})
  out=paired.analyze(rows);ids,mean,boot=r.bootstrap(main);self.assertEqual(out['image_groups'],len(ids))
  for i,n in enumerate(r.ARMS):
   self.assertAlmostEqual(out['arm_mean_pct'][n],mean[i],places=10)
   if i:self.assertAlmostEqual(out['rres_minus'][n]['difference_pp'],mean[0]-mean[i],places=10)
 def test_missing_completion_stops_before_read(self):
  with patch.object(r,'term',side_effect=RuntimeError('not completed')),patch.object(r,'read',side_effect=AssertionError('premature read')):
   with self.assertRaisesRegex(RuntimeError,'not completed'):r.main(I/'dummy.json')
if __name__=='__main__':unittest.main()
