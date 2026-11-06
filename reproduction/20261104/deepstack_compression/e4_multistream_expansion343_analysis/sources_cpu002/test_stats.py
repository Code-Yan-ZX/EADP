import unittest
import numpy as np
import stats
class TestStats(unittest.TestCase):
 def make(self):
  rows=[]
  for task in stats.TASKS:
   for i in range(128):
    fs={n:float(i%7) for n in stats.FEATURES};rows.append({'task':task,'qid':str(i),'group_id':task+str(i),'features':fs,'scores':[.1,.2,.1+.8*(i%7)/6]})
  return rows
 def test_accepted_e1_metadata_without_split(self):
  r={'task':'DocVQA_VAL','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','image_path':'unused','message':[]};target={k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256')}
  self.assertTrue(stats.accepted_input_identity(r,target));self.assertFalse(stats.accepted_input_identity(dict(r,split='FINAL'),target));self.assertFalse(stats.accepted_input_identity(dict(r,group_id='wrong'),target))
 def test_ranks_ties(self):self.assertEqual(stats.rank([2,1,2,3]).tolist(),[2.5,1.,2.5,4.])
 def test_complete_ties_all_correlations(self):
  r=stats.summarize(self.make());self.assertEqual(len(r['correlations'])*len(stats.FEATURES),24)
  for c in r['correlations'].values():
   for v in c.values():self.assertEqual(v['pairs'],128);self.assertEqual(v['feature_tied_observations'],121);self.assertAlmostEqual(v['rho'],1.)
 def test_missing_duplicate_and_group(self):
  r=self.make()
  for bad in (r[:-1],r[:-1]+[r[0]],r[:1]+[dict(r[1],group_id=r[0]['group_id'])]+r[2:]):
   with self.assertRaises(ValueError):stats.summarize(bad)
 def test_undefined_and_constant(self):
  r=self.make()
  for x in r[:128]:x['features']['main_cv']=None;x['features']['ds0_cv']=1.
  s=stats.summarize(r)['correlations'][stats.TASKS[0]];self.assertEqual(s['main_cv']['reason'],'fewer_than_three_defined_pairs');self.assertEqual(s['main_cv']['feature_undefined'],128);self.assertEqual(s['ds0_cv']['reason'],'constant_feature')
 def test_constant_outcome_and_nonfinite(self):
  r=self.make()
  for x in r[:128]:x['scores']=[0.,.5,1.]
  self.assertEqual(stats.summarize(r)['correlations'][stats.TASKS[0]]['main_cv']['reason'],'constant_outcome')
  r[0]['scores']=[0.,.5,float('nan')]
  with self.assertRaises(ValueError):stats.summarize(r)
 def test_norms(self):
  n=np.tile(np.arange(1024,dtype=np.float64),(4,1));f=stats.norm_features(n);self.assertAlmostEqual(f['main_ds0_rank_corr'],1.);self.assertEqual(f['four_stream_rank_disagreement'],0.)
  z=np.zeros((4,1024));f=stats.norm_features(z);self.assertIsNone(f['main_cv']);self.assertIsNone(f['main_ds0_rank_corr'])
  for bad in (np.zeros((4,1023)),np.zeros((4,1024),dtype=np.float32)):
   with self.assertRaises(ValueError):stats.norm_features(bad)
 def test_cross_task_overlap(self):
  r=self.make();r[128]['group_id']=r[0]['group_id'];self.assertEqual(stats.summarize(r)['cross_task_group_overlap'][stats.TASKS[0]+'|'+stats.TASKS[1]],1)
if __name__=='__main__':unittest.main()
