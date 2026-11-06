"""Independent pure-synthetic checks of proposed 192-image statistics."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_multibudget_analysis192_v2'))
import metrics
from test_metrics import synthetic
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.jobs,cls.events=synthetic()
  for i,j in enumerate(cls.jobs):
   if j['phase']!='measure':continue
   g=int(j['group_id'][1:]);rep=j['replicate'];k=j['budget'];t=1000
   first=t+(k*1000000)+(g%7)*1000000+rep*1000;last=first+(g%9+1)*1000000
   cls.events[3*i+1]['raw']['generation_complete_ns']=last
   cls.events[3*i+2]['record']['boundaries'].update(first_token_selected_ns=first,generation_complete_ns=last)
 def test_independent_variable_bootstrap(self):
  r=metrics.summarize(self.jobs,self.events);x=r['tasks']['DocVQA_VAL']['natural'];self.assertEqual(len(x['image_medians']),64)
  ids=sorted({j['group_id'] for j in self.jobs if j['phase']=='measure' and j['task']=='DocVQA_VAL'});a=np.array([128+int(g[1:])%7+.002 for g in ids]);b=np.array([256+int(g[1:])%7+.002 for g in ids]);draw=np.random.default_rng(20261105).integers(0,64,(10000,64));diff=a-b;ratio=a/b
  actual=x['128']['ttft_ms'];self.assertAlmostEqual(actual['time']['median'],float(np.median(a)));self.assertAlmostEqual(actual['paired_difference_vs_k256_ms']['median'],float(np.median(diff)));self.assertAlmostEqual(actual['paired_ratio_vs_k256']['median'],float(np.median(ratio)));self.assertTrue(np.allclose(actual['paired_ratio_vs_k256']['bootstrap_ci95'],np.quantile(np.median(ratio[draw],axis=1),[.025,.975]),rtol=0,atol=1e-12))
  self.assertEqual(x['128']['counts']['runs'],320);self.assertAlmostEqual(x['image_medians'][0]['budgets']['128']['ttft_ms'],a[0])
 def test_joint_identity_tamper_rejected(self):
  e=list(self.events);i=240
  for off,key in ((1,'raw'),(2,'record')):
   x=copy.deepcopy(e[3*i+off]);x[key]['qid']='wrong';e[3*i+off]=x
  with self.assertRaises(ValueError):metrics.summarize(self.jobs,e)
 def test_renumbered_repeat_rejected(self):
  jobs=copy.deepcopy(self.jobs);events=copy.deepcopy(self.events)
  for i in range(240,245):
   jobs[i]['replicate']+=10
   for off in range(3):events[3*i+off]['job']=jobs[i]
  with self.assertRaises(ValueError):metrics.summarize(jobs,events)
if __name__=='__main__':unittest.main()
