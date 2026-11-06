import unittest
import analysis
class Tests(unittest.TestCase):
 def make(self):
  jobs=[]
  for phase,reps in [('warmup',[0]),('measurement',[0,1,2])]:
   for t in range(3):
    for arm in ('author','wavefront'):
     for rep in reps:jobs.append({'order':len(jobs),'phase':phase,'task':str(t),'qid':str(t),'arm':arm,'repeat':rep})
  rows=[{'order':j['order'],'phase':j['phase'],'ttft_ns':100 if j['arm']=='author' else 50} for j in jobs];ann=[{'order':j['order'],'background':{'status':'unknown'}} for j in jobs];return jobs,rows,ann
 def test_paired_ratios_and_all_jobs(self):
  j,r,a=self.make();v=analysis.analyze(j,r,a);self.assertEqual(len(v['all_jobs']),24);self.assertEqual(v['measured'],18);self.assertEqual(v['per_anchor'][0]['wavefront_over_author_paired_ratios'],[.5,.5,.5])
 def test_missing_or_zero_rejected(self):
  j,r,a=self.make()
  with self.assertRaises(ValueError):analysis.analyze(j,r[:-1],a)
  r[8]['ttft_ns']=0
  with self.assertRaises(ValueError):analysis.analyze(j,r,a)
if __name__=='__main__':unittest.main()
