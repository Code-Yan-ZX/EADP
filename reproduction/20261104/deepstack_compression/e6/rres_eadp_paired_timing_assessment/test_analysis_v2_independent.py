"""Independent scalar bootstrap recomputation on synthetic, not project data."""
import importlib.util,statistics,sys,unittest
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];D=B/'e6/rres_eadp_timing_analysis_proposal_v2';sys.path.insert(0,str(D))
import metrics
spec=importlib.util.spec_from_file_location('_paired_stat_fixtures',D/'test_metrics.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
def percentile_linear(xs,p):
 xs=sorted(xs);v=(len(xs)-1)*p;lo=int(v);f=v-lo
 return xs[lo]*(1-f)+xs[min(lo+1,len(xs)-1)]*f
class TestIndependentStats(unittest.TestCase):
 def test_scalar_recompute_all_tasks_and_endpoints(self):
  schedule,records=fixture.fixtures()
  # Distinct arm, image and repeat timings defeat aggregate-ratio and repetition-pairing mistakes.
  for process,recs in zip(schedule['processes'],records):
   for r in recs:
    q=int(r['request']['qid'][1:]);rep=r['request']['replicate'];arm=process['arm'];t=r['times'];base=(q+1)*1000+rep*23+(67 if arm=='rres' else 0)
    t['first_ns']=100+base;t['generate_end_ns']=100+2*base+(17 if arm=='rres' else 0);t['response_end_ns']=100+3*base+q
  result=metrics.summarize(schedule,records);rng=np.random.default_rng(20261106)
  for task in metrics.TASKS:
   images=[x for x in schedule['measured_inputs'] if x['task']==task];draws=rng.integers(0,len(images),size=(10000,len(images)))
   for key,endpoint in metrics.ENDPOINTS.items():
    diff=[];ratio=[]
    for image in images:
     med={}
     for arm in metrics.ARMS:
      values=[(r['times'][endpoint]-r['times']['t0_ns'])/1e6 for proc,recs in zip(schedule['processes'],records) if proc['task']==task and proc['arm']==arm for r in recs if r['request']['phase']=='measured' and r['request']['qid']==image['qid']]
      self.assertEqual(len(values),5);med[arm]=statistics.median(values)
     diff.append(med['rres']-med['eadp']);ratio.append(med['rres']/med['eadp'])
    db=[statistics.median(diff[int(j)] for j in draw) for draw in draws];rb=[statistics.median(ratio[int(j)] for j in draw) for draw in draws];got=result['tasks'][task]['metrics'][key]
    self.assertAlmostEqual(got['paired_difference_median_ms'],statistics.median(diff));self.assertAlmostEqual(got['paired_ratio_median'],statistics.median(ratio))
    for i,p in enumerate((.025,.975)):
     self.assertAlmostEqual(got['difference_ci95_ms'][i],percentile_linear(db,p));self.assertAlmostEqual(got['ratio_ci95'][i],percentile_linear(rb,p))
if __name__=='__main__':unittest.main()
