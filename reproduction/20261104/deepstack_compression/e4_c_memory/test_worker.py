import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_latency'))
from schedule import build
from worker import run_memory_jobs

class MemoryWorkerTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'task':task,'qid':str(i),'group_id':task+'-'+str(i),'split':'DEV'} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)];self.s=build(self.rows,[],'smoke');self.events=[]
 def test_all_mode_jobs_preserved(self):
  def measure(row,job,partial):return {'mode':job['mode'],'protocol':job['protocol'],'memory_only':{'allocated_peak_bytes':1,'reserved_peak_bytes':2,'kv_cache_bytes_final':3,'kv_cache_bytes_peak_logical':3,'kv_observed_forwards':1}}
  result=run_memory_jobs(self.s,self.s,self.rows,'smoke','C-ON',measure,self.events.append)
  self.assertEqual(result['jobs'],24);self.assertEqual(len(self.events),48);self.assertTrue(all(x['job']['mode']=='C-ON' for x in self.events))
 def test_missing_memory_accounting_is_not_completed(self):
  def measure(row,job,partial):return {'mode':job['mode'],'protocol':job['protocol'],'memory_only':{}}
  with self.assertRaisesRegex(ValueError,'accounting mismatch'):run_memory_jobs(self.s,self.s,self.rows,'smoke','C-ON',measure,self.events.append)
  self.assertEqual([x['status'] for x in self.events],['generated','failed'])
 def test_bad_late_input_rejected_before_any_job(self):
  rows=copy.deepcopy(self.rows);target=next(x for x in reversed(self.s['schedule']) if x['mode']=='C-ON');row=next(r for r in rows if r['task']==target['task'] and r['qid']==target['qid']);row['group_id']='wrong'
  with self.assertRaisesRegex(ValueError,'identity mismatch'):run_memory_jobs(self.s,self.s,rows,'smoke','C-ON',lambda *_:None,self.events.append)
  self.assertEqual(self.events,[])
 def test_partial_generation_and_exception_remain_recorded(self):
  def fail(row,job,partial):partial({'output_ids':[1]});raise RuntimeError('post-generation failure')
  with self.assertRaisesRegex(RuntimeError,'post-generation'):run_memory_jobs(self.s,self.s,self.rows,'smoke','C-ON',fail,self.events.append)
  self.assertEqual([x['status'] for x in self.events],['generated_partial','failed'])
if __name__=='__main__':unittest.main()
