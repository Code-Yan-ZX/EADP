import json,tempfile,unittest
from pathlib import Path
import controller as c
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.j=Path(self.t.name)/'journal.jsonl'
  self.images=[{'anchor':{'task':t,'qid':str(n),'group_id':f'g{n}','image_sha256':f'i{n}','message_sha256':f'm{n}'}} for n,t in enumerate(('D','T','A'))]
 def good(self,job):return {'container_id':f"{job['order']+1:064x}",'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'result_status':'passed'}
 def events(self):return [json.loads(x) for x in self.j.read_text().splitlines()]
 def test_all_six_and_existing_journal_rejected(self):
  self.assertEqual(len(c.execute_with_injected_runner(self.images,self.j,self.good)),12);self.assertEqual(self.events()[-1]['event'],'all_arms_completed')
  with self.assertRaises(ValueError):c.execute_with_injected_runner(self.images,self.j,self.good)
 def test_runner_failure_persists_started_and_stops(self):
  calls=[]
  def runner(j):
   calls.append(j['order'])
   if j['order']==2:raise RuntimeError('synthetic failure')
   return self.good(j)
  with self.assertRaises(RuntimeError):c.execute_with_injected_runner(self.images,self.j,runner)
  self.assertEqual(calls,[0,1,2]);self.assertEqual([x['event'] for x in self.events()][-2:],['arm_invoked','arm_failed'])
 def test_exit_failure_and_container_reuse_stop(self):
  for failure in ('exit','reused'):
   j=self.j.with_name(failure+'.jsonl');calls=[]
   def runner(job):
    calls.append(job['order']);r=self.good(job)
    if job['order']==1:
     if failure=='exit':r['returncode']=1
     else:r['container_id']=f'{1:064x}'
    return r
   with self.assertRaises(ValueError):c.execute_with_injected_runner(self.images,j,runner)
   self.assertEqual(calls,[0,1]);self.assertEqual(json.loads(j.read_text().splitlines()[-1])['event'],'arm_failed')
if __name__=='__main__':unittest.main(verbosity=2)
