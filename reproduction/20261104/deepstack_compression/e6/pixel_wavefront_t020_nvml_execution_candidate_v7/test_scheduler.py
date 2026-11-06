import subprocess,unittest
from unittest.mock import patch
import launch
class SchedulerTest(unittest.TestCase):
 def jobs(self):return [{'order':i,'arm':('t0_author','t020_author','t020_wavefront')[i%3]} for i in range(9)]
 def test_all_jobs_in_fixed_order(self):
  calls=[];events=[]
  def one(job):calls.append(job['order']);return {'status':'passed','job':job}
  self.assertEqual(launch.run_all(self.jobs(),one,events.append)['jobs'],9);self.assertEqual(calls,list(range(9)));self.assertEqual(len(events),18)
 def test_failure_stops_and_is_preserved(self):
  calls=[];events=[]
  def one(job):
   calls.append(job['order'])
   if job['order']==3:raise RuntimeError('simulated')
   return {'status':'passed','job':job}
  with self.assertRaises(RuntimeError):launch.run_all(self.jobs(),one,events.append)
  self.assertEqual(calls,[0,1,2,3]);self.assertEqual(events[-1]['event'],'failed')
 def test_wrong_order_fails_before_any_job(self):
  jobs=self.jobs();jobs[0]['order']=3
  with self.assertRaises(ValueError):launch.run_all(jobs,lambda _:self.fail('called'),lambda _:None)
 def test_timed_out_nested_supervisor_gets_term_and_cleanup_wait(self):
  class Fake:
   def __init__(self):self.calls=[];self.n=0
   def wait(self,timeout):
    self.calls.append(('wait',timeout));self.n+=1
    if self.n==1:raise subprocess.TimeoutExpired('fake',timeout)
    return 0
   def poll(self):return None
   def terminate(self):self.calls.append(('terminate',))
   def kill(self):self.calls.append(('kill',))
  fake=Fake()
  with patch.object(launch.subprocess,'Popen',return_value=fake):
   with self.assertRaises(subprocess.TimeoutExpired):launch.monitored(['x'],'/',10,30)
  self.assertIn(('terminate',),fake.calls);self.assertIn(('wait',30),fake.calls);self.assertNotIn(('kill',),fake.calls)
