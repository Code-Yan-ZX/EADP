import sys,subprocess,unittest
from pathlib import Path
import controller
class Tests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)]
  self.jobs=controller.contract.make_schedule(self.rows)
 def ok(self,j):
  return {'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'failure_type':None,'client_exited_while_container_running':False,'gpu_pids_after':[],'container_identity':{'host_init':{'pid':j['order']+100,'start_ticks':j['order']+1,'boot_id':'boot'}}}
 def test_all_nine_serial(self):
  seen=[];log=[]
  def run(j):seen.append(j);return self.ok(j)
  self.assertEqual(controller.run_sequential(self.jobs,self.rows,lambda:[],run,log.append),9)
  self.assertEqual(seen,self.jobs);self.assertEqual([x['event'] for x in log],['launching','finished']*9)
 def test_exclusivity_checked_before_first_launch(self):
  log=[];started=[]
  with self.assertRaises(ValueError):controller.run_sequential(self.jobs,self.rows,lambda:[123],lambda j:started.append(j),log.append)
  self.assertEqual(started,[]);self.assertEqual(log,[])
 def test_midrun_failure_preserved_and_stops(self):
  log=[];seen=[]
  def run(j):
   seen.append(j);x=self.ok(j)
   if j['order']==2:x['container_stopped_verified']=False
   return x
  with self.assertRaises(ValueError):controller.run_sequential(self.jobs,self.rows,lambda:[],run,log.append)
  self.assertEqual(len(seen),3);self.assertEqual(log[-1]['event'],'failed')
 def test_reused_pid_identity_is_rejected(self):
  log=[]
  def run(j):
   x=self.ok(j);x['container_identity']['host_init']={'pid':1,'start_ticks':1,'boot_id':'boot'};return x
  with self.assertRaises(ValueError):controller.run_sequential(self.jobs,self.rows,lambda:[],run,log.append)
  self.assertEqual(log[-1]['event'],'failed')
 def test_gpu_main_locked(self):
  x=subprocess.run([sys.executable,str(Path(controller.__file__))],capture_output=True,text=True)
  self.assertNotEqual(x.returncode,0);self.assertIn('GPU controller locked',x.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
