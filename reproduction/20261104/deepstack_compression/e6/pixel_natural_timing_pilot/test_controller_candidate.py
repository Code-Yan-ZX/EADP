import json,subprocess,sys,unittest
from pathlib import Path
from contract import make_schedule
from controller_candidate import run_fake
class Controller(unittest.TestCase):
 def setUp(self):self.rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)];self.jobs=make_schedule(self.rows)
 def test_serial_distinct_processes(self):
  log=[];self.assertEqual(run_fake(self.jobs,self.rows,log.append),9)
  self.assertEqual([x['event'] for x in log],['started','completed']*9)
  ids=[(x['child_identity']['pid'],x['child_identity']['start_ticks']) for x in log if x['event']=='completed'];self.assertEqual(len(set(ids)),9)
 def test_invalid_schedule_refused_before_launch(self):
  self.jobs[0]['qid']='wrong';log=[]
  with self.assertRaises(ValueError):run_fake(self.jobs,self.rows,log.append)
  self.assertEqual(log,[])
 def test_failure_retained(self):
  log=[]
  with self.assertRaises(ValueError):run_fake(self.jobs,self.rows,log.append,child=[sys.executable,'-c','raise SystemExit(9)'])
  self.assertEqual([x['event'] for x in log],['started','failed'])
 def test_gpu_locked(self):
  r=subprocess.run([sys.executable,str(Path(__file__).with_name('controller_candidate.py')),'--real-gpu'],capture_output=True,text=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('locked',r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
