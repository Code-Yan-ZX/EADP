import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gpu_gate as g
class GateTests(unittest.TestCase):
 def test_missing_plan_rejects_before_other_evidence(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(g,'D',Path(temp)),patch.object(g,'load',side_effect=AssertionError('must not load')):
   with self.assertRaises(FileNotFoundError):g.admit()
 def test_plan_refuses_bad_inventory(self):
  with tempfile.TemporaryDirectory() as temp,patch.object(g,'D',Path(temp)):
   p=Path(temp)/'gpu_plan001.json';p.write_text(json.dumps({'stage':'192-image four-budget timing','jobs':7920,'warmups':240,'measured':7680,'GT_opened':False,'GPU_authorized_by_plan':False,'source_sha256':{'/missing':'x'}}))
   with self.assertRaises((ValueError,FileNotFoundError)):g.plan()
 def test_incomplete_supervisor_cannot_authorize(self):
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'started.json').write_text(json.dumps({'worker_pid':1,'cwd':temp,'command':['python','review.py','--out','out.json']}));(d/'terminal.json').write_text(json.dumps({'worker_pid':1,'reason':'completed','worker_returncode':1,'remaining_process_group':[]}))
   with self.assertRaises(ValueError):g.terminal(d,d/'review.py',d/'out.json')
 def test_observer_lifetime_insufficient_expired_or_invalid(self):
  ns=10**9;g.observer_lifetime(10*ns,20*ns,40000,28800,1200)
  for start,now,duration in ((10*ns,20*ns,30000),(10*ns,40020*ns,40000),(21*ns,20*ns,40000)):
   with self.assertRaises(ValueError):g.observer_lifetime(start,now,duration,28800,1200)
 def test_observer_rejects_fake_pid_zombie_or_changed_command(self):
  s={'worker_pid':7,'process_group':7,'cwd':'/tmp','command':['python','watch.py']}
  g.observer_process(s,['python','watch.py'],'S','/tmp')
  for override,actual,state in (({'worker_pid':0},s['command'],'S'),({},s['command'],'Z'),({},['python','foreign.py'],'S'),({'process_group':8},s['command'],'S')):
   with self.assertRaises(ValueError):g.observer_process(dict(s,**override),actual,state,'/tmp')
 def test_foreign_preloaded_module_rejected(self):
  import sys
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'mod.py';p.write_text('raise AssertionError("must not run")')
   with patch.dict(sys.modules,{'_foreign_gpu_gate':object()}):
    with self.assertRaises(ValueError):g.load('_foreign_gpu_gate',p,g.sha(p))
if __name__=='__main__':unittest.main()
