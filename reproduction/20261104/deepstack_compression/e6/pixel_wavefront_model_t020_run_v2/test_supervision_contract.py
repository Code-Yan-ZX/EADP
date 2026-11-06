import copy,unittest
from pathlib import Path
from supervision_contract import validate
class Tests(unittest.TestCase):
 def fixture(self):
  project=Path('/p');run=project/'e6/run';job=['python',str(run/'controller.py'),'--run-id','001'];p={'outer_timeout_seconds':6480,'grace_seconds':90,'stale_seconds':1200,'budget_start_unix':0,'budget_end_unix':20000,'acceptance_reserve_seconds':3600}
  s={'worker_pid':5,'process_group':5,'command':job,'cwd':'/p','timeout_seconds':6480,'grace_seconds':90,'stale_seconds':1200,'poll_seconds':5,'snapshot_seconds':3600,'progress_paths':[str(run/'run_001/journal.jsonl')]}
  parent=['python',str(project/'monitor/supervise.py'),'--run-dir',str(run/'monitor_run001'),'--timeout','6480','--grace-seconds','90','--stale','1200','--progress',str(run/'run_001/journal.jsonl'),'--poll-seconds','5','--snapshot-seconds','3600','--']+job
  return s,p,parent,job,project,run
 def test_positive_and_parent_tampering(self):
  s,p,parent,job,project,run=self.fixture();self.assertTrue(validate(s,p,5,5,5,'/p',parent,job,10,project,run))
  for bad in (parent[:-1],parent[:2]+['--run-dir','/wrong']+parent[4:]):
   with self.assertRaises(ValueError):validate(s,p,5,5,5,'/p',bad,job,10,project,run)
 def test_session_and_reserve(self):
  s,p,parent,job,project,run=self.fixture()
  with self.assertRaises(ValueError):validate(s,p,5,5,7,'/p',parent,job,10,project,run)
  with self.assertRaises(ValueError):validate(s,p,5,5,5,'/p',parent,job,15000,project,run)
if __name__=='__main__':unittest.main(verbosity=2)
