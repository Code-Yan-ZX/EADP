import json,os,tempfile,unittest
from pathlib import Path
from launch import clean_exit,validate_supervision,configure,BASE
from gate import sha
class LaunchTests(unittest.TestCase):
 def test_all_clean_exit_fields_required(self):
  r={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'failure':None,'container_stopped_verified':True,'client_exited_while_container_running':False,'gpu_pids_after':[],'container_identity':{'container_id':'x'}}
  self.assertTrue(clean_exit(r))
  for key,val in (('returncode',1),('container_exit_code',1),('timed_out',True),('interrupted',15),('failure','inspect'),('container_stopped_verified',False),('client_exited_while_container_running',True),('gpu_pids_after',None),('container_identity',None)):
   b=dict(r);b[key]=val;self.assertFalse(clean_exit(b))
 def test_supervisor_and_deadline(self):
  with tempfile.TemporaryDirectory() as d:
   run=Path(d)/'run';budget={'budget_source_sha256':sha(BASE/'initial_plan.json'),'outer_timeout_seconds':100,'journal_stale_seconds':20,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
   started={'worker_pid':os.getpid(),'timeout_seconds':100,'stale_seconds':20,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90,'progress_paths':[str(run/'data/generated.jsonl')]}
   validate_supervision(started,run,budget,9900,{'budget_end_unix':10000})
   with self.assertRaises(ValueError):validate_supervision(started,run,budget,9901,{'budget_end_unix':10000})
   started['progress_paths']=[]
   with self.assertRaises(ValueError):validate_supervision(started,run,budget,9800,{'budget_end_unix':10000})
 def test_container_environment_overrides_and_duplicate_rejection(self):
  cmd=['docker','run','--env','HF_HOME=/old','--entrypoint','old','--workdir','/work','image']
  out=configure(cmd,'/tmp/priortrtest');self.assertIn('HF_HOME=/tmp/priortrtest/hf',out);self.assertIn('USER=eadp',out)
  with self.assertRaises(ValueError):configure(cmd[:4]+['--env','HF_HOME=/other']+cmd[4:],'/tmp/priortrtest')
if __name__=='__main__':unittest.main()
