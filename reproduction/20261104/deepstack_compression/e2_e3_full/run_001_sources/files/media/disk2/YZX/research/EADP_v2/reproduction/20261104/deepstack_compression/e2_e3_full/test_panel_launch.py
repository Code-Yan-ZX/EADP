import json,os,tempfile,unittest
from pathlib import Path
from panel_launch import validate_supervision,clean_exit
from panel_gate import sha
class LauncherTests(unittest.TestCase):
 def test_success_requires_every_exit_condition(self):
  r={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'gpu_pids_after':[],'container_identity':{'container_id':'x'},'failure':None}
  self.assertTrue(clean_exit(r))
  for key,value in [('returncode',1),('container_exit_code',1),('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('container_stopped_verified',False),('gpu_pids_after',None),('container_identity',None),('failure','inspect error')]:
   b=dict(r);b[key]=value;self.assertFalse(clean_exit(b))
 def test_deadline_and_exact_supervisor_guard(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);initial=d/'initial';initial.write_text(json.dumps({'budget_end_unix':10000}));run=d/'run'
   budget={'budget_source_sha256':sha(initial),'budget_end_unix':10000,'outer_timeout_seconds':100,'journal_stale_seconds':10,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}
   st={'worker_pid':os.getpid(),'timeout_seconds':100,'stale_seconds':10,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90,'progress_paths':[str(run/'data/generated.jsonl')]}
   validate_supervision(st,run,budget,9900,initial)
   with self.assertRaises(ValueError):validate_supervision(st,run,budget,9901,initial)
   for k,v in [('worker_pid',-1),('timeout_seconds',99),('stale_seconds',None),('progress_paths',[])]:
    b=dict(st);b[k]=v
    with self.assertRaises(ValueError):validate_supervision(b,run,budget,9800,initial)
if __name__=='__main__':unittest.main()
