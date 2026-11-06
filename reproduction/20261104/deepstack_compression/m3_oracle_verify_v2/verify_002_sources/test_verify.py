import json,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
import verify

def ident(pid):
 raw=Path('/proc',str(pid),'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 return {'pid':pid,'start_ticks':int(fields[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
def sample(identity):
 root=verify.ROOT;here=verify.HERE;run=here/'runtime_test';p={'outer_timeout_seconds':10,'progress_stale_seconds':20,'snapshot_seconds':3600}
 s={'worker_pid':identity['pid'],'process_group':identity['pid'],'timeout_seconds':10,'stale_seconds':20,'snapshot_seconds':3600,
    'progress_paths':[str((run/'panel/panel_events.jsonl').resolve())],
    'command':['python',str(here/'runtime_launch.py'),'--run-id','test'],'cwd':str(root)}
 t={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':identity['pid']}
 launch={'controller':identity,'command':['docker',str(here/'runtime_worker.py'),'--execute','--out',str(run)],
         'worker_sha256':verify.runtime_gate.sha(here/'runtime_worker.py'),'launcher_sha256':verify.runtime_gate.sha(here/'runtime_launch.py')}
 ex={'controller':identity,'container_identity':{'container_id':'test','host_init':identity},'returncode':0,'container_exit_code':0,
     'timed_out':False,'interrupted':None,'container_stopped_verified':True,'client_exited_while_container_running':False,'errors':[],'gpu_pids_after':[]}
 return run,s,t,launch,ex,p

class VerifyTests(unittest.TestCase):
 def test_actual_completion_alive_rejects_running_controller(self):
  args=sample(ident(os.getpid()))
  with patch.object(verify.subprocess,'run') as docker:
   with self.assertRaisesRegex(ValueError,'still alive'):verify.check_terminal(*args)
   docker.assert_not_called()
 def test_actual_completion_alive_accepts_reaped_process(self):
  proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);identity=ident(proc.pid)
  proc.terminate();proc.wait(timeout=5);args=sample(identity)
  with patch.object(verify.subprocess,'run',return_value=type('Result',(),{'stdout':json.dumps([{}])})()),patch.object(verify.container_identity,'validate',return_value=None):
   verify.check_terminal(*args)
 def test_actual_frozen_run_panel_does_not_have_assumed_function(self):
  # This was the precise defect: prove the real author module exists and that
  # the replacement comes from actual frozen completion.py, not a mock.
  sys.path.insert(0,str(verify.PLAN.parent))
  try:
   import run_panel
   self.assertEqual(Path(run_panel.__file__).resolve(),(verify.PLAN.parent/'run_panel.py').resolve())
   self.assertFalse(hasattr(run_panel,'alive'))
  finally:sys.path.pop(0)
if __name__=='__main__':unittest.main()
