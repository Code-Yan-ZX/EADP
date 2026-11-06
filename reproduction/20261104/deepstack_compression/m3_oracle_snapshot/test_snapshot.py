import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import snapshot
class SnapshotTests(unittest.TestCase):
 def test_partial_panel_cannot_snapshot(self):
  def read(path):
   name=str(path)
   if name.endswith('started.json'):return {'worker_pid':123}
   if name.endswith('terminal.json'):return {'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':123}
   if name.endswith('launch.json'):return {'controller':{'pid':123}}
   if name.endswith('exit.json'):return {'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,
       'container_stopped_verified':True,'client_exited_while_container_running':False,'errors':[]}
   if name.endswith('worker_complete.json'):return {'status':'completed','questions':127,'new_generations':762}
   raise AssertionError('unexpected read '+name)
  with tempfile.TemporaryDirectory() as d,patch.object(snapshot,'read',side_effect=read),patch.object(sys,'argv',['snapshot.py','--out',str(Path(d)/'snapshot')]),patch.object(snapshot.subprocess,'run') as docker:
   with self.assertRaisesRegex(ValueError,'not confirmed'):snapshot.main()
   docker.assert_not_called();self.assertFalse((Path(d)/'snapshot').exists())
if __name__=='__main__':unittest.main()
