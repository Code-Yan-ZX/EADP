"""Exercise target's actual separate executor/artifact path contract synthetically."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import admission
class EntryPaths(unittest.TestCase):
 def test_target_dispatches_real_executor_paths(self):
  calls=[]
  def terminal(mon,script,out=None):
   calls.append((mon,script,out))
   if len(calls)==3:raise RuntimeError('sentinel before any target reads')
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);run=root/'e6/rres_k256_dev384';exe=root/'e6/rres_k256_dev384_exec'
   with patch.object(admission,'RUN',run),patch.object(admission,'EXEC',exe),patch.object(admission,'terminal',side_effect=terminal):
    with self.assertRaisesRegex(RuntimeError,'sentinel'):admission.target()
   self.assertEqual(calls,[(run/'monitor_run001',exe/'launch.py',None),(run/'monitor_snapshot001',exe/'snapshot.py',None),(run/'monitor_verify001',exe/'verify.py',run/'verification001.json')])
 def test_supervisor_rejects_same_name_wrong_directory_and_wrong_output(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);mon=root/'artifacts/monitor_verify001';mon.mkdir(parents=True);script=root/'executor/verify.py';script.parent.mkdir();script.write_text('')
   output=root/'artifacts/verification001.json';start={'cwd':str(root),'worker_pid':321,'command':['python',str(script),'--out',str(output)]};end={'worker_pid':321,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
   def store():
    (mon/'started.json').write_text(json.dumps(start));(mon/'terminal.json').write_text(json.dumps(end))
   store();admission.terminal(mon,script,output)
   start['command'][1]=str(root/'artifacts/verify.py');store()
   with self.assertRaises(ValueError):admission.terminal(mon,script,output)
   start['command'][1]=str(script);start['command'][-1]=str(root/'wrong.json');store()
   with self.assertRaises(ValueError):admission.terminal(mon,script,output)
if __name__=='__main__':unittest.main()
