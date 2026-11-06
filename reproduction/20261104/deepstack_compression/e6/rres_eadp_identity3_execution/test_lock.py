import subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import controller,gpu_worker
class Tests(unittest.TestCase):
 def test_both_real_cli_entry_points_refuse_before_device_or_docker(self):
  def refuse():raise ValueError('admission refused')
  gate=SimpleNamespace(admit=refuse)
  with tempfile.TemporaryDirectory() as d,patch.object(subprocess,'Popen',side_effect=AssertionError('subprocess forbidden')):
   root=Path(d);out=root/'run001/rres/observations';out.mkdir(parents=True)
   with patch.object(gpu_worker,'HERE',root),patch.object(gpu_worker,'load',return_value=gate),patch.object(sys,'argv',['gpu_worker.py','--arm','rres','--out',str(out)]):
    with self.assertRaisesRegex(ValueError,'admission refused'):gpu_worker.main()
   with patch.object(controller,'load',return_value=gate),patch.object(sys,'argv',['controller.py','--run-id','001']):
    with self.assertRaisesRegex(ValueError,'admission refused'):controller.main()
