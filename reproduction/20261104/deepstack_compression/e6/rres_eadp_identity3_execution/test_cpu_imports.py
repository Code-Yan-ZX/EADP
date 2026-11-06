import hashlib,unittest
from pathlib import Path
import controller,cpu_entry,cpu_gate,cpu_launch,gpu_gate,gpu_worker,snapshot,verify
HERE=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_actual_cpu_and_gpu_entry_imports_never_authorize_gpu(self):
  digest=hashlib.sha256((HERE/'gpu_gate.py').read_bytes()).hexdigest()
  self.assertEqual(gpu_worker.GATE_SHA,digest);self.assertEqual(controller.GATE_SHA,digest)
  with self.assertRaises(RuntimeError):cpu_gate.gpu()
  self.assertTrue(callable(cpu_entry.main) and callable(cpu_launch.main) and callable(snapshot.main) and callable(verify.main))
