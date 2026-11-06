import json,tempfile,unittest
from pathlib import Path
import gate

class Tests(unittest.TestCase):
 def test_gpu_lock_cannot_be_bypassed_by_file(self):
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):gate.admit_gpu()
 def test_missing_and_changed_runtime_source_rejected(self):
  old=gate.HERE
  with tempfile.TemporaryDirectory() as directory:
   gate.HERE=Path(directory)
   try:
    with self.assertRaises(FileNotFoundError):gate.validate_runtime_sources()
    p=gate.HERE/'source.py';p.write_text('before');digest=gate.sha(p)
    (gate.HERE/'runtime_plan001.json').write_text(json.dumps({'source_sha256':{str(p):digest}}))
    gate.validate_runtime_sources();p.write_text('after')
    with self.assertRaisesRegex(ValueError,'source changed'):gate.validate_runtime_sources()
   finally:gate.HERE=old
if __name__=='__main__':unittest.main()
