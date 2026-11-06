import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import identity_gate as gate
class Tests(unittest.TestCase):
 def test_missing_independent_proof_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'FORMAL',Path(d)/'formal'),patch.object(gate,'REVIEW',Path(d)/'review'):
   (Path(d)/'formal').write_text('{}')
   with self.assertRaisesRegex(ValueError,'not independently accepted'):gate.admit_identity()
 def test_failed_proof_rejected_before_proof_references(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'FORMAL',Path(d)/'formal'),patch.object(gate,'REVIEW',Path(d)/'review'):
   (Path(d)/'formal').write_text(json.dumps({'status':'failed'}));(Path(d)/'review').write_text(json.dumps({'status':'passed'}))
   with self.assertRaisesRegex(ValueError,'identity proof mismatch'):gate.admit_identity()
