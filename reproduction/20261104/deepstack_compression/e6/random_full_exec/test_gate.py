import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate

class TestFailClosed(unittest.TestCase):
 def test_missing_preliminary_review_rejected(self):
  with patch.object(gate,'plan',return_value={}),patch.object(gate,'prior',return_value={}),patch.object(gate,'review',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):gate.admit(True)
 def test_gpu_rejected_without_completed_preflight(self):
  with tempfile.TemporaryDirectory() as d, patch.object(gate,'D',Path(d)),patch.object(gate,'plan',return_value={}),patch.object(gate,'prior',return_value={}),patch.object(gate,'review',return_value='approved'),patch.object(gate,'sha',return_value='sha'):
   with self.assertRaises(FileNotFoundError):gate.admit(False)

if __name__=='__main__':unittest.main()
