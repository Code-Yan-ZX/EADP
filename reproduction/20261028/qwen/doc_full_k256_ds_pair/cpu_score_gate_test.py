"""A missing root exit proof must fail before any scoring or GT access."""
import os,unittest
from unittest.mock import patch
import common as c
import score_terminal
class Tests(unittest.TestCase):
 def test_missing_root_proof_fails_without_gt_access(self):
  seen=[]
  def fake_read(path):
   seen.append(str(path))
   if path==c.PLAN:return {'run_id':c.RUN_ID}
   raise FileNotFoundError('no actual root exit proof')
  with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(score_terminal,'Path') as path_type,patch.object(c,'read',side_effect=fake_read),patch.object(c,'check'),patch.object(c,'sha',return_value='plan'):
   path_type.return_value.exists.return_value=False
   with self.assertRaises(FileNotFoundError):score_terminal.gate()
  self.assertTrue(any(x.endswith('root_terminal_process_proof.json') for x in seen))
  self.assertFalse(any('canonical' in x or 'score_report' in x for x in seen))
