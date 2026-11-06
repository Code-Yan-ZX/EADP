import unittest
from worker import anchors,upstream_selector
class Tests(unittest.TestCase):
 def test_stock_capture_exact_reviewed_source_except_unused_import(self):
  from worker import BASE,HERE
  old=(BASE/'e4_native/native_capture.py').read_text().replace('from native_contract import E4\n','')
  self.assertEqual(old,(HERE/'stock_capture.py').read_text())
 def test_k256_requires_passing_prior_keepall(self):
  from worker import ensure_path_ready
  from unittest.mock import patch
  with self.assertRaises(ValueError):ensure_path_ready('local_stacktok',[])
  with patch('arm.compare_keepall',side_effect=ValueError('mismatch')) as check:
   with self.assertRaises(ValueError):ensure_path_ready('local_stacktok',[{},{}])
   self.assertEqual(check.call_count,1)
  with patch('arm.compare_keepall',return_value=True) as check:
   ensure_path_ready('local_stacktok',[{},{}]);self.assertEqual(check.call_count,1)
 def test_full_actual_import_chain(self):
  import arm,verify,stock_capture
  from worker import HERE
  from pathlib import Path
  self.assertEqual(Path(stock_capture.__file__).resolve(),HERE/'stock_capture.py')
  self.assertEqual(Path(arm.__file__).resolve(),HERE/'arm.py')
  self.assertEqual(Path(verify.__file__).resolve(),HERE/'verify.py')
 def test_selector_is_exact_original_core(self):
  c=upstream_selector();self.assertEqual(c(target_vision_tokens=256).target_vision_tokens,256)
  with self.assertRaises(ValueError):upstream_selector()
 def test_anchors_reject_missing_or_duplicate(self):
  import json
  from worker import HERE
  rows=json.loads((HERE/'plan001.json').read_text())['anchors'];self.assertEqual(len(anchors({'rows':rows})),3)
  with self.assertRaises(ValueError):anchors({'rows':rows[:-1]})
  with self.assertRaises(ValueError):anchors({'rows':rows+[rows[0]]})
  rows[0]['image_sha256']='0'*64
  with self.assertRaises(ValueError):anchors({'rows':rows})
if __name__=='__main__':unittest.main(verbosity=2)
