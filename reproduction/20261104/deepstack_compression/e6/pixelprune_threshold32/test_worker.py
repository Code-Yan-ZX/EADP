import unittest
from unittest import mock
import worker
class TestWorker(unittest.TestCase):
 def test_bad_plan_rejected_before_archive_read(self):
  with mock.patch.object(worker,'sha',side_effect=AssertionError('must not hash')):
   with self.assertRaises(ValueError):worker.validate_plan({'status':'bad'})
 def test_duplicate_groups_rejected(self):
  samples=[{'group_id':'same'} for _ in range(32)]
  with self.assertRaises(ValueError):worker.validate_plan({'status':'frozen','GT_opened':False,'predictions_opened':False,'gpu_authorized':False,'samples':samples,'thresholds':[0.,.4,.45,.5,.55,.6,.65,.7,.75,.8]})
if __name__=='__main__':unittest.main()
