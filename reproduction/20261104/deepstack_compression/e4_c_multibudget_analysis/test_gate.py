import unittest
from unittest.mock import patch
import analysis_gate as g
class Tests(unittest.TestCase):
 def test_bad_independent_review_does_not_open_timing_files(self):
  formal={'status':'passed','jobs':1520,'warmups':240,'measured':1280,'GT_opened':False,'scoring_performed':False,'errors':[],'verifier_sha256':'x','verification_plan_sha256':'x'};independent={'status':'failed'};opened=[]
  def read(path):return formal if path.name=='verification001.json' else independent
  def sha(path):
   opened.append(str(path))
   if str(path).endswith(('journal.jsonl','health.jsonl')):raise AssertionError('timing data accessed before independent acceptance')
   return 'x'
  with patch.object(g,'prelaunch',return_value={}),patch.object(g,'terminal'),patch.object(g,'read',side_effect=read),patch.object(g,'sha',side_effect=sha):
   with self.assertRaisesRegex(ValueError,'independent timing review'):g.admit()
  self.assertFalse(any(x.endswith(('journal.jsonl','health.jsonl')) for x in opened))
 def test_missing_watcher_prevents_prediction_read(self):
  formal={'status':'passed','jobs':1520,'warmups':240,'measured':1280,'GT_opened':False,'scoring_performed':False,'errors':[],'verifier_sha256':'x','verification_plan_sha256':'x'}
  review={'status':'passed','independent_review':True,'jobs':1520,'warmups':240,'measured':1280,'GT_opened':False,'scoring_performed':False,'errors':[]}
  for k in ('formal_sha256','formal_started_sha256','formal_terminal_sha256','run_started_sha256','run_terminal_sha256','snapshot_manifest_sha256','reviewer_sha256'):review[k]='x'
  opened=[]
  def read(path):
   if path.name=='verification001.json':return formal
   if path.name=='e4c_multibudget_worker_v3_result_review.json':return review
   raise FileNotFoundError('watcher absent')
  def sha(path):
   opened.append(str(path))
   if str(path).endswith('journal.jsonl'):raise AssertionError('journal opened too early')
   return 'x'
  with patch.object(g,'prelaunch',return_value={}),patch.object(g,'terminal'),patch.object(g,'read',side_effect=read),patch.object(g,'sha',side_effect=sha):
   with self.assertRaises(FileNotFoundError):g.admit()
  self.assertFalse(any(x.endswith('journal.jsonl') for x in opened))
 def test_missing_prelaunch_refuses(self):
  with patch.object(g,'read',side_effect=FileNotFoundError):
   with self.assertRaises(FileNotFoundError):g.prelaunch()
if __name__=='__main__':unittest.main()
