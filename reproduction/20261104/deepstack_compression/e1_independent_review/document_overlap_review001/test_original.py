"""Synthetic-only counterexamples for the archived overlap diagnostic."""
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('archived_overlap',Path(__file__).parent/'original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Cases(unittest.TestCase):
 def test_eval_only_one_fit_split_is_missing_from_named_union(self):
  rows=[{'document_id':'evalA','split':'fit'}]
  result=m.document_stats({'evalA'},rows)
  self.assertEqual(result['train_rows_from_eval_documents_by_split'],{'fit':1})
  self.assertEqual(result['fit_validation_or_eval_connected_documents'],0)
 def test_fit_and_validation_without_eval_still_counts_named_union(self):
  rows=[{'document_id':'unrelated','split':'fit'},{'document_id':'unrelated','split':'validation'}]
  result=m.document_stats({'evalA'},rows)
  self.assertEqual(result['train_documents_shared_with_eval'],0)
  self.assertEqual(result['fit_validation_or_eval_connected_documents'],1)
 def test_excluded_rows_included_in_train_total(self):
  rows=[{'document_id':'a','split':'fit'},{'document_id':'b','split':'excluded_eval'}]
  result=m.document_stats(set(),rows)
  self.assertEqual(result['train_rows'],2)
  self.assertEqual(result['rows_by_split'],{'fit':1,'excluded_eval':1})
 def test_invalid_missing_ids_or_splits_are_not_rejected(self):
  rows=[{'document_id':'','split':'fit'},{'document_id':'','split':'validation'},{'document_id':'x','split':'bogus'}]
  result=m.document_stats(set(),rows)
  self.assertEqual(result['fit_validation_shared_documents'],1)
  self.assertEqual(result['rows_by_split']['bogus'],1)
if __name__=='__main__':unittest.main(verbosity=2)
