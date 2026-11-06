import importlib.util,unittest
from pathlib import Path
from independent_recalc import recalculate
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'m3_train_data/document_overlap_analysis_v2.py';s=importlib.util.spec_from_file_location('overlap_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Cases(unittest.TestCase):
 def test_mixed_overlap_and_excluded_counts(self):
  rows=[{'question_id':i,'document_id':d,'split':s} for i,(d,s) in enumerate([('a','fit'),('a','fit'),('a','validation'),('b','fit'),('c','excluded_eval'),('d','validation'),('e','excluded_eval')])]
  self.assertEqual(m.document_stats({'a','b','c','absent'},rows),recalculate({'a','b','c','absent'},rows))
 def test_single_split_eval_and_non_eval_fit_validation(self):
  rows=[{'question_id':0,'document_id':'a','split':'fit'},{'question_id':1,'document_id':'b','split':'fit'},{'question_id':2,'document_id':'b','split':'validation'}]
  r=m.document_stats({'a'},rows);self.assertEqual(r,recalculate({'a'},rows));self.assertEqual((r['source_documents_also_in_eval'],r['fit_validation_shared_documents']),(1,1))
 def test_missing_doc_unknown_split_rejected(self):
  for r in ({'document_id':'','split':'fit'},{'document_id':None,'split':'fit'},{'document_id':'a','split':'unknown'}):
   with self.assertRaises(ValueError):m.document_stats(set(),[r])
if __name__=='__main__':unittest.main(verbosity=2)
