import json,tempfile,unittest
from pathlib import Path
import independent_recalc as m
from independent_recalc import recalculate

class Cases(unittest.TestCase):
 def test_mixed_intersections_exclusion_and_denominators(self):
  rows=[{'question_id':i,'document_id':doc,'split':split} for i,(doc,split) in enumerate([('a','fit'),('a','fit'),('a','validation'),('b','fit'),('c','excluded_eval'),('d','validation'),('e','excluded_eval')])]
  result=recalculate({'a','b','c','absent'},rows)
  self.assertEqual(result,{'source_rows_all':7,'source_rows_by_split':{'fit':3,'validation':2,'excluded_eval':2},'eligible_rows_fit_plus_validation':5,'source_document_ids':5,'evaluation_docvqa_document_ids':4,'source_documents_also_in_eval':3,'source_rows_from_eval_documents_by_split':{'fit':3,'validation':1,'excluded_eval':1},'source_documents_also_in_eval_by_split':{'fit':2,'validation':1,'excluded_eval':1},'fit_validation_shared_documents':1,'rows_from_fit_validation_shared_documents_by_split':{'fit':2,'validation':1}})
 def test_disjoint_sets(self):
  self.assertEqual(recalculate({'eval'},[{'question_id':1,'document_id':'a','split':'fit'}])['source_documents_also_in_eval'],0)
 def test_invalid_data_rejected(self):
  for rows in ([{'question_id':0,'document_id':'','split':'fit'}],[{'question_id':0,'document_id':'a','split':'bad'}],[{'question_id':0,'document_id':'a','split':'fit'},{'question_id':0,'document_id':'b','split':'validation'}],[{'question_id':True,'document_id':'a','split':'fit'}],[{'question_id':0,'document_id':' a','split':'fit'}]):
   with self.subTest(rows=rows),self.assertRaises(ValueError):recalculate({'eval'},rows)
  with self.assertRaises(ValueError):recalculate({''},[])
 def test_supervisor_clean_exit_and_matching_command_required(self):
  with tempfile.TemporaryDirectory() as td:
   old=m.ROOT;m.ROOT=Path(td);d=m.ROOT/'monitor';d.mkdir();script=m.ROOT/'worker.py';script.write_text('pass\n')
   started={'cwd':td,'command':['python','worker.py','--out','report.json'],'worker_pid':4,'process_group':4};terminal={'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':4}
   def write():
    (d/'started.json').write_text(json.dumps(started));(d/'terminal.json').write_text(json.dumps(terminal))
   try:
    write();m.supervised(d,script,['--out','report.json'])
    for key,value in [('worker_returncode',1),('worker_pid',5),('reason','worker_failed'),('remaining_process_group',[4])]:
     original=terminal[key];terminal[key]=value;write()
     with self.assertRaises(ValueError):m.supervised(d,script,['--out','report.json'])
     terminal[key]=original
    started['command'][1]='elsewhere/worker.py';write()
    with self.assertRaises(ValueError):m.supervised(d,script,['--out','report.json'])
   finally:m.ROOT=old
if __name__=='__main__':unittest.main(verbosity=2)
