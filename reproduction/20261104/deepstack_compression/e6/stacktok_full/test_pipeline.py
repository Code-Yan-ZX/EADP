import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from worker import execute_rows
class Tests(unittest.TestCase):
 def setup_inputs(self):
  rows=[{'task':'fake','qid':str(i),'group_id':'group'+str(i)} for i in range(384)];refs={(x['task'],x['qid']):{} for x in rows};return rows,refs
 def test_384_records_and_durable_failure(self):
  rows,refs=self.setup_inputs()
  def one(row,ref,returned,generated):
   result=dict(row,output_ids=[1]);returned(dict(row,full_ids=[1],input_length=0));generated(result);return result
  with tempfile.TemporaryDirectory() as tmp,patch('worker.validate_reference',return_value=True):
   path=Path(tmp);self.assertEqual(execute_rows(rows,refs,path,one),384)
   self.assertEqual(len((path/'generated.jsonl').read_text().splitlines()),384);self.assertEqual(len((path/'records.jsonl').read_text().splitlines()),384)
  def fail(row,ref,returned,generated):
   result=dict(row,output_ids=[1]);returned(dict(row,full_ids=[1],input_length=0));generated(result)
   if row['qid']=='7':raise ValueError('post-generation failure')
   return result
  with tempfile.TemporaryDirectory() as tmp,patch('worker.validate_reference',return_value=True):
   path=Path(tmp)
   with self.assertRaises(ValueError):execute_rows(rows,refs,path,fail)
   self.assertEqual(len((path/'returned.jsonl').read_text().splitlines()),8);self.assertEqual(len((path/'generated.jsonl').read_text().splitlines()),8);self.assertEqual(len((path/'records.jsonl').read_text().splitlines()),7)
   self.assertEqual(json.loads((path/'journal.jsonl').read_text().splitlines()[-1])['event'],'failed')
 def test_bad_prefix_still_returns_complete_ids(self):
  import torch
  from full_arm import returned_ids
  captured=[]
  with self.assertRaisesRegex(ValueError,'prompt changed'):
   returned_ids(torch.tensor([[8,9,10]]),torch.tensor([[1,2]]),captured.append,{'task':'x'})
  self.assertEqual(captured,[{'task':'x','full_ids':[8,9,10],'input_length':2}])
 def test_missing_and_duplicate_inputs_rejected(self):
  rows,refs=self.setup_inputs()
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaises(ValueError):execute_rows(rows[:-1],refs,Path(tmp),lambda *_:None)
   rows[-1]=rows[0]
   with self.assertRaises(ValueError):execute_rows(rows,refs,Path(tmp),lambda *_:None)
if __name__=='__main__':unittest.main(verbosity=2)
