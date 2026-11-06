import json,tempfile,unittest
from pathlib import Path
from inventory import execute,SEEDS
class InventoryTest(unittest.TestCase):
 def fixture(self):
  rows=[{'task':t,'qid':str(i),'group_id':f'g{i}','image_sha256':f'i{i}','message_sha256':f'm{i}','split':'DEV'} for i,t in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  refs={(r['task'],r['qid']):[{'arm':'P3-D1','forced':False}] for r in rows}
  frozen={(r['task'],r['qid'],s):{**r,'spatial_ids':list(range(256))} for r in rows for s in SEEDS}
  return rows,refs,frozen,rows
 def run_fake(self,fail=False):
  rows,refs,frozen,anchors=self.fixture()
  def one(row,ref,seed,forced,generated):
   rec={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'seed':seed,'forced':forced,'output_ids':[100],'spatial_ids':list(range(256))}
   generated(rec)
   if fail:raise RuntimeError('failure after generation')
   return rec
  return rows,refs,frozen,anchors,one
 def test_complete_all_arms(self):
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(execute(*self.run_fake()[:4],d,self.run_fake()[4])['arms'],18)
   self.assertEqual(len(Path(d,'generated.jsonl').read_text().splitlines()),18)
 def test_failure_preserves_exposure_and_refuses_completion(self):
  rows,refs,frozen,anchors,one=self.run_fake(True)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):execute(rows,refs,frozen,anchors,d,one)
   self.assertEqual(len(Path(d,'generated.jsonl').read_text().splitlines()),1)
   self.assertFalse(Path(d,'worker_complete.json').exists())
   self.assertEqual(json.loads(Path(d,'journal.jsonl').read_text().splitlines()[-1])['event'],'failed')
 def test_bad_exposure_identity_is_preserved(self):
  rows,refs,frozen,anchors,_=self.run_fake()
  def one(row,ref,seed,forced,generated):
   generated({'task':'wrong','qid':row['qid'],'group_id':row['group_id'],'seed':seed,'forced':forced,'output_ids':[42]})
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):execute(rows,refs,frozen,anchors,d,one)
   self.assertEqual(json.loads(Path(d,'generated.jsonl').read_text())['raw']['output_ids'],[42])
   self.assertFalse(Path(d,'worker_complete.json').exists())
 def test_reject_wrong_frozen_and_duplicate_ref(self):
  rows,refs,frozen,anchors,one=self.run_fake();frozen[(rows[0]['task'],rows[0]['qid'],SEEDS[0])]['image_sha256']='wrong'
  with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):execute(rows,refs,frozen,anchors,d,one)
  rows,refs,frozen,anchors,one=self.run_fake();refs[(rows[0]['task'],rows[0]['qid'])]*=2
  with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):execute(rows,refs,frozen,anchors,d,one)
if __name__=='__main__':unittest.main()
