import json,tempfile,types,unittest
from pathlib import Path
import diagnostic_run,inventory
class DiagnosticTests(unittest.TestCase):
 def fixture(self):
  rows=[{'task':t,'qid':str(i),'group_id':f'g{i}','image_sha256':f'i{i}','message_sha256':f'm{i}','split':'DEV'} for i,t in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  refs={(r['task'],r['qid']):[{'arm':'P3-D1','forced':False}] for r in rows}
  frozen={(r['task'],r['qid'],s):{**r,'spatial_ids':list(range(256)),'derived_seed64':s} for r in rows for s in inventory.SEEDS}
  selection=lambda image,seed:{'spatial_ids':list(range(256)),'derived_seed64':seed}
  wrapper=types.SimpleNamespace(visual_token_num=256,pruner=types.SimpleNamespace(visual_token_num=256))
  return rows,refs,frozen,selection,wrapper
 def test_all_three_tasks_and_seeds(self):
  rows,refs,frozen,select,wrapper=self.fixture()
  def arm(torch,w,native,row,seed,ref,sel,runner,adapter,generate,raw,archive,identity,cache,forced):
   rec={k:row[k] for k in ('task','qid','group_id')};rec.update(seed=seed,forced=forced,spatial_ids=list(range(256)),derived_seed64=seed,output_ids=[7],actual_injection_verified=True);raw(rec);return rec
  with tempfile.TemporaryDirectory() as out:
   result=diagnostic_run.run(None,wrapper,None,rows,refs,frozen,rows,out,None,select,None,None,None,None,arm,inventory,lambda:None)
   self.assertEqual(result['arms'],18);self.assertEqual(len(Path(out,'records.jsonl').read_text().splitlines()),18)
 def test_mismatched_runtime_selection_stops_before_generation(self):
  rows,refs,frozen,select,wrapper=self.fixture();frozen[(rows[0]['task'],rows[0]['qid'],inventory.SEEDS[0])]['derived_seed64']=0
  with tempfile.TemporaryDirectory() as out:
   with self.assertRaisesRegex(ValueError,'selector differs'):diagnostic_run.run(None,wrapper,None,rows,refs,frozen,rows,out,None,select,None,None,None,None,lambda *a:None,inventory,lambda:None)
   self.assertFalse(Path(out,'worker_complete.json').exists());self.assertEqual(json.loads(Path(out,'journal.jsonl').read_text().splitlines()[-1])['event'],'failed')
if __name__=='__main__':unittest.main()
