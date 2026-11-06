import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
import core

P = Path(__file__).resolve().parent.parent/'random_full/inventory.py'
spec = importlib.util.spec_from_file_location('random_full_inventory_test',P)
inventory = importlib.util.module_from_spec(spec); spec.loader.exec_module(inventory)

class TestFullCore(unittest.TestCase):
 def setUp(self):
  self.rows = [dict(task='doc',qid=str(i),group_id=str(i),split='DEV',image_sha256='a',message_sha256='b') for i in range(384)]
  self.refs = {(r['task'],r['qid']):[{}] for r in self.rows}
  self.choices = {(r['task'],r['qid'],s):{**r,'seed':s,'spatial_ids':list(range(256)),'derived_seed64':s} for r in self.rows for s in inventory.SEEDS}
  self.wrapper = types.SimpleNamespace(visual_token_num=256,pruner=types.SimpleNamespace(visual_token_num=256),model=types.SimpleNamespace(dtype='bf16',config=types.SimpleNamespace(_attn_implementation='flash_attention_2')))
  self.events = []; self.old_emit = inventory.emit; inventory.emit = lambda path,obj:self.events.append((Path(path).name,obj))
  self.prepared = {'rows':self.rows,'refs':{(r['task'],r['qid']):{} for r in self.rows}}
 def tearDown(self): inventory.emit = self.old_emit
 def select(self,image,seed): return {'spatial_ids':list(range(256)),'derived_seed64':seed}
 def arm(self,torch,wrapper,native,row,seed,reference,selection,budget,adapter,generate,on_generated,archive,identity,cache,forced):
  raw={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'seed':seed,'forced':False,'output_ids':[1]}
  on_generated(raw)
  return {**raw,'derived_seed64':seed,'spatial_ids':list(range(256)),'actual_injection_verified':True,'arrays':archive({})}
 def invoke(self,path,selector=None):
  return core.run(self.prepared,self.choices,self.refs,path,types.SimpleNamespace(bfloat16='bf16'),None,self.wrapper,None,
                  selector or self.select,None,None,None,None,self.arm,inventory,lambda x:x,lambda *args:None,lambda:None)
 def test_complete_full_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   result=self.invoke(d)
   self.assertEqual(result['natural_generations'],1152)
   self.assertEqual(sum(name=='generated.jsonl' for name,_ in self.events),1152)
   self.assertEqual(sum(name=='records.jsonl' for name,_ in self.events),1152)
 def test_bad_selection_preserves_failure(self):
  def wrong(image,seed): return {'spatial_ids':list(range(1,257)),'derived_seed64':seed}
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):self.invoke(d,wrong)
   self.assertFalse((Path(d)/'worker_complete.json').exists())
   self.assertEqual(self.events[-1][1]['event'],'failed')
 def test_generation_then_failure_keeps_raw_and_no_completion(self):
  # Exercise raw callback then failure.
  def broken(torch,wrapper,native,row,seed,reference,selection,budget,adapter,generate,on_generated,archive,identity,cache,forced):
   on_generated({'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'seed':seed,'forced':False,'output_ids':[1]})
   raise RuntimeError('failure after generate')
  self.arm=broken
  inventory.emit=self.old_emit
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):self.invoke(d)
   import json
   self.assertEqual(len((Path(d)/'generated.jsonl').read_text().splitlines()),1)
   journal=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
   self.assertEqual([x['event'] for x in journal],['started','failed'])
   self.assertFalse((Path(d)/'worker_complete.json').exists())

if __name__=='__main__': unittest.main()
