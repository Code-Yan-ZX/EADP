import json,tempfile,types,unittest
from pathlib import Path
import core
class Tests(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(task=f't{i}',qid=str(i),group_id=f'g{i}',image_sha256='i',message_sha256='m',split='DEV') for i in range(3)]
  self.refs={'rows':self.rows,'records':{(r['task'],r['qid'],k):dict({x:'fixture' for x in core.PREFILL_FIELDS},**r,budget=k,ds_on=True,actual_injection_verified=True,decode_trace=[{'cache':'fixture'}]) for r in self.rows for k in core.BUDGETS}}
  self.wrapper=types.SimpleNamespace(pruner=types.SimpleNamespace());self.calls=[]
 def fresh(self,_torch,w,_native,row,k,_ds,_generate,raw,_archive,**kw):
  self.calls.append(('fresh',row['qid'],k));raw({'output_ids':[1]*32});return dict(self.refs['records'][(row['task'],row['qid'],k)])
 def light(self,_torch,w,_native,row,k,_generate,raw,_archive,**kw):
  self.calls.append(('light',row['qid'],k));raw({'output_ids':[1]*32});return dict(self.refs['records'][(row['task'],row['qid'],k)])
 def compare(self,fail=False):
  def check(a,b):
   if fail:raise ValueError('injected mismatch')
   if a!=b:raise ValueError('pair mismatch')
   return {'status':'passed'}
  return types.SimpleNamespace(compare_metadata=check,compare_archives=check)
 def invoke(self,path,comp=None):return core.run(None,self.wrapper,None,self.refs,path,None,self.fresh,self.light,comp or self.compare(),lambda p:'a'*64)
 def test_24_actual_callbacks_and_denominator(self):
  with tempfile.TemporaryDirectory() as d:
   result=self.invoke(d);rows=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
   self.assertEqual(result['generations'],24);self.assertEqual(len(self.calls),24)
   self.assertEqual(sum(r['event']=='raw_generation' for r in rows),24);self.assertEqual(sum(r['event']=='pair_verified' for r in rows),12)
   with self.assertRaisesRegex(ValueError,'reused'):self.invoke(d)
 def test_mismatch_preserves_both_actual_raw_records(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'injected mismatch'):self.invoke(d,self.compare(True))
   rows=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
   self.assertEqual([r['event'] for r in rows],['arm_started','raw_generation','arm_completed','arm_started','raw_generation','arm_completed','arm_failed'])
   self.assertFalse((Path(d)/'worker_complete.json').exists())
 def test_invalid_panel_prevents_any_generation(self):
  self.refs['records'][('t0','0',128)]['group_id']='bad'
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(ValueError,'recipe/input'):self.invoke(d)
  self.assertEqual(self.calls,[])
 def test_common_fake_prefill_cannot_match_panel(self):
  rec=dict(self.refs['records'][('t0','0',128)]);rec['ordered_ids']=['same wrong selection']
  with self.assertRaisesRegex(ValueError,'prefill mismatch'):core.compare_panel_prefill(rec,self.refs['records'][('t0','0',128)])
  rec=dict(self.refs['records'][('t0','0',128)]);rec['decode_trace']=[{'cache':'wrong'}]
  with self.assertRaisesRegex(ValueError,'cache/position'):core.compare_panel_prefill(rec,self.refs['records'][('t0','0',128)])
if __name__=='__main__':unittest.main()
