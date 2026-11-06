import tempfile,unittest
from pathlib import Path
from panel_core import ARMS,run

class CoreTests(unittest.TestCase):
 def setUp(self):
  self.rows=[dict(task=t,qid=str(i),group_id=t+str(i),image_sha256='i',message_sha256='m') for t,n in (('DocVQA_VAL',128),('TextVQA_VAL',128),('AI2D_TEST',128)) for i in range(n)]
  self.refs={(r['task'],r['qid']):{'processor':{'input_ids':'same'},'generation_scalars':{'max_new_tokens':2048}} for r in self.rows}
 def invoke(self,row,arm,journal,full):
  scalars={'max_new_tokens':2048};event={k:row[k] for k in ('task','qid','group_id')};event.update(arm=arm,output_ids=[2],generation_scalars=scalars,forced=False);journal(event)
  prune=[] if arm=='stock_native' else [{'layer':2,'range_after':[4,260 if arm=='priortr_k256' else 117]}]
  return dict(row,**{k:v for k,v in event.items() if k not in row},processor={'input_ids':'same'},eos_ids=[2],eos=True,truncated=False,backend='sdpa',precision='bfloat16',arrays={'kind':'full_arrays' if full else 'summary_only'},trace={'calls':[1],'layers':[1],'injections':[1,1,1],'prunes':prune})
 def test_complete_arm_major_and_fixed_array_policy(self):
  with tempfile.TemporaryDirectory() as d:
   result=run(rows=self.rows,e1_references=self.refs,out=Path(d)/'run',invoke=self.invoke)
   self.assertEqual(result['natural_generations'],1152)
   records=[__import__('json').loads(x) for x in (Path(d)/'run/records.jsonl').read_text().splitlines()]
   self.assertEqual([r['arm'] for r in records],[a for a in ARMS for _ in range(384)])
   self.assertEqual(sum(r['arrays']['kind']=='full_arrays' for r in records),9)
 def test_event_is_preserved_when_downstream_fails(self):
  def bad(row,arm,journal,full):
   r=self.invoke(row,arm,journal,full);r['processor']={'changed':True};return r
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run'
   with self.assertRaisesRegex(ValueError,'processor'):run(rows=self.rows,e1_references=self.refs,out=out,invoke=bad)
   self.assertEqual(len((out/'generated.jsonl').read_text().splitlines()),1);self.assertFalse((out/'worker_complete.json').exists())
 def test_wrong_k_or_missing_eos_fails(self):
  for change in ('k','eos'):
   def bad(row,arm,journal,full):
    r=self.invoke(row,arm,journal,full)
    if change=='k' and arm=='priortr_k256':r['trace']['prunes'][0]['range_after'][1]=255
    if change=='eos':r['eos']=False
    return r
   with tempfile.TemporaryDirectory() as d:
    with self.assertRaises(ValueError):run(rows=self.rows,e1_references=self.refs,out=Path(d)/'run',invoke=bad)
if __name__=='__main__':unittest.main()
