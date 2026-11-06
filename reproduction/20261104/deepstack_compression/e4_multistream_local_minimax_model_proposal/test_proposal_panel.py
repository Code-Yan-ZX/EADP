import json,tempfile,unittest
from pathlib import Path
from test_cpu import dataset
import panel

class PanelTests(unittest.TestCase):
 def fixtures(self):
  sch,inv,rows,formal,independent=dataset()
  cohort=[];inputs=[];refs={}
  for i,(identity,row) in enumerate(zip(sch['input_identities'],rows)):
   identity=dict(identity);identity['processor']={'input_ids':{'shape':[1,1029]}}
   cohort.append(dict(identity,index=i,ids={k:v['ids'] for k,v in row['arms'].items()}));inputs.append(identity)
   refs[(identity['task'],identity['qid'])]=dict(identity,arm='P3-D1',forced=False)
  return cohort,inputs,refs
 def runner(self,fail=None):
  def one(row,arm,ids,ref,on_generated,on_raw):
   data={'task':row['task'],'qid':row['qid'],'group_id':row['group_id'],'arm':arm,'returned_batch':[[7,2]],
         'processor_input_length':1029,'generate_inputs_embeds_shape':[1,261,4],'generate_input_ids_shape':None}
   on_raw(data)
   if fail=='after_raw':raise RuntimeError('synthetic downstream failure')
   on_generated({'output_ids':[7,2]})
   result={**row,'output_ids':[7,2]}
   if fail=='corrupt':result['output_ids']=[7,3]
   return result
  return one
 def test_complete_all_64_and_fsync_raw(self):
  c,inputs,refs=self.fixtures()
  with tempfile.TemporaryDirectory() as t:
   result=panel.run(c,inputs,refs,t,lambda *a:None,self.runner(),lambda:None)
   self.assertEqual(result,64);d=Path(t);self.assertEqual(len(list((d/'observations').glob('*_raw.json'))),64)
   self.assertEqual(json.loads((d/'worker_complete.json').read_text())['generations'],64)
 def test_failure_after_return_preserves_raw_and_blocks_completion(self):
  c,inputs,refs=self.fixtures()
  for fail in ('after_raw','corrupt'):
   with self.subTest(fail=fail),tempfile.TemporaryDirectory() as t:
    with self.assertRaises((RuntimeError,ValueError)):panel.run(c,inputs,refs,t,lambda *a:None,self.runner(fail),lambda:None)
    d=Path(t);raw=list((d/'observations').glob('*_raw.json'));self.assertEqual(len(raw),1)
    self.assertEqual(json.loads(raw[0].read_text())['returned_batch'],[[7,2]])
    self.assertFalse((d/'worker_complete.json').exists());self.assertEqual(json.loads((d/'failure.json').read_text())['intended'],64)
 def test_identity_mismatch_before_any_exposure(self):
  c,inputs,refs=self.fixtures();inputs[0]['image_sha256']='changed'
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):panel.run(c,inputs,refs,t,lambda *a:None,self.runner(),lambda:None)
   self.assertFalse((Path(t)/'observations').exists())
if __name__=='__main__':unittest.main()
