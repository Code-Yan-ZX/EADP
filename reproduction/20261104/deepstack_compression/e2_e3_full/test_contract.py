import copy,unittest,sys
from pathlib import Path
from contract import bind_dev_inputs,inventory,validate_records,e3_views,record_digest,locked
class FullContractTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'task':t,'qid':str(i),'group_id':t+str(i),'split':'DEV'} for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
  self.records=[{'task':t,'qid':q,'group_id':g,'family':f,'arm':a,'forced':False} for t,q,g,f,a in inventory(self.rows)]
 def test_complete_and_new_endpoints(self):
  x=validate_records(self.rows,self.records);self.assertEqual(x['physical_generations'],3840)
  v=e3_views(self.rows,self.records);self.assertEqual(len(v),384)
  self.assertEqual(v[0]['arms']['OFF']['source_family'],'E2');self.assertEqual(v[0]['arms']['MATCH']['source_arm'],'111')
  self.assertEqual(v[0]['arms']['OFF']['record_sha256'],record_digest(self.records[0]))
  tampered=dict(self.records[0],qid='changed');self.assertNotEqual(record_digest(tampered),v[0]['arms']['OFF']['record_sha256'])
 def test_missing_duplicated_forced_or_nondevelopment_rejected(self):
  for bad in (self.records[:-1],self.records+[self.records[-1]],self.records[1:]+self.records[:1]):
   with self.assertRaises(ValueError):validate_records(self.rows,bad)
  bad=copy.deepcopy(self.records);bad[10]['forced']=True
  with self.assertRaises(ValueError):validate_records(self.rows,bad)
  bad=copy.deepcopy(self.rows);bad[0]['split']='FINAL'
  with self.assertRaises(ValueError):inventory(bad)
 def test_duplicate_group_rejected_and_execution_locked(self):
  bad=copy.deepcopy(self.rows);bad[1]['group_id']=bad[0]['group_id']
  with self.assertRaises(ValueError):inventory(bad)
  with self.assertRaisesRegex(ValueError,'locked'):locked()
 def test_join_e1_inputs_without_split_against_verified_dev_panel(self):
  inputs=[{k:v for k,v in r.items() if k!='split'} for r in self.rows]
  joined=bind_dev_inputs(inputs,self.rows);self.assertEqual(joined,self.rows)
  bad=copy.deepcopy(inputs);bad[0]['group_id']='wrong'
  with self.assertRaises(ValueError):bind_dev_inputs(bad,self.rows)
  bad=copy.deepcopy(inputs);bad[0]['split']='FINAL'
  with self.assertRaises(ValueError):bind_dev_inputs(bad,self.rows)
 def test_e3_alias_slot_values_identical_to_same_run_e2_endpoints(self):
  import torch
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e2_e3_contract'))
  from interventions import e2_slots,e3_streams
  inputs=[(torch.arange(256,dtype=torch.bfloat16).reshape(256,1)+i) for i in range(3)]
  inputs[1][0,0]=-0.0
  for bits,mode in (('000','OFF'),('111','MATCH')):
   e2=e2_slots(torch,inputs,bits);e3,_=e3_streams(torch,inputs,mode)
   self.assertEqual(len(e2),len(e3))
   for a,b in zip(e2,e3):self.assertTrue(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
if __name__=='__main__':unittest.main()
