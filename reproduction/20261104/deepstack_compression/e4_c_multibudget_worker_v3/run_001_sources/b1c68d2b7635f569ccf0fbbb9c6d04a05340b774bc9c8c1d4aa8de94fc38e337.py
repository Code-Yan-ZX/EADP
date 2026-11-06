import copy,importlib.util,json,types,unittest
from pathlib import Path
import worker_core
BASE=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('independent_pilot_contract',BASE/'e4_c_multibudget_expansion/pilot_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((BASE/'e4_c_multibudget_expansion/pilot_plan001.json').read_text())
  cls.plan['_validated_full_cohort']=json.loads((BASE/'e4_c_latency_full/full_001/frozen.json').read_text())['schedule']['cohort']
  cls.rows=[dict(r,split='DEV',processor_expected={'fixed':'fixture'},image_start=2) for r in cls.plan['input_identities']]
  recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  cls.records={(r['task'],str(r['qid']),k):dict(r,budget=k,ds_on=True,processor={'fixed':'fixture'},generation_scalars=recipe,eos_ids=[9]) for r in cls.rows for k in worker_core.BUDGETS}
  cls.fixed={(r['task'],str(r['qid']),k):dict(r,budget=k,protocol='fixed32',min_new_tokens=32,generation_scalars=dict(recipe,max_new_tokens=32),eos_ids=[9],output_ids=[1]*32,answer='x') for r in cls.rows for k in worker_core.BUDGETS}
 def setUp(self):self.events=[];self.calls=[]
 def execute(self,row,k,protocol,panel,raw):
  self.calls.append((row['qid'],k,protocol));value={'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':protocol,'output_ids':[1]*32, 'answer':'x','eos_ids':[9],'generation_scalars':dict(panel['generation_scalars'],max_new_tokens=32),'min_new_tokens':32};raw(value);return value
 def run_all(self,plan=None,rows=None,records=None,execute=None):
  return worker_core.run(plan or self.plan,rows or self.rows,records or self.records,self.fixed,contract,execute or self.execute,self.events.append)
 def test_full_schedule_and_every_raw(self):
  result=self.run_all();self.assertEqual(result['jobs'],1520);self.assertEqual(len(self.calls),1520)
  self.assertEqual(sum(e['event']=='raw_generation' for e in self.events),1520)
  self.assertEqual(sum(e['event']=='job_completed' for e in self.events),1520)
 def test_mutated_order_or_input_fails_before_generation(self):
  p=copy.deepcopy(self.plan);p['jobs'][2],p['jobs'][3]=p['jobs'][3],p['jobs'][2]
  with self.assertRaisesRegex(ValueError,'schedule'):self.run_all(plan=p)
  rows=copy.deepcopy(self.rows);rows[0]['image_sha256']='bad'
  with self.assertRaisesRegex(ValueError,'identity'):self.run_all(rows=rows)
  self.assertEqual(self.calls,[])
 def test_failed_repeated_fixed_output_is_preserved(self):
  counts={}
  def mismatch(row,k,protocol,panel,raw):
   v=self.execute(row,k,protocol,panel,raw);key=(row['qid'],k,protocol);counts[key]=counts.get(key,0)+1
   if protocol=='fixed32' and counts[key]>1:v=dict(v,output_ids=[999])
   return v
  with self.assertRaisesRegex(ValueError,'repeat mismatch'):self.run_all(execute=mismatch)
  self.assertEqual(self.events[-1]['event'],'job_failed');self.assertEqual(self.events[-2]['event'],'job_completed');self.assertEqual(self.events[-2]['record']['output_ids'],[999])
 def test_unaccepted_panel_rejected_before_generation(self):
  refs=copy.deepcopy(self.records);next(iter(refs.values()))['processor']={'bad':True}
  with self.assertRaisesRegex(ValueError,'endpoint'):self.run_all(records=refs)
  self.assertEqual(self.calls,[])
 def test_missing_fixed_reference_stops_before_any_generation(self):
  refs=dict(self.fixed);refs.pop(next(iter(refs)))
  with self.assertRaisesRegex(ValueError,'references incomplete'):worker_core.run(self.plan,self.rows,self.records,refs,contract,self.execute,self.events.append)
  self.assertEqual(self.calls,[])
if __name__=='__main__':unittest.main()
