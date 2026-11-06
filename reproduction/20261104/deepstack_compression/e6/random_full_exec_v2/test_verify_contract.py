"""Use one already accepted smoke record as a schema fixture, not new evaluation."""
import copy,json,unittest
from pathlib import Path
import verify_contract as v

class TestRecord(unittest.TestCase):
 def setUp(self):
  root=Path(__file__).resolve().parent.parent
  with (root/'random_smoke/model_run001/records.jsonl').open() as f:
   self.r=next(x for x in map(json.loads,f) if x['forced'] is False)
  r=self.r
  # Full panel retains summaries; the accepted smoke retained full arrays.
  r['arrays']={'kind':'summary_only','tensors':{}}
  self.choice={k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256','seed','derived_seed64','spatial_ids')}
  self.e1={k:r[k] for k in ('task','qid','group_id','message_sha256','generation_scalars','eos_ids')}
  self.e1.update(processor=r['processor'],visual_sha256=r['main_visual'],deepstack_full_sha256=r['full_deepstack'],original_position_ids=r['full_position_ids'],full_rope_delta=r['full_delta'])
  self.decode=lambda ids:r['answer']
 def test_accepted_smoke_shape(self):self.assertTrue(v.check_record(self.r,self.choice,self.e1,self.decode))
 def test_bad_position_or_seed_fails(self):
  for field,value in [('adjusted_delta',[[999999]]),('seed',0),('answer','bad')]:
   bad=copy.deepcopy(self.r);bad[field]=value
   with self.assertRaises(ValueError):v.check_record(bad,self.choice,self.e1,self.decode)
 def test_inventory_full_and_tamper(self):
  schedule=[];records=[];generated=[];journal=[]
  for i in range(384):
   row={'task':'doc','qid':str(i)}
   for seed in (20261104,20261105,20261106):
    job={'task':'doc','qid':str(i),'seed':seed}
    result={**job,'output_ids':[1],'eos_ids':[1],'generation_scalars':{'do_sample':False}}
    schedule.append((row,seed));records.append(result);generated.append({'scheduled':job,'raw':result})
    journal.extend([{'event':'started',**job},{'event':'recorded',**job}])
  self.assertTrue(v.inventory(records,generated,journal,schedule))
  bad=copy.deepcopy(generated);bad[20]['raw']['output_ids']=[2]
  with self.assertRaises(ValueError):v.inventory(records,bad,journal,schedule)
  with self.assertRaises(ValueError):v.inventory(records[:-1],generated,journal,schedule)

if __name__=='__main__':unittest.main()
