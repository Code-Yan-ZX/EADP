import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_latency'))
from schedule import build
from plan import split

class MemoryPlanTests(unittest.TestCase):
 def test_smoke_uses_each_configuration_in_separate_process_schedule(self):
  rows=[{'task':task,'qid':str(i),'group_id':task+'-'+str(i),'split':'DEV'} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
  source=build(rows,[],'smoke');plans=split(source,'smoke')
  for mode,plan in plans.items():
   self.assertEqual(len(plan['jobs']),24);self.assertTrue(all(x['mode']==mode for x in plan['jobs']))
   self.assertEqual([x['latency_order'] for x in plan['jobs']],[j['order'] for j in source['schedule'] if j['mode']==mode])
  bad=copy.deepcopy(source);bad['schedule'][0]['order']=999
  with self.assertRaises(ValueError):split(bad,'smoke')
if __name__=='__main__':unittest.main()
