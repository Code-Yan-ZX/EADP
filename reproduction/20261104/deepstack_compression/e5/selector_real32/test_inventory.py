import copy,unittest
from case_contract import partition
from trials import measurement_order
class Tests(unittest.TestCase):
 def fixture(self):
  pilot=[{'task':'D','qid':str(i),'group_id':'g'+str(i)} for i in range(32)];events=[{'status':'synthetic_completed','cases':[]}]
  for i,row in enumerate(pilot):
   events.append(dict(status='image_started',image_index=i,**row))
   for k in (128,256,512):
    traces=[{'repeat':n,'implementation':name,'status':'measured'} for n in range(5) for name in measurement_order(n)]
    events.extend({'status':'case_measurement','image_index':i,'budget':k,'measurement':m} for m in traces)
    events.append({'status':'case_completed','image_index':i,'budget':k,'result':{'status':'measured','measurements':traces}})
   events.append(dict(status='image_completed',image_index=i,**row))
  return pilot,events
 def test_all_96_and_missing_case_rejected(self):
  p,e=self.fixture();self.assertEqual(len(partition(e,p)),32)
  bad=copy.deepcopy(e);bad.pop(next(i for i,x in enumerate(bad) if x.get('status')=='case_completed'))
  with self.assertRaises(ValueError):partition(bad,p)
 def test_wrong_image_and_missing_measurement_rejected(self):
  p,e=self.fixture();bad=copy.deepcopy(e);bad[next(i for i,x in enumerate(bad) if x.get('status')=='image_completed')]['qid']='other'
  with self.assertRaises(ValueError):partition(bad,p)
  bad=copy.deepcopy(e);bad.pop(next(i for i,x in enumerate(bad) if x.get('status')=='case_measurement'))
  with self.assertRaises(ValueError):partition(bad,p)
if __name__=='__main__':unittest.main()

class CandidateTests(unittest.TestCase):
 def test_matching_and_falsely_claimed_output(self):
  from case_contract import validate_candidate_ids
  original=[0,1];measurements=[]
  for n in range(5):
   for method in measurement_order(n):measurements.append({'repeat':n,'implementation':method,'status':'measured','gpu_event_ms':1.,'host_wall_ms':1.,'equals_author':True,'ordered_ids':[0,1]})
  case={'status':'measured','identity':{'author_equals_endpoint':True,'fused_ordered_ids':[0,1],'graph_ordered_ids':[0,1],'fused_equals_author':True,'graph_equals_author':True},'measurements':measurements,'candidate_matches_author':{'fused':True,'graph':True}}
  validate_candidate_ids(case,original,{'fused':True,'graph':True});case['measurements'][0]['ordered_ids']=[1,0]
  with self.assertRaises(ValueError):validate_candidate_ids(case,original,{'fused':True,'graph':True})
