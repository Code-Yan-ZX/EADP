import unittest
from summarize import summarize,COMPONENTS
class Tests(unittest.TestCase):
 def test_image_is_sampling_unit_and_missing_rejected(self):
  identities=[{'task':'D','qid':str(i),'group_id':'g'+str(i)} for i in range(32)];events=[]
  for i,r in enumerate(identities):
   for c in COMPONENTS:
    for j in range(20):events.append({'status':'measured','image_index':i,'job':{'task':'D','group_id':r['group_id'],'component':c},'gpu_event_ms':float(i)})
  x=summarize(events,identities);self.assertEqual(x['overall']['n_images'],32);self.assertEqual(x['overall']['components']['selector']['median_of_image_medians_ms'],15.5)
  with self.assertRaises(ValueError):summarize(events[:-1],identities)
if __name__=='__main__':unittest.main()
