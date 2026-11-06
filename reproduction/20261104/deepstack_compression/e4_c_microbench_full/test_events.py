import copy,unittest
from full_verify import group_events
class Tests(unittest.TestCase):
 def test_order_and_complete(self):
  events=[]
  for i in range(32):events.extend([{'status':'image_started','image_index':i},{'status':'capture_generated','image_index':i},{'status':'image_completed','image_index':i}])
  self.assertEqual(len(group_events(events)),32)
  for bad in (events[:-1],events[1:]):
   with self.assertRaises(ValueError):group_events(bad)
  bad=copy.deepcopy(events);bad[4]['image_index']=9
  with self.assertRaises(ValueError):group_events(bad)
if __name__=='__main__':unittest.main()
