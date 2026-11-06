import copy,unittest
from runner import measure_component,replay

class Event:
 def __init__(self):self.recorded=False;self.synced=False
 def record(self):self.recorded=True
 def synchronize(self):self.synced=True
 def elapsed_time(self,other):
  if not self.recorded or not other.recorded or not other.synced:raise ValueError('event sequence')
  return 1.25

class ProfilerTests(unittest.TestCase):
 def make_schedule(self):
  components=['selector','ds_merger_three_streams','ds_injection'];images=[['task',str(i)] for i in range(32)];jobs=[]
  for task,group in images:
   for repeat in range(20):
    for component in components:jobs.append({'task':task,'group_id':group,'repeat':repeat,'component':component,'order':len(jobs)})
  return {'scope':'separate batch1 DEV profiler only; not end-to-end TTFT','jobs':jobs,'images':images,'components':components,'repeat_per_image_component':20}
 def test_component_order_and_synchronization(self):
  called=[];ticks=iter([0,2_000_000]);self.assertEqual(measure_component(lambda:called.append(1),Event,lambda:next(ticks)),{'gpu_event_ms':1.25,'host_wall_ms':2.0});self.assertEqual(called,[1])
 def test_all_jobs_and_warmups_are_separate(self):
  schedule=self.make_schedule();prepared=[];calls=[];sync=[];journal=[]
  def prepare(task,group,component):
   prepared.append((group,component));return lambda:calls.append((group,component)),{'task':task,'group_id':group,'component':component}
  expected=copy.deepcopy(schedule);expected['images']=[tuple(x) for x in expected['images']]
  result=replay(schedule,expected,prepare,Event,lambda:sync.append(1),journal.append)
  self.assertEqual(result['measurements'],1920);self.assertEqual(len(journal),1920);self.assertEqual(len(prepared),96);self.assertEqual(len(sync),96);self.assertEqual(len(calls),1920+96*3)
  self.assertTrue(all(x['status']=='measured' and x['gpu_event_ms']==1.25 for x in journal))
 def test_operation_failure_is_recorded_without_later_jobs(self):
  schedule=self.make_schedule();journal=[]
  def prepare(task,group,component):
   count=[0]
   def operation():
    count[0]+=1
    if count[0]==4:raise RuntimeError('synthetic failure')
   return operation,{'task':task,'group_id':group,'component':component}
  with self.assertRaisesRegex(RuntimeError,'synthetic failure'):replay(schedule,copy.deepcopy(schedule),prepare,Event,lambda:None,journal.append)
  self.assertEqual(len(journal),1);self.assertEqual(journal[0]['status'],'failed')
if __name__=='__main__':unittest.main()
