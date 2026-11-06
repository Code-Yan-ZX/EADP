"""Synthetic tests for the independent oracle; no model/GPU/DEV outputs."""
import copy,unittest
import contract_oracle as o
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def rows():
 measured=[];warm=[]
 for task,n in zip(TASKS,(11,11,10)):
  for i in range(n):measured.append({'task':task,'qid':'m'+str(i),'group_id':task+'m'+str(i)})
  for i in range(10):warm.append({'task':task,'qid':'w'+str(i),'group_id':task+'w'+str(i)})
 return measured,warm
def make_processes(measured,warm):
 out=[]
 for task in TASKS:
  m=[x for x in measured if x['task']==task];w=[x for x in warm if x['task']==task]
  for block in (0,1):
   for arm in (('rres','eadp') if (len(out)//2)%2==0 else ('eadp','rres')):
    requests=[{**r,'phase':'warmup','replicate':0} for r in w[block*5:(block+1)*5]]
    requests += [{**r,'phase':'measured','replicate':rep} for r in m[block*((len(m)+1)//2):(block+1)*((len(m)+1)//2)] for rep in range(5)]
    out.append({'order':len(out),'task':task,'arm':arm,'block':block,'requests':requests})
 return out
def valid_endpoint():return {'arm':'rres','raw':[1,2,11,99],'prompt':[1,2],'same_arm_expected':[11,99],'eos_ids':[99],'eos':True,'truncated':False,'t0':1,'first':2,'generation_done':3,'response_done':4,'first_calls':1,'generation_calls':1,'persist_before_response':False,'time_namespace':'time:[1]','clock':'CLOCK_MONOTONIC'}
class TestEndpoints(unittest.TestCase):
 def test_distinct_output_formats(self):
  self.assertEqual(o.normalize('rres',[1,2,11,99],[1,2]),[11,99]);self.assertEqual(o.normalize('eadp',[11,99],[1,2]),[11,99])
  with self.assertRaises(ValueError):o.normalize('rres',[1,3,11],[1,2])
  with self.assertRaises(ValueError):o.endpoint({**valid_endpoint(),'arm':'eadp'})
 def test_valid_and_equal_end_boundary(self):
  r=valid_endpoint();self.assertEqual(o.endpoint(r),[11,99]);self.assertEqual(o.endpoint({**r,'response_done':3}),[11,99])
 def test_time_or_event_rejection(self):
  for update in ({'first':1},{'generation_done':1},{'response_done':2},{'first_calls':0},{'first_calls':2},{'generation_calls':2},{'persist_before_response':True},{'clock':None}):
   with self.subTest(update=update),self.assertRaises(ValueError):o.endpoint({**valid_endpoint(),**update})
 def test_wrong_arm_reference_and_eos(self):
  for update in ({'same_arm_expected':[99]},{'eos':False},{'truncated':True},{'raw':[1,2]},{'arm':'other'}):
   with self.subTest(update=update),self.assertRaises(ValueError):o.endpoint({**valid_endpoint(),**update})
class TestSchedule(unittest.TestCase):
 def setUp(self):self.m,self.w=rows();self.p=make_processes(self.m,self.w)
 def check(self,p=None):return o.schedule(o.flatten_processes(p or self.p),self.m,self.w)
 def test_full_schedule(self):self.assertEqual(self.check(),{'jobs':380,'measured':320,'warmups':60,'processes':12})
 def test_missing_duplicate_and_rep(self):
  for operation in ('missing','duplicate','wrong_rep'):
   p=copy.deepcopy(self.p);req=p[0]['requests']
   if operation=='missing':req.pop()
   elif operation=='duplicate':req[-1]=copy.deepcopy(req[-2])
   else:req[-1]['replicate']=5
   with self.subTest(operation=operation),self.assertRaises(ValueError):self.check(p)
 def test_wrong_process_task_and_arm(self):
  for operation in ('task','arm','block','process_order'):
   p=copy.deepcopy(self.p)
   if operation=='task':p[0]['requests'][0]['task']=TASKS[1]
   elif operation=='arm':p[0]['arm']=p[1]['arm']
   elif operation=='block':p[0]['block']=1
   else:p[0]['order']=1
   with self.subTest(operation=operation),self.assertRaises(ValueError):self.check(p)
 def test_late_warmup_and_group_overlap(self):
  p=copy.deepcopy(self.p);p[0]['requests'][0],p[0]['requests'][-1]=p[0]['requests'][-1],p[0]['requests'][0]
  with self.assertRaises(ValueError):self.check(p)
  bad=copy.deepcopy(self.w);bad[0]['group_id']=self.m[0]['group_id']
  with self.assertRaises(ValueError):o.schedule(o.flatten_processes(self.p),self.m,bad)
if __name__=='__main__':unittest.main()
