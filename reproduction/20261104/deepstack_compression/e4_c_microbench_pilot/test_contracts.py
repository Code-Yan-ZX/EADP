import copy,unittest
from contracts import run_jobs,COMPONENTS
class Event:
 def record(self):pass
 def synchronize(self):pass
 def elapsed_time(self,_):return 1.0
def schedule():
 jobs=[{'order':i,'task':'D','group_id':'g','component':COMPONENTS[i%3]} for i in range(60)]
 return {'stage':'pilot','images':[['D','g']],'components':list(COMPONENTS),'jobs':jobs}
class Tests(unittest.TestCase):
 def test_all_69_outputs_checked(self):
  x=schedule();count={'equal':0};journal=[]
  def eq(a,b):count['equal']+=1;return a==b
  ops={k:(lambda k=k:[k,1,2]) for k in COMPONENTS};refs={k:[k,1,2] for k in COMPONENTS}
  result=run_jobs(x,x,ops,refs,Event,lambda:None,eq,repr,journal.append,lambda:True,lambda:True,lambda:True)
  self.assertEqual((result['measurements'],result['warmup_checks'],count['equal']),(60,9,69));self.assertEqual(len(journal),70)
 def test_measured_output_failure_and_hook_failure(self):
  x=schedule();calls={k:0 for k in COMPONENTS};journal=[]
  def operation(k):
   calls[k]+=1
   return -1 if k=='selector' and calls[k]==4 else 1
  ops={k:(lambda k=k:operation(k)) for k in COMPONENTS};refs={k:1 for k in COMPONENTS}
  with self.assertRaisesRegex(ValueError,'measured full output'):run_jobs(x,x,ops,refs,Event,lambda:None,lambda a,b:a==b,repr,journal.append,lambda:True,lambda:True,lambda:True)
  self.assertEqual(journal[-1]['status'],'measurement_failed')
  with self.assertRaisesRegex(ValueError,'hooks remain'):run_jobs(x,x,ops,refs,Event,lambda:None,lambda a,b:a==b,repr,journal.append,lambda:False,lambda:True,lambda:True)
  with self.assertRaisesRegex(ValueError,'inference mode required'):run_jobs(x,x,ops,refs,Event,lambda:None,lambda a,b:a==b,repr,journal.append,lambda:True,lambda:False,lambda:True)
  with self.assertRaisesRegex(ValueError,'unexpected CUDA stream'):run_jobs(x,x,ops,refs,Event,lambda:None,lambda a,b:a==b,repr,journal.append,lambda:True,lambda:True,lambda:False)
 def test_wrong_schedule_rejected(self):
  x=schedule();y=copy.deepcopy(x);y['jobs'][1]['order']=17
  with self.assertRaisesRegex(ValueError,'frozen pilot schedule'):run_jobs(y,x,{}, {},Event,lambda:None,lambda a,b:a==b,repr,lambda _:None,lambda:True,lambda:True,lambda:True)
if __name__=='__main__':unittest.main(verbosity=2)
