"""Independent CPU black-box checks against candidate public APIs; never uses GPU."""
import copy,importlib.util,os,sys,unittest
from pathlib import Path
import contract_oracle as oracle
from test_oracle import rows
HERE=Path(__file__).resolve().parent;B=HERE.parents[1];C=B/'e6'/os.environ.get('PAIRED_TIMING_CANDIDATE','rres_eadp_timing_pilot')
RUNNER=C/('paired_runner.py' if (C/'paired_runner.py').is_file() else 'runner.py')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Shape:
 shape=(1,1)
@unittest.skipUnless(RUNNER.is_file() and (C/'schedule.py').is_file(),'candidate not yet present')
class Candidate(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.runner=load('_independent_candidate_runner',RUNNER);cls.schedule=load('_independent_candidate_schedule',C/'schedule.py')
 def test_actual_schedule_and_mutations(self):
  m,w=rows();jobs=self.schedule.build(m,w)
  self.assertTrue(self.schedule.validate(jobs,m,w));self.assertEqual(oracle.schedule(oracle.flatten_processes(jobs),m,w)['measured'],320)
  for label in ('missing','duplicate','wrong_arm','reordered','warmup_replaced'):
   bad=copy.deepcopy(jobs)
   if label=='missing':bad[0]['requests'].pop()
   elif label=='duplicate':bad[0]['requests'][-1]=copy.deepcopy(bad[0]['requests'][-2])
   elif label=='wrong_arm':bad[0]['arm']=bad[1]['arm']
   elif label=='reordered':bad[0]['requests'][-1],bad[0]['requests'][-2]=bad[0]['requests'][-2],bad[0]['requests'][-1]
   else:bad[0]['requests'][0]=copy.deepcopy(bad[0]['requests'][-1])
   with self.subTest(label=label),self.assertRaises(ValueError):self.schedule.validate(bad,m,w)
 def test_actual_endpoint_order_and_persist(self):
  events=[];ticks=iter([10,20,30,40]);raw=object()
  def sync():events.append('sync')
  def clock():events.append('clock');return next(ticks)
  def persist(x):self.assertIs(x,raw);events.append('persist')
  def invoke(boundary):
   events.append('image_read');boundary(Shape(),None);boundary(Shape(),None);events.append('model_return');self.assertIs(boundary.generation_returned(raw),raw);events.append('text_decoded');return 'answer'
  response,b=self.runner.measure(invoke,sync,clock=clock,persist=persist)
  self.assertEqual(response,'answer');self.assertEqual([b[k] for k in ('t0_ns','first_ns','generate_end_ns','response_end_ns')],[10,20,30,40]);self.assertEqual(b['criteria_calls'],2)
  self.assertEqual(events,['sync','clock','image_read','sync','clock','model_return','sync','clock','text_decoded','sync','clock','persist'])
 def test_actual_raw_survives_generation_sync_failure(self):
  calls=[];raw=object();n=[0];ticks=iter([10,20,30,40])
  def sync():
   n[0]+=1
   if n[0]==3:raise RuntimeError('generation boundary sync failed')
  def invoke(boundary):boundary(Shape(),None);boundary.generation_returned(raw);return 'unreachable'
  with self.assertRaisesRegex(RuntimeError,'boundary sync'):self.runner.measure(invoke,sync,clock=lambda:next(ticks),persist=calls.append)
  self.assertEqual(calls,[raw],'returned raw must be retained on boundary failure')
 def test_actual_failure_after_return_retains_raw(self):
  for mode in ('missing_first','decode_error','duplicate_generation'):
   raw=object();saved=[];ticks=iter([10,20,30,40])
   def invoke(boundary):
    if mode=='missing_first':boundary.generation_returned(raw)
    boundary(Shape(),None);boundary.generation_returned(raw)
    if mode=='decode_error':raise RuntimeError('decode')
    boundary.generation_returned(object())
   with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
    self.runner.measure(invoke,lambda:None,clock=lambda:next(ticks),persist=saved.append)
   self.assertEqual(saved,[raw])
 def test_actual_normalization(self):
  self.assertEqual(self.runner.normalize('rres',[1,2,11,99],[1,2]),[11,99]);self.assertEqual(self.runner.normalize('eadp',[11,99],[1,2]),[11,99])
  for arm,raw,prompt in [('rres',[1,3,99],[1,2]),('rres',[1,2],[1,2]),('eadp',[],[]),('eadp',[True],[])]:
   with self.subTest(arm=arm,raw=raw),self.assertRaises(ValueError):self.runner.normalize(arm,raw,prompt)
if __name__=='__main__':unittest.main()
