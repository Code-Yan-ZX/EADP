import unittest
from diagnostic_core import schedule,run,compare_panel,PANEL_FIELDS
class Tests(unittest.TestCase):
 def setUp(self):self.rows=[{'task':str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)]
 def test_fixed_jobs_and_corruption_fail(self):
  x=schedule(self.rows);self.assertEqual(len(x),16);self.assertEqual(sum(j['forced'] for j in x),4)
  self.rows[0]['split']='FINAL'
  with self.assertRaises(ValueError):schedule(self.rows)
 def test_panel_mismatch_fail(self):
  x={k:1 for k in PANEL_FIELDS};y=dict(x,actual_injection_verified=True);self.assertTrue(compare_panel(x,y));y['adjusted_delta']=2
  with self.assertRaisesRegex(ValueError,'adjusted_delta'):compare_panel(x,y)
 def test_sequential_raw_durability_and_failure(self):
  events=[];refs={}
  for j in schedule(self.rows):
   if not j['forced']:refs[(j['task'],j['qid'],j['budget'])]={**{k:1 for k in PANEL_FIELDS},'actual_injection_verified':True}
  def fresh(row,budget,token,raw,archive):raw({'ids':[9]});return {k:1 for k in PANEL_FIELDS}
  def light(row,budget,token,raw,archive):raw({'ids':[9]});return {k:1 for k in PANEL_FIELDS}
  result=run(self.rows,refs,7,fresh,light,lambda j,l:None,events.append,lambda a,b:{},lambda a,b:{})
  self.assertEqual(result['generations'],32);self.assertEqual(sum(x['event']=='raw_generation' for x in events),32)
  def bad(row,budget,token,raw,archive):raw({'ids':[9]});raise ValueError('after raw')
  events=[]
  with self.assertRaises(ValueError):run(self.rows,refs,7,bad,light,lambda j,l:None,events.append,lambda a,b:{},lambda a,b:{})
  self.assertEqual([x['event'] for x in events],['arm_started','raw_generation','arm_failed'])
if __name__=='__main__':unittest.main(verbosity=2)
