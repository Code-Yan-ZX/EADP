import copy,unittest
import schedule_contract as s
class Tests(unittest.TestCase):
 def setUp(self):
  self.images=[{'anchor':{'task':t,'qid':str(n),'group_id':f'g{n}','image_sha256':f'i{n}','message_sha256':f'm{n}'}} for n,t in enumerate(('D','T','A'))]
  self.jobs=s.schedule(self.images);self.events=[]
  for j in self.jobs:
   cid=f"{j['order']+1:064x}"
   self.events.extend([{'event':'started','job':j,'container_id':cid},{'event':'completed','job':j,'container_id':cid,'returncode':0,'container_exit_code':0,'container_stopped_verified':True,'timed_out':False,'interrupted':None,'result_status':'passed'}])
 def test_six_alternate_and_complete(self):
  self.assertEqual([x['arm'] for x in self.jobs],['author','wavefront','wavefront','author','author','wavefront']);self.assertEqual(s.validate_history(self.jobs,self.events)['jobs'],6)
 def test_duplicates_and_missing_rejected(self):
  bad=copy.deepcopy(self.images);bad[1]['anchor']['task']='D'
  with self.assertRaises(ValueError):s.schedule(bad)
  with self.assertRaises(ValueError):s.validate_history(self.jobs,self.events[:-1])
 def test_failed_or_reused_container_rejected(self):
  bad=copy.deepcopy(self.events);bad[3]['result_status']='failed'
  with self.assertRaises(ValueError):s.validate_history(self.jobs,bad)
  bad=copy.deepcopy(self.events);bad[2]['container_id']=bad[0]['container_id']
  with self.assertRaises(ValueError):s.validate_history(self.jobs,bad)
 def test_wrong_job_and_timeout_rejected(self):
  bad=copy.deepcopy(self.events);bad[1]['job']['arm']='wavefront'
  with self.assertRaises(ValueError):s.validate_history(self.jobs,bad)
  bad=copy.deepcopy(self.events);bad[1]['timed_out']=True
  with self.assertRaises(ValueError):s.validate_history(self.jobs,bad)
if __name__=='__main__':unittest.main(verbosity=2)
