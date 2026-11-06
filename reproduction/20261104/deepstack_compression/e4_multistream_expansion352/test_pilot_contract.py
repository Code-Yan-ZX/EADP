import copy,unittest
import pilot_contract as p
class PilotTests(unittest.TestCase):
 def rows(self):
  x=dict(task='DocVQA_VAL',qid='1',group_id='g',image_sha256='i',message_sha256='m',processor={'ids':'abc'},main_visual={'hash':'a'},full_deepstack=[{'hash':'x'},{'hash':'y'},{'hash':'z'}]);return x,dict(x,budget=256,ds_on=True,selection_mode='author_greedy',actual_injection_verified=True)
 def test_identity(self):
  a,r=self.rows();self.assertTrue(p.check_identity(a,a,r))
  for field in ('group_id','message_sha256','processor','main_visual','full_deepstack'):
   bad=copy.deepcopy(a);bad[field]='changed'
   with self.assertRaises(ValueError):p.check_identity(a,bad,r)
  r['ds_on']=False
  with self.assertRaises(ValueError):p.check_identity(a,a,r)
 def test_time(self):
  d={k:None for k in p.TIMES};self.assertTrue(p.check_elapsed(d))
  d[p.TIMES[0]]=-1
  with self.assertRaises(ValueError):p.check_elapsed(d)
  del d[p.TIMES[0]]
  with self.assertRaises(ValueError):p.check_elapsed(d)
 def test_no_partial_success(self):
  a=[dict(task='T',qid=str(i)) for i in range(9)];b=[dict(x,status='passed') for x in a];self.assertTrue(p.check_complete(a,b));b[4]['status']='failed';self.assertFalse(p.check_complete(a,b))
  with self.assertRaises(ValueError):p.check_complete(a,b[:8])
  b[0]['qid']='7'
  with self.assertRaises(ValueError):p.check_complete(a,b)
if __name__=='__main__':unittest.main()
