import copy,unittest
import schedule
def cohort():
 a=[{'task':t,'qid':str(i),'group_id':t+str(i)} for t,n in zip(schedule.TASKS,(11,11,10)) for i in range(n)]
 w=[{'task':t,'qid':str(i+100),'group_id':t+str(i+100)} for t in schedule.TASKS for i in range(10)];return a,w
class Tests(unittest.TestCase):
 def test_full_and_balanced(self):
  a,w=cohort();jobs=schedule.build(a,w);self.assertTrue(schedule.validate(jobs,a,w));self.assertEqual(len(jobs),12);self.assertEqual(sum(r['phase']=='warmup' for j in jobs for r in j['requests']),60);self.assertEqual(sum(r['phase']=='measured' for j in jobs for r in j['requests']),320)
  for i in range(0,12,2):self.assertEqual({jobs[i]['arm'],jobs[i+1]['arm']},{'eadp','rres'});self.assertEqual(jobs[i]['requests'],jobs[i+1]['requests'])
 def test_tamper_and_overlap(self):
  a,w=cohort();jobs=schedule.build(a,w);jobs[0]['requests'].pop()
  with self.assertRaises(ValueError):schedule.validate(jobs,a,w)
  w[0]=copy.deepcopy(a[0])
  with self.assertRaises(ValueError):schedule.build(a,w)
if __name__=='__main__':unittest.main()
