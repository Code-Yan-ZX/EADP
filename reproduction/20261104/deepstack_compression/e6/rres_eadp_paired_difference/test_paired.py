import copy,unittest
from paired import compute,TASKS
def rows():
 anchors=[{'task':t,'qid':str(i),'group_id':t+str(i)} for t in TASKS for i in range(128)];score=[{**a,'rres':{'score':1. if i%3==0 else 0.,'independent_score':1. if i%3==0 else 0.},'eadp':{'score':1. if i%3==1 else 0.,'independent_score':1. if i%3==1 else 0.}} for t in TASKS for i in range(128) for a in [{'task':t,'qid':str(i),'group_id':t+str(i)}]];return score,anchors
class Tests(unittest.TestCase):
 def test_counts_and_bootstrap(self):
  s,a=rows();v=compute(s,a,draws=50);self.assertEqual(len(v['per_question']),384)
  for x in v['tasks'].values():self.assertEqual(x['rres_higher']+x['ties']+x['eadp_higher'],128)
 def test_missing_duplicate_and_invalid_scores(self):
  s,a=rows()
  with self.assertRaises(ValueError):compute(s[:-1],a,draws=2)
  s[-1]=copy.deepcopy(s[0])
  with self.assertRaises(ValueError):compute(s,a,draws=2)
  s,a=rows();s[0]['rres']['independent_score']=.5
  with self.assertRaises(ValueError):compute(s,a,draws=2)
  s,a=rows();s[0]['rres']['independent_score']=float('nan')
  with self.assertRaises(ValueError):compute(s,a,draws=2)
if __name__=='__main__':unittest.main()
