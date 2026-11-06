import copy,math,unittest
from unittest import mock
import admission,score,paired
class Tests(unittest.TestCase):
 def fixture(self):
  a={'task':'DocVQA_VAL','qid':'1','group_id':'g','message_sha256':'m'}
  scalars={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  z={k:a[k] for k in ('task','qid','group_id')};z.update(arm='user_text_256',status='completed',local_rope_adapter=True,full_user_text_adapter=True,generation_controls={**scalars,'eos_token_id':[9]},answer='answer')
  p={'processor':{'hash':'x'}};e={**a,'processor':p['processor'],'generation_scalars':scalars,'eos_ids':[9],'answer':'base','eos':True}
  es={k:a[k] for k in ('task','qid','group_id')};es['arms']={'P3-D1':{'raw_answer':'base','eos':True,'score':.5,'independent_score':.5}}
  return [a,z,p,e,es,'user_text_256']
 def test_pair_and_mismatch(self):
  x=self.fixture();score.validate_pair(*x)
  for i,k,value in [(1,'group_id','bad'),(1,'full_user_text_adapter',False),(3,'eos_ids',[]),(3,'generation_scalars',{})]:
   x=self.fixture();x[i][k]=value
   with self.assertRaises(ValueError):score.validate_pair(*x)
 def test_admission_hash_and_preloaded_rejected(self):
  import sys,types
  with self.assertRaises(ValueError):score.load_admission(admission.HERE/'admission.py','0'*64)
  with mock.patch.dict(sys.modules,{'_zip_full_score_admission':types.ModuleType('fake')}):
   with self.assertRaises(ValueError):score.load_admission(admission.HERE/'admission.py',score.ADMISSION_SHA)
 def test_bad_reference_score(self):
  x=self.fixture();x[4]['arms']['P3-D1']['independent_score']=float('nan')
  with self.assertRaises(ValueError):score.validate_pair(*x)
 def test_prelaunch_absent_blocks(self):
  with mock.patch.object(score.admission,'PRE',score.admission.HERE/'does_not_exist.json'):
   with self.assertRaises(ValueError):admission.admission()
 def test_xlsx_separate_normalization(self):
  rows=[{'task':'t','qid':'1','group_id':'a','answer':''},{'task':'t','qid':'2','group_id':'b','answer':'123'}];v=score.native_predictions(rows)
  self.assertEqual(v[score.key(rows[0])],'nan');self.assertEqual(v[score.key(rows[1])],'123.0')
 def test_bootstrap_direction_and_nan(self):
  rows=[{'group_id':str(i),'scores':dict(zip(score.ARMS,[1.,.5,0.])),'answers':dict(zip(score.ARMS,['a','b','c']))} for i in range(128)]
  out=paired.analyze(rows,list(score.ARMS),score.CONTRASTS,replicates=50,seed=1)
  self.assertEqual(out['contrasts']['prefix_minus_eadp']['difference_pp'],100.)
  rows[0]['scores']['eadp_p3_d1']=float('nan')
  with self.assertRaises(ValueError):paired.analyze(rows,list(score.ARMS),score.CONTRASTS)
 def test_full_synthetic_inventory(self):
  rows=[]
  for i in range(384):
   x=self.fixture();task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[i//128]
   for j in (0,1,3,4):x[j].update(task=task,qid=str(i),group_id='g'+str(i))
   score.validate_pair(*x);rows.append(x[1])
  self.assertEqual(len(score.unique(rows)),384)
if __name__=='__main__':unittest.main(verbosity=2)
