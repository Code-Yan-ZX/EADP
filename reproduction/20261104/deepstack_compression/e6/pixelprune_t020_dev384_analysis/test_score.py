import copy,math,unittest
from unittest import mock
import admission,score,paired
class PixelScoringTests(unittest.TestCase):
 def fixture(self):
  a={'task':'DocVQA_VAL','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','processor':{'x':1}}
  scalars={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}
  p={**a,'processor_identity':{'processor':{'x':1}},'generation_kwargs':{**scalars,'num_beams':1,'use_cache':True},'eos_token_ids':[9]}
  e={k:v for k,v in a.items() if k!='image_sha256'};e.update(generation_scalars=scalars,eos_ids=[9],answer='A',eos=True)
  b={**e,'image_sha256':'i','answer':'B'};es={**a,'arms':{'P3-D1':{'raw_answer':'A','score':1.,'independent_score':1.,'eos':True}}};bs={'answer':'B','score':0.,'independent_score':0.,'eos':True}
  return a,p,e,b,es,bs
 def test_selector_configuration(self):
  x={'pixelprune_config':{'PIXELPRUNE_ENABLED':'true','PIXELPRUNE_METHOD':'pred_2d','PIXELPRUNE_METRIC':'max','PIXELPRUNE_THRESHOLD':'0.2','PIXELPRUNE_ANCHORED':'true','PIXELPRUNE_VIT_LAYER':'0'}};score.validate_pixel_threshold(x)
  for k in x['pixelprune_config']:
   z=copy.deepcopy(x);z['pixelprune_config'][k]='bad'
   with self.assertRaises(ValueError):score.validate_pixel_threshold(z)
 def test_pair_and_identity_tamper(self):
  x=self.fixture();score.validate_pair(*x)
  for index,name,value in [(1,'image_sha256','bad'),(2,'generation_scalars',{}),(3,'eos_ids',[]),(4,'group_id','bad')]:
   y=copy.deepcopy(x);y[index][name]=value
   with self.assertRaises(ValueError):score.validate_pair(*y)
 def test_bad_score_and_duplicate(self):
  x=self.fixture();x[-1]['independent_score']=float('nan')
  with self.assertRaises(ValueError):score.validate_pair(*x)
  with self.assertRaises(ValueError):score.unique([x[0],x[0]])
 def test_gate_absent_before_prediction(self):
  with mock.patch.object(admission,'PRE',admission.HERE/'does_not_exist.json'):
   with self.assertRaises(ValueError):admission.admission()
 def test_xlsx_empty_and_number(self):
  rows=[{'task':'t','qid':'a','group_id':'g1','decoded':''},{'task':'t','qid':'b','group_id':'g2','decoded':'123'}];out=score.native_predictions(rows)
  self.assertEqual(out[score.key(rows[0])],'nan');self.assertEqual(out[score.key(rows[1])],'123.0')
 def test_full_synthetic_384_pairing(self):
  rows=[]
  for i in range(384):
   x=list(self.fixture());task=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')[i//128]
   for j in range(5):x[j].update(task=task,qid=str(i),group_id='g'+str(i))
   score.validate_pair(*x);rows.append({'group_id':'g'+str(i),'scores':dict(zip(score.ARMS,[1.,0.,1.,0.])), 'answers':dict(zip(score.ARMS,['a','b','a','b']))})
  for offset in (0,128,256):
   o=paired.analyze(rows[offset:offset+128],list(score.ARMS),score.CONTRASTS,replicates=50,seed=1);self.assertEqual(o['n_questions'],128);self.assertEqual(o['contrasts']['t020_minus_t0']['image_cluster_percentile_95_ci_pp'],[100.,100.])
 def test_paired_direction_and_nan(self):
  rows=[{'group_id':'g','scores':dict(zip(score.ARMS,[1,0,1,0])),'answers':dict(zip(score.ARMS,['a','b','a','b']))}]
  out=paired.analyze(rows,list(score.ARMS),score.CONTRASTS,replicates=10,seed=1)
  self.assertEqual(out['contrasts']['t020_minus_t0']['difference_pp'],100)
  rows[0]['scores']['pixel_t020']=float('nan')
  with self.assertRaises(ValueError):paired.analyze(rows,list(score.ARMS),score.CONTRASTS,replicates=10)
if __name__=='__main__':unittest.main()
