import math,sys,unittest
from unittest.mock import patch
import score,paired
class Tests(unittest.TestCase):
 def test_missing_prelaunch_refuses_before_prediction_loader(self):
  with patch.object(score.admission,'require',side_effect=ValueError('locked')),patch.object(score,'lines',side_effect=AssertionError('read prematurely')),patch.object(sys,'argv',['score.py','--out','unused']):
   with self.assertRaisesRegex(ValueError,'locked'):score.main()
 def test_full_inventory_and_mismatch(self):
  anchors={};records={};prior={};refs={}
  meta={'dtype':'fake'};sc={'max_new_tokens':2048};eos=[2];old={'score':.5,'independent_score':.5}
  for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
   for i in range(128):
    q=str(i);group=task+q;k=(task,q,group);a={'task':task,'qid':q,'group_id':group,'message_sha256':'m','image_sha256':'i','processor':meta};anchors[k]=a
    records[k]={**a,'generation_scalars':sc,'eos_ids':eos,'path':'local_stacktok','budget':256};prior[k]={**a,'arms':{'pixel_t0':dict(old),'pixel_t045':dict(old),'e4_k256_d1':dict(old)}};refs[(task,q)]={'processor':meta,'generation_scalars':sc,'eos_ids':eos}
  self.assertTrue(score.validate_pairs(anchors,records,prior,refs));first=next(iter(records));records[first]['message_sha256']='tamper'
  with self.assertRaises(ValueError):score.validate_pairs(anchors,records,prior,refs)
  records[first]['message_sha256']='m';prior[first]['arms']['pixel_t0']['score']=float('nan')
  with self.assertRaises(ValueError):score.validate_pairs(anchors,records,prior,refs)
 def test_native_xlsx_empty_and_bootstrap(self):
  rows=[{'task':t,'qid':str(i),'group_id':t+str(i),'answer':'' if i==0 else 'abc'} for t in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
  out=score.native_predictions(rows);self.assertEqual(len(out),384);self.assertEqual(out[score.key(rows[0])],'nan')
  synthetic=[{'group_id':str(i//2),'scores':{x:1. if x=='stacktok_k256' else 0. for x in score.ARMS},'answers':{x:'' for x in score.ARMS}} for i in range(128)]
  r=score.analyze_task(synthetic,paired);self.assertEqual(r['n_questions'],128);self.assertEqual(r['n_image_groups'],64);self.assertEqual(r['contrasts']['stacktok_minus_eadp']['image_cluster_percentile_95_ci_pp'],[100.,100.])
if __name__=='__main__':unittest.main(verbosity=2)
