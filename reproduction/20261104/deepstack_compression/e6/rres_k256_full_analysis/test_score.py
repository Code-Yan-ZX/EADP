import copy,unittest
import score

def sample():
 anchors={};target={};e1={};eadp={};random_records={};random_scores={};e1_full={}
 for task in score.TASKS:
  for i in range(128):
   k=(task,str(i),task+str(i));p={'input_ids':{'shape':[1,1028]},'pixel_values':{'shape':[4096,1536]},'image_grid_thw':{'shape':[1,3]}}
   a={'task':task,'qid':str(i),'group_id':k[2],'image_sha256':'image','message_sha256':'message','processor':p};anchors[k]=a
   e={'message_sha256':'message','processor':p,'answer':'same','eos':True,'generation_scalars':{'do_sample':False},'eos_ids':[2]};e1[k]=e
   old={'score':.5,'independent_score':.5,'raw_answer':'same','eos':True};eadp[k]={'arms':{'P3-D1':old,'P1-D1':old}};e1_full[k]=copy.deepcopy(e)
   low={'input_ids':{'shape':[1,260]},'pixel_values':{'shape':[1024,1536]},'image_grid_thw':{'shape':[1,3]}}
   target[k]={**a,'processor':low,'resolution':512,'visual_tokens':256,'generation_scalars':e['generation_scalars'],'eos_ids':[2],'hooks_and_methods_restored':True,'answer':'out','eos':True,'truncated':False}
   arms={'eadp':old}
   for seed in score.SEEDS:
    random_records[(*k,seed)]={**a,'seed':seed,'generation_scalars':e['generation_scalars'],'eos_ids':[2],'budget':256,'ds_on':True,'protocol':'natural','forced':False,'answer':'rnd','eos':True,'truncated':False}
    arms[str(seed)]={'score':.4,'independent_score':.4,'answer':'rnd','eos':True,'truncated':False}
   random_scores[k]={'arms':arms}
 return anchors,target,e1,eadp,random_records,random_scores,e1_full

class ScoreTests(unittest.TestCase):
 def test_384_pairing_and_failures(self):
  data=sample();self.assertTrue(score.validate_pairs(*data));k=next(iter(data[0]));bad=copy.deepcopy(data);del bad[1][k]
  with self.assertRaises(ValueError):score.validate_pairs(*bad)
  bad=copy.deepcopy(data);bad[1][k]['image_sha256']='wrong'
  with self.assertRaises(ValueError):score.validate_pairs(*bad)
  bad=copy.deepcopy(data);bad[4][(*k,score.SEEDS[0])]['generation_scalars']={'do_sample':True}
  with self.assertRaises(ValueError):score.validate_pairs(*bad)
  bad=copy.deepcopy(data);bad[5][k]['arms'][str(score.SEEDS[1])]['independent_score']=.1
  with self.assertRaises(ValueError):score.validate_pairs(*bad)
 def test_xlsx_is_per_task(self):
  rows=[{'task':score.TASKS[0],'qid':'1','group_id':'g1','answer':'123'}, {'task':score.TASKS[0],'qid':'2','group_id':'g2','answer':''}, {'task':score.TASKS[1],'qid':'1','group_id':'g3','answer':'word'}]
  out=score.native_predictions(rows);self.assertEqual(len(out),3);self.assertEqual(out[score.key(rows[2])],'word')
 def test_unapproved_gate_does_not_reach_truth(self):
  from unittest.mock import patch
  import sys
  with patch.object(score,'load_gate',side_effect=RuntimeError('unapproved')),patch.object(sys,'argv',['score.py','--out',str(score.D/'score001')]):
   with self.assertRaisesRegex(RuntimeError,'unapproved'):score.main()
  self.assertNotIn('metrics',sys.modules)
 def test_gate_cache_pollution_rejected(self):
  from unittest.mock import patch
  import sys,types
  with patch.dict(sys.modules,{'_rres_score_admission':types.ModuleType('_rres_score_admission')}):
   with self.assertRaises(ValueError):score.load_gate()
if __name__=='__main__':unittest.main()
