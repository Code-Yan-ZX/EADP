import copy,unittest
import reference
class ReferenceTest(unittest.TestCase):
 def fixture(self):
  a={'task':'T','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m'}
  r={**a,'status':'completed','selected_indices':[0,2,9],'selected_count':3,'generated_count':2,'ends_eos':True,'eos_token_ids':[5],'processor_identity':{'processor':{'x':1}},'generation_kwargs':{'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.,'num_beams':1,'use_cache':True},'diagnostic_checks':{k:True for k in ('unique_selection','vision_and_three_ds','same_main_text_positions','natural_complete','calls_match_tokens','hooks_removed','positions_and_kv')}}
  return a,r,{'prompt_length':2,'full_sequence_ids':[7,8,4,5],'generated_token_ids':[4,5]}
 def test_same_t0_identity(self):
  a,r,raw=self.fixture();self.assertEqual(reference.validate_t0(r,raw,a)['selected_indices'],[0,2,9])
 def test_reject_wrong_identity_indices_recipe_tokens_or_diagnostics(self):
  a,r,raw=self.fixture()
  for key,value in [('group_id','other'),('selected_indices',[0,2,2]),('selected_indices',[0,2,1024]),('generation_kwargs',{}),('diagnostic_checks',{})]:
   changed=copy.deepcopy(r);changed[key]=value
   with self.assertRaises(ValueError):reference.validate_t0(changed,raw,a)
  bad=copy.deepcopy(raw);bad['full_sequence_ids'][-1]=6
  with self.assertRaises(ValueError):reference.validate_t0(r,bad,a)
