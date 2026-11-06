import copy,hashlib,struct,unittest
import request,verify_contract
class VerifyTest(unittest.TestCase):
 def fixture(self):
  identity={'task':'T','qid':'1','group_id':'G','processor':{'input_ids':{'dtype':'torch.int64','shape':[1,2],'sha256':hashlib.sha256(struct.pack('<2q',3,4)).hexdigest()}}};job={'arm':'author','block':0};r={'status':'completed','task':'T','qid':'1','group_id':'G','arm':'author','processor':identity['processor'],'recipe':request.RECIPE,'rope_state_reset_before_request':True,'hook_cleanup_verified':True,'cache_check':{'initial_kv_lengths':[0]*36,'initial_cache_position':0,'cache_position_check_synchronizes':True},'selected_indices':[1,3],'selected_count':2,'input_length':2,'first_token_id':9,'eos_ids':[9],'first_token_is_eos':True,'boundaries':{'t0_before_image_read_ns':100,'first_token_selected_ns':110,'generation_complete_ns':120,'first_criteria_length':3,'time_namespace':'time:[1]','explicit_sync_boundaries':['before_t0','first_token','generation_end']}};raw={'request':{'phase':'measure','index':1,'block':0,'arm':'author','task':'T','qid':'1','group_id':'G'},'full_sequence_ids':[[3,4,9]]};return r,raw,identity,job
 def test_actual_identity_and_time(self):
  r,raw,i,j=self.fixture();self.assertEqual(verify_contract.check(r,raw,i,j,'measure',1)['first_ns'],110)
  bad=copy.deepcopy(raw);bad['full_sequence_ids'][0][0]=4
  with self.assertRaises(ValueError):verify_contract.check(r,bad,i,j,'measure',1)
 def test_one_layer_or_time_mutation_rejected(self):
  r,raw,i,j=self.fixture();r['cache_check']['initial_kv_lengths'][17]=1
  with self.assertRaises(ValueError):verify_contract.check(r,raw,i,j,'measure',1)
  r,raw,i,j=self.fixture();r['boundaries']['first_token_selected_ns']=100
  with self.assertRaises(ValueError):verify_contract.check(r,raw,i,j,'measure',1)
if __name__=='__main__':unittest.main()
