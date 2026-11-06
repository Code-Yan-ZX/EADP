import copy,unittest
from generation_control_oracle import EXPECTED,validate
class TestGenerationControlOracle(unittest.TestCase):
 def fixture(self):
  return {'prepare_calls':1,'effective_generation_config':{**EXPECTED,'eos_token_id':[151645,151643]},'explicit_kwargs':{k:EXPECTED[k] for k in ('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')}}
 def test_expected(self):self.assertTrue(validate(self.fixture(),[151645,151643]))
 def test_same_output_cannot_substitute_for_controls(self):
  for key,value in [('num_beams',2),('use_cache',False),('num_return_sequences',2),('do_sample',1),('max_new_tokens',2047)]:
   r=copy.deepcopy(self.fixture());r['effective_generation_config'][key]=value
   with self.assertRaises(ValueError,msg=key):validate(r,[151645,151643])
 def test_eos_and_config_count(self):
  for edit in ('eos','count','missing'):
   r=self.fixture()
   if edit=='eos':r['effective_generation_config']['eos_token_id']=[151643]
   elif edit=='count':r['prepare_calls']=2
   else:del r['explicit_kwargs']['top_p']
   with self.assertRaises(ValueError,msg=edit):validate(r,[151645,151643])
if __name__=='__main__':unittest.main()
