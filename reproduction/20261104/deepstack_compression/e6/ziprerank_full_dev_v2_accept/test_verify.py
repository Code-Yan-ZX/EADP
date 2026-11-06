import copy,hashlib,struct,unittest
import verify
class FakeTokenizer:
 def decode(self,*_a,**_kw):return 'answer'
class Tests(unittest.TestCase):
 def fixture(self):
  seq=1028;prompt=list(range(seq));digest=hashlib.sha256(struct.pack('<'+'q'*seq,*prompt)).hexdigest();tokens=[7,9]
  anchor={'task':'DocVQA_VAL','qid':'1','group_id':'g'}
  scalars={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  ref={'processor':{'input_ids':{'dtype':'torch.int64','shape':[1,seq],'sha256':digest}},'image_start':3,'generation_scalars':scalars,'eos_ids':[9]}
  processor={'processor':ref['processor'],'image_start':3,'visual_tokens':1024}
  row={**anchor,'arm':'user_text_256','status':'completed','prompt_sha256':digest,'generation_controls':{**scalars,'eos_token_id':[9]},'local_rope_adapter':True,'full_user_text_adapter':True,'prompt_ids_unchanged':True,'selector_restored':True,'selection_trace':{'calls':1,'input_ids_sha256':digest,'returned_exact_selected_embeddings':True,'returned_embedding_sha256':'a'*64},'selected_indices':[list(range(256))],'prefill':{'selected_count':256,'selected_unique':True,'selected_in_range':True,'positions_match':True,'visual_count':256,'ds_rows':[256]*3},'output_ids':tokens,'answer':'answer','eos':True,'truncated':False,'calls':[{'input_length':260,'cache_length_before':0},{'input_length':1,'cache_length_before':260}],'final_cache':[{'key_shape':[1,8,261,128],'value_shape':[1,8,261,128]} for _ in range(36)]}
  raw={'input_sequence_length':seq,'sequences':[prompt+tokens]}
  return [anchor,ref,processor,row,raw,FakeTokenizer()]
 def test_valid(self):
  x=self.fixture();self.assertEqual(verify.validate_item(*x)['output_count'],2)
 def test_identity_and_control_tampering(self):
  for key,value in [('group_id','other'),('prompt_sha256','bad')]:
   x=self.fixture();x[3][key]=value
   with self.assertRaises(ValueError):verify.validate_item(*x)
  x=self.fixture();x[3]['generation_controls']['top_k']=20
  with self.assertRaises(ValueError):verify.validate_item(*x)
 def test_output_and_cache_tampering(self):
  x=self.fixture();x[4]['sequences'][0][-1]=8
  with self.assertRaises(ValueError):verify.validate_item(*x)
  x=self.fixture();x[3]['final_cache'][0]['key_shape'][-2]=1
  with self.assertRaises(ValueError):verify.validate_item(*x)
 def test_selector_and_deepstack_tampering(self):
  x=self.fixture();x[3]['selection_trace']['calls']=0
  with self.assertRaises(ValueError):verify.validate_item(*x)
  x=self.fixture();x[3]['prefill']['ds_rows']=[256,255,256]
  with self.assertRaises(ValueError):verify.validate_item(*x)
if __name__=='__main__':unittest.main(verbosity=2)
