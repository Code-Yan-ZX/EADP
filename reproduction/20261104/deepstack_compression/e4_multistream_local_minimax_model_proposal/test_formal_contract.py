import copy,unittest
import formal_contract
from test_cpu import dataset

class FormalTests(unittest.TestCase):
 def build(self):
  ids=dataset()[2][0]['arms']['minimax']['ids'];start=2;n=1029;retained=list(range(start))+[start+i for i in ids]+list(range(start+1024,n))
  positions=[[[10000*a+i for i in range(n)]] for a in range(3)];pre=[[[axis[0][i] for i in retained]] for axis in positions]
  item={'task':'DocVQA_VAL','qid':'q','group_id':'g','image_sha256':'i','message_sha256':'m','ids':{'minimax':ids}}
  controls={'do_sample':False,'max_new_tokens':2048,'top_p':1.0,'top_k':1,'temperature':1.0,'repetition_penalty':1.0}
  ref={'processor':{'input_ids':{'shape':[1,n]}},'visual_sha256':{'sha':'m'},'deepstack_full_sha256':[{'sha':str(i)} for i in range(3)],'original_position_ids':positions,'full_rope_delta':[[3]],'generation_scalars':controls,'eos_ids':[2]}
  record={k:item[k] for k in ('task','qid','group_id','image_sha256','message_sha256')}
  record.update(candidate_arm='minimax',budget=256,ds_on=True,selection_mode='oracle_fixed_indices',ordered_ids=ids,spatial_ids=ids,actual_injection_verified=True,cleanup_passed=True,
   processor=ref['processor'],main_visual=ref['visual_sha256'],full_deepstack=ref['deepstack_full_sha256'],full_position_ids=positions,full_delta=[[3]],
   selected_deepstack=[{'sha':str(i)} for i in range(3)],injected_deepstack=[{'sha':str(i)} for i in range(3)],generation_scalars=controls,eos_ids=[2],original_length=n,image_start=start,retained_ids=retained,prefill_position_ids=pre,injected_mask=[[False]*start+[True]*256+[False]*3],adjusted_delta=[[771]],output_ids=[9,2],eos=True,truncated=False,decode_trace=['trace'])
  raw={'task':item['task'],'qid':'q','group_id':'g','arm':'minimax','processor_input_length':n,'generate_input_ids_shape':None,'generate_inputs_embeds_shape':[1,len(retained),4],'returned_batch':[[9,2]]}
  return item,ref,record,raw
 def test_valid_full_record(self):
  item,ref,record,raw=self.build();calls=[]
  self.assertTrue(formal_contract.case(item,'minimax',ref,record,raw,lambda *x:calls.append(x)));self.assertEqual(len(calls),1)
 def test_each_tamper_fails(self):
  for key in ('ids','processor','ds','position','mask','delta','raw','eos','cleanup','input_length'):
   item,ref,r,raw=(copy.deepcopy(x) for x in self.build())
   if key=='ids':r['ordered_ids'][0]=999
   if key=='processor':r['processor']['input_ids']['shape']=[1,99]
   if key=='ds':r['injected_deepstack'][2]['sha']='changed'
   if key=='position':r['prefill_position_ids'][2][0][3]=-1
   if key=='mask':r['injected_mask'][0][0]=True
   if key=='delta':r['adjusted_delta'][0][0]=3
   if key=='raw':raw['returned_batch'][0][-1]=99
   if key=='eos':r['eos']=False
   if key=='cleanup':r['cleanup_passed']=False
   if key=='input_length':raw['processor_input_length']=12
   with self.subTest(key=key),self.assertRaises(ValueError):formal_contract.case(item,'minimax',ref,r,raw,lambda *x:None)
