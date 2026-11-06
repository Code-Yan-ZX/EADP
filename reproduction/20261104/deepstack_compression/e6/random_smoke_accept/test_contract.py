import copy,unittest
import numpy as np
from contract import validate_record,tensor_bytes
class Tests(unittest.TestCase):
 def fixture(self):
  ids=list(range(256));start=3;n=1030;ret=list(range(start))+[start+i for i in ids]+list(range(start+1024,n));axis=list(range(n));full=[[axis],[axis],[axis]];visual=np.arange(1024*4,dtype=np.uint16).reshape(1024,4).view(np.uint8);prefill=np.zeros((1,len(ret),8),dtype=np.uint8);prefill[0,start:start+256]=visual[ids];ds=[(visual+i).astype(np.uint8) for i in range(3)];inj=[x[ids] for x in ds];logits=np.array([0x3f80],dtype=np.uint16).view(np.uint8);arrays={'visual':visual,'prefill':prefill,'prefill_logits':logits,**{'ds'+str(i):x for i,x in enumerate(ds)},**{'injected'+str(i):x for i,x in enumerate(inj)}}
  meta=lambda x:{'dtype':'torch.bfloat16','sha256':tensor_bytes(x),'shape':list(x.shape)}
  sample={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','seed':20261104,'derived_seed64':1,'spatial_ids':ids};controls={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.};proc={'input_ids':{'sha256':'x'}};e1={'task':'D','qid':'1','group_id':'g','message_sha256':'m','processor':proc,'visual_sha256':meta(visual),'deepstack_full_sha256':[meta(x) for x in ds],'original_position_ids':full,'full_rope_delta':[[0]],'generation_scalars':controls,'eos_ids':[9]}
  mask=[False]*len(ret);mask[start:start+256]=[True]*256
  rec={**sample,'budget':256,'ds_on':True,'selection_mode':'e6_random_fixed_seed','greedy_calls':1,'actual_injection_verified':True,'extra_author_greedy':True,'timing_eligible':False,'ordered_ids':ids,'processor':proc,'main_visual':meta(visual),'pruned_visual':meta(visual[ids]),'full_deepstack':[meta(x) for x in ds],'selected_deepstack':[meta(x) for x in inj],'injected_deepstack':[meta(x) for x in inj],'full_position_ids':full,'full_delta':[[0]],'image_start':start,'original_length':n,'retained_ids':ret,'prefill_position_ids':[[[axis[i] for i in ret]]]*3,'adjusted_delta':[[768]],'injected_mask':[mask],'prefill_embeds':meta(prefill),'prefill_logits':meta(logits),'forced':False,'eos':True,'truncated':False,'protocol':'natural','output_ids':[9],'eos_ids':[9],'generation_scalars':controls,'decode_trace':[{'position_ids':[[[axis[i] for i in ret]]]*3,'cache_position':list(range(len(ret))),'rope_delta':[[768]],'layer_cache_lengths':[],'layer_cache_lengths_after':[len(ret)]*36}]}
  return rec,sample,e1,arrays
 def test_matching(self):self.assertTrue(validate_record(*self.fixture()))
 def test_forced32_decode_and_all_cache_fingerprints(self):
  r,c,e,a=self.fixture();n=len(r['retained_ids']);base=r['decode_trace'][0];r['forced']=True;r['eos']=False;r['truncated']=False;r['protocol']='forced33';r['output_ids']=[100]*33;r['eos_ids']=[];r['generation_scalars']={**r['generation_scalars'],'max_new_tokens':33};r['decode_trace']=[base]+[{'position_ids':[[[n+i-1+768]]]*3,'cache_position':[n+i-1],'rope_delta':[[768]],'layer_cache_lengths':[n+i-1]*36,'layer_cache_lengths_after':[n+i]*36} for i in range(1,33)]
  r['cache_fingerprints']=[[{k:{'dtype':'torch.bfloat16','shape':[1,8,n+i,128],'sha256':'a'*64} for k in ('key','value')} for _ in range(36)] for i in range(33)]
  processor={}
  for key in ('input_ids','attention_mask','pixel_values','image_grid_thw'):
   x=np.array([1,2],dtype=np.uint8);a['processor_'+key]=x;processor[key]={'sha256':tensor_bytes(x)}
  r['processor']=processor;e['processor']=processor
  self.assertTrue(validate_record(r,c,e,a))
  r['cache_fingerprints'][32][35]['value']['sha256']='invalid'
  with self.assertRaises(ValueError):validate_record(r,c,e,a)
 def test_source_selection_decode_and_nonfinite_rejected(self):
  for category in ('selection','visual','ds','position','cache','logits','e1','termination'):
   r,c,e,a=self.fixture()
   if category=='selection':r['ordered_ids'][0]=999
   if category=='visual':a['visual'][0,0]^=1
   if category=='ds':a['injected0'][0,0]^=1
   if category=='position':r['prefill_position_ids'][0][0][0]=999
   if category=='cache':r['decode_trace'][0]['layer_cache_lengths_after']=[0]*36
   if category=='logits':a['prefill_logits'][:]=np.array([0x7f80],dtype=np.uint16).view(np.uint8);r['prefill_logits']['sha256']=tensor_bytes(a['prefill_logits'])
   if category=='e1':e['message_sha256']='wrong'
   if category=='termination':r['output_ids']=[1]
   with self.subTest(category=category),self.assertRaises(ValueError):validate_record(r,c,e,a)
if __name__=='__main__':unittest.main()
