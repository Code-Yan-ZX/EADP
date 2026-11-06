import copy,hashlib,unittest
import numpy as np
from full_imports import HERE,load,prepare
contract=load('eadp_selected_full_contract',HERE/'full_contract.py')
class Tokenizer:
 def decode(self,ids,**kw):return 'answer'
def meta(a,shape,dtype='torch.bfloat16'):
 return {'dtype':dtype,'shape':shape,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def fixture():
 n=256;start=2;length=1027;spatial=list(range(n));retained=list(range(start))+[start+i for i in spatial]+[1026]
 arrays={name:(np.arange(4096,dtype=np.uint8).reshape(1024,4)+i).astype(np.uint8) for i,name in enumerate(('visual','ds0','ds1','ds2'))}
 full=[meta(arrays[k],[1024,2]) for k in arrays];picked=[meta(a[spatial],[256,2]) for a in arrays.values()]
 pf=np.concatenate((np.zeros((2,4),dtype=np.uint8),arrays['visual'][spatial],np.zeros((1,4),dtype=np.uint8)))[None,:,:]
 positions=[[[i for i in range(length)]] for _ in range(3)];pos=[[[axis[0][i] for i in retained]] for axis in positions]
 gen={'max_new_tokens':2048,'do_sample':False};reference={'processor':{'identity':'same'},'visual_sha256':full[0],
  'deepstack_full_sha256':full[1:],'original_position_ids':positions,'full_rope_delta':[[0]],'image_start':start,
  'original_length':length,'generation_scalars':gen,'eos_ids':[7]}
 mask=meta(np.ones((1,len(retained)),dtype=np.int64),[1,len(retained)],'torch.int64')
 rec={'method':'div','budget':256,'ds_on':True,'selection_mode':'author_div','selector_calls':1,'greedy_calls':0,
  'image_start':start,'original_length':length,'ordered_ids':spatial,'spatial_ids':spatial,'retained_ids':retained,
  'full_position_ids':positions,'prefill_position_ids':pos,'full_delta':[[0]],'adjusted_delta':[[768]],
  'processor':reference['processor'],'main_visual':full[0],'full_deepstack':full[1:],'pruned_visual':picked[0],
  'selected_deepstack':picked[1:],'injected_mask':[[False,False]+[True]*256+[False]],'injected_deepstack':picked[1:],
  'output_ids':[7],'answer':'answer','generation_scalars':gen,'eos_ids':[7],'max_new_tokens':2048,'eos':True,'truncated':False,'forced':False,
  'prefill_mask':mask,'prefill_embeds':meta(pf,[1,len(retained),2]),'prefill_text_before':meta(pf[:,:2],[1,2,2]),
  'prefill_text_after':meta(pf[:,258:],[1,1,2]),'visual_calls':1,'main_merger_calls':1,'ds_calls':[1,1,1],
  'block_calls':[1,1],'actual_injection_verified':True,
  'decode_trace':[{'cache_position':list(range(len(retained))),'rope_delta':[[768]],'layer_cache_lengths':[],
                    'layer_cache_lengths_after':[len(retained)]*36,'position_ids':pos}]}
 rec['arrays']={'kind':'summary_only','tensors':{'visual':full[0],'prefill':rec['prefill_embeds'],
   **{'ds'+str(i):full[i+1] for i in range(3)},**{'injected'+str(i):picked[i+1] for i in range(3)}}}
 return rec,reference
class ContractTests(unittest.TestCase):
 def test_adapted_summary_and_injection_tampering(self):
  r,ref=fixture();self.assertTrue(contract.validate_adapted(r,ref,Tokenizer()))
  bad=copy.deepcopy(r);bad['injected_deepstack']=copy.deepcopy(bad['injected_deepstack']);bad['injected_deepstack'][0]['sha256']='0'*64
  with self.assertRaises(ValueError):contract.validate_adapted(bad,ref,Tokenizer())
 def test_incomplete_output_and_wrong_selector_are_rejected(self):
  r,ref=fixture();bad=copy.deepcopy(r);bad.update(output_ids=[8],eos=False,truncated=False)
  with self.assertRaisesRegex(ValueError,'output'):contract.validate_adapted(bad,ref,Tokenizer())
  bad=copy.deepcopy(r);bad['selector_calls']=0
  with self.assertRaisesRegex(ValueError,'author selected'):contract.validate_adapted(bad,ref,Tokenizer())
 def test_stock_summary_requires_retained_ids_and_real_trace(self):
  r,ref=fixture();size=len(r['retained_ids']);proc={k:meta(np.ones((1,8),dtype=np.uint8),[1,1],'torch.int64')
      for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  ref['processor']=proc;r.update(path='author_stock',processor=proc,injected_mask=None,injected_deepstack=None,
      rope_delta=[[0]],prefill_logits=meta(np.ones((1,4),dtype=np.uint8),[1,2]),
      prefill_position_ids=[[[i for i in range(size)]] for _ in range(3)])
  r['decode_trace']=[{'cache_position':list(range(size)),'rope_delta':[[0]],'layer_cache_lengths':[],
                      'layer_cache_lengths_after':[size]*36,'position_ids':r['prefill_position_ids']}]
  r['arrays']={'kind':'summary_only','tensors':{'visual':r['main_visual'],'pruned':r['pruned_visual'],
    'prefill':r['prefill_embeds'],'prefill_logits':r['prefill_logits'],**{'ds'+str(i):v for i,v in enumerate(r['full_deepstack'])},
    **{'processor_'+k:v for k,v in proc.items()}}}
  self.assertTrue(contract.validate_stock(r,ref,Tokenizer()))
  bad=copy.deepcopy(r);bad['retained_ids'][2]+=1
  with self.assertRaisesRegex(ValueError,'selected indices'):contract.validate_stock(bad,ref,Tokenizer())
  bad=copy.deepcopy(r);bad['rope_delta']=[[1]];bad['decode_trace'][0]['rope_delta']=[[1]]
  with self.assertRaisesRegex(ValueError,'P1 positions or delta'):contract.validate_stock(bad,ref,Tokenizer())
if __name__=='__main__':unittest.main()
