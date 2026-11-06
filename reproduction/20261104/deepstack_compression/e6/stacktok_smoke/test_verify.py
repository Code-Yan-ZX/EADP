import unittest
from io import BytesIO
import numpy as np
import torch
from verify import local_arrays,local_identity
from identity_records import tensor_meta
class Tests(unittest.TestCase):
 def test_actual_prefill_and_three_stream_reconstruction(self):
  n=1024;k=256;length=1026;visual=torch.arange(n*2,dtype=torch.float32).reshape(n,2);pre=visual.repeat_interleave(4,0);ids=torch.tensor([[7]+[99]*n+[8]]);mask=torch.ones((1,length),dtype=torch.long);positions=torch.arange(length).view(1,1,-1).expand(3,1,-1);vm=ids==99;full=torch.zeros((1,length,2));full[:,1:1025]=visual;ds=[visual+i for i in range(1,4)];keep=torch.tensor([0]+list(range(1,257))+[1025]);sel=list(range(k));comp=length-n+k
  tensor={'premerger':pre,'averaged_premerger':pre.reshape(-1,4,2).mean(1),'visual':visual,'prefill':full.index_select(1,keep),'prefill_logits':torch.ones(3),'question_ids':torch.tensor([[1]]),'question_embeddings':torch.ones((1,2)),'prefill_mask':mask.index_select(1,keep),'prefill_positions':positions.index_select(2,keep),'visual_mask':vm.index_select(1,keep),'retained_indices':keep,**{'ds'+str(i):v for i,v in enumerate(ds)},**{'injected'+str(i):v[sel] for i,v in enumerate(ds)},'processor_input_ids':ids,'processor_attention_mask':mask,'processor_pixel_values':torch.ones(1),'processor_image_grid_thw':torch.tensor([[1,64,64]])}
  metadata={key:tensor_meta(val) for key,val in tensor.items()};proc={x:metadata['processor_'+x] for x in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
  native={'processor':proc,'prefill_embeds':tensor_meta(full),'prefill_position_ids':positions.tolist(),'injected_mask':vm.tolist(),'full_deepstack':[tensor_meta(v) for v in ds],'main_visual':tensor_meta(visual),'rope_delta':[[3]]}
  r={'processor':proc,'premerger':metadata['premerger'],'averaged_premerger':metadata['averaged_premerger'],'main_visual':metadata['visual'],'prefill_embeds':metadata['prefill'],'prefill_logits':metadata['prefill_logits'],'question_ids':metadata['question_ids'],'question_embeddings':metadata['question_embeddings'],'prefill_mask':metadata['prefill_mask'],'prefill_positions':metadata['prefill_positions'],'visual_mask':metadata['visual_mask'],'retained_indices':metadata['retained_indices'],'full_deepstack':[tensor_meta(v) for v in ds],'injected_deepstack':[metadata['injected'+str(i)] for i in range(3)],'spatial_ids':sel,'greedy_ids':sel,'selector_repeat_exact':True,'budget':k,'rope_delta':[[3]],'output_ids':[1],'decode_trace':[{'cache_position':list(range(comp)),'rope_delta':[[3]],'layer_cache_lengths':[],'layer_cache_lengths_after':[comp]*36}]}
  def raw(x):return x.detach().contiguous().view(torch.uint8).numpy()
  local=BytesIO();np.savez(local,**{key:raw(val) for key,val in tensor.items()});local.seek(0)
  original=BytesIO();np.savez(original,processor_input_ids=raw(ids),processor_attention_mask=raw(mask),prefill=raw(full),visual=raw(visual),**{'ds'+str(i):raw(v) for i,v in enumerate(ds)});original.seek(0)
  with np.load(local) as a,np.load(original) as b:
   observed=local_arrays(r,a);local_identity(native,r,observed,b);observed['injected1'][0,0]+=1
   with self.assertRaises(ValueError):local_identity(native,r,observed,b)
if __name__=='__main__':unittest.main(verbosity=2)
