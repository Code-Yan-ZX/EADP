import copy,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
spec=importlib.util.spec_from_file_location('m3_baseline_test',Path(__file__).resolve().parent/'baseline.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class BaselineTests(unittest.TestCase):
    def test_selected_inputs_and_recipe_must_match_e1(self):
        torch.manual_seed(3);embedding=torch.nn.Embedding(512,4)
        class LM(torch.nn.Module):
            def forward(self,inputs_embeds,attention_mask,position_ids,visual_pos_masks,deepstack_visual_embeds,use_cache,cache_position):
                return SimpleNamespace(last_hidden_state=inputs_embeds+sum(v.mean(0).view(1,1,-1) for v in deepstack_visual_embeds))
        lm=LM();head=torch.nn.Linear(4,16,bias=False);model=SimpleNamespace(get_input_embeddings=lambda:embedding,model=SimpleNamespace(language_model=lm),lm_head=head)
        ids=torch.tensor([[1]+[511]*1024+[2]]);streams=[torch.randn(1024,4) for _ in range(4)]
        positions=torch.arange(1026).view(1,1,-1).expand(3,1,-1);order=list(range(255,-1,-1));spatial=sorted(order)
        processor={'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.ones(1),'image_grid_thw':torch.tensor([[1,64,64]])}
        full={'input_ids':ids,'processor':processor,'streams':streams,'positions':positions,'delta':-2,'start':1,'greedy_order':order}
        row={'task':'fake','qid':'q','group_id':'g','message_sha256':'m'};retained=torch.tensor([0]+[i+1 for i in spatial]+[1025]);chosen=[x[spatial] for x in streams]
        prefill=torch.cat((embedding(ids)[:,:1],chosen[0].unsqueeze(0),embedding(ids)[:,1025:]),dim=1)
        mask=processor['attention_mask'].index_select(1,retained);injected=torch.zeros_like(mask,dtype=torch.bool);injected[:,1:257]=True
        scalars={'do_sample':False,'max_new_tokens':2048};eos=[2]
        rec={'task':'fake','qid':'q','group_id':'g','message_sha256':'m','arm':'P3-D1','forced':False,
             'processor':{k:b.meta(v) for k,v in processor.items()},'image_start':1,'original_length':1026,'ordered_ids':order,
             'spatial_ids':spatial,'retained_ids':retained.tolist(),'original_position_ids':positions.tolist(),'full_rope_delta':[[-2]],
             'visual_sha256':b.meta(streams[0]),'pruned_sha256':b.meta(chosen[0]),'deepstack_full_sha256':[b.meta(x) for x in streams[1:]],
             'prefill_embed_sha256':b.meta(prefill),'attention_mask_sha256':b.meta(mask),'prefill_position_ids':positions.index_select(-1,retained).tolist(),
             'rope_delta':[[766]],'injected_mask':injected.tolist(),'injected_deepstack':[b.meta(x) for x in chosen[1:]],
             'selected_deepstack':[b.meta(x) for x in chosen[1:]],'generation_scalars':scalars,'eos_ids':eos,'output_ids':[2],
             'answer':'','eos':True,'truncated':False}
        with torch.no_grad():
            out=lm(inputs_embeds=prefill,attention_mask=mask,position_ids=positions.index_select(-1,retained),visual_pos_masks=injected,deepstack_visual_embeds=chosen[1:],use_cache=True,cache_position=torch.arange(mask.shape[1]))
            rec['logits']=[b.meta(head(out.last_hidden_state[:,-1,:]))]
        provenance={'accepted_unique_line':True,'acceptance_sha256':'a'*64,'inputs_sha256':'b'*64,'journal_sha256':'c'*64,'review_sha256':'d'*64,'weights_identity_sha256':'e'*64,'plan_sha256':'f'*64}
        raw=json.dumps(rec).encode();receipt=b.validate(torch,model,row,full,raw,provenance,scalars,eos,lambda _:None)
        self.assertEqual(receipt['arm'],'P3-D1')
        for field in ('generation_scalars','processor','ordered_ids','injected_deepstack','prefill_position_ids'):
            bad=copy.deepcopy(rec);bad[field]=None
            with self.assertRaisesRegex(ValueError,field):b.validate(torch,model,row,full,json.dumps(bad).encode(),provenance,scalars,eos,lambda _:None)
        bad=copy.deepcopy(rec);bad['logits'][0]['sha256']='0'*64;observed=[]
        with self.assertRaisesRegex(ValueError,'first logits mismatch'):
            b.validate(torch,model,row,full,json.dumps(bad).encode(),provenance,scalars,eos,observed.append)
        self.assertFalse(observed[0]['identical'])
        with self.assertRaisesRegex(ValueError,'provenance'):b.validate(torch,model,row,full,raw,{},scalars,eos,lambda _:None)
if __name__=='__main__':unittest.main()
