"""CPU contracts for actual native invocation and strict identity."""
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from native_contract import E4,compare_pair,select_anchors
from native_capture import run_native,CacheFingerprints
from native_verify import verify_array
from identity_records import tensor_meta
import native_launch
sys.path.insert(0,str(E4))
from test_off_adapter import FakeWrapper


class NativeTests(unittest.TestCase):
    def test_exact_pair_rejects_output_logits_or_kv_difference(self):
        native={'path':'stock_native','forced':False,'full_deepstack':[1,2,3],'injected_deepstack':[1,2,3],
                'cache_fingerprints':[[{'key':'x','value':'y'}]*36]}
        names={'task':'task','qid':'qid','group_id':'group_id','image_sha256':'image_sha256','message_sha256':'message_sha256',
               'processor':'processor','main_visual':'main_visual','prefill_embeds':'prefill_embeds','prefill_mask':'prefill_mask',
               'prefill_position_ids':'prefill_position_ids','rope_delta':'adjusted_delta','injected_mask':'injected_mask',
               'prefill_logits':'prefill_logits','output_ids':'output_ids','answer':'answer','eos_ids':'eos_ids','eos':'eos',
               'truncated':'truncated','max_new_tokens':'max_new_tokens','generation_scalars':'generation_scalars','decode_trace':'decode_trace'}
        budget={'budget':1024,'ds_on':True,'selection_mode':'keep_all','greedy_calls':0,'spatial_ids':list(range(1024)),
                'ordered_ids':list(range(1024)),'forced':False,'full_deepstack':[1,2,3],'injected_deepstack':[1,2,3],
                'cache_fingerprints':copy.deepcopy(native['cache_fingerprints'])}
        for a,b in names.items():native[a]='same';budget[b]='same'
        self.assertTrue(compare_pair(native,budget))
        for key in ('prefill_logits','output_ids','cache_fingerprints','injected_deepstack','decode_trace'):
            bad=copy.deepcopy(budget);bad[key]='different'
            with self.assertRaises(ValueError):compare_pair(native,bad)

    def test_one_predeclared_first_anchor_per_task(self):
        rows=[{'task':task,'qid':str(i),'group_id':task+str(i)} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(4)]
        with patch('native_contract.select_smoke',return_value=rows):selected=select_anchors({})
        self.assertEqual([r['qid'] for r in selected],['0','0','0'])
        with patch('native_contract.select_smoke',return_value=rows[:4]):
            with self.assertRaises(ValueError):select_anchors({})

    def test_stock_native_path_strips_only_prompt_and_captures_kv(self):
        w=FakeWrapper();w.model.generation_config=types.SimpleNamespace(eos_token_id=99);w.model.lm_head=nn.Linear(4,100,bias=False)
        class Cache:
            def __init__(self,n):
                self.n=n;self.layers=[types.SimpleNamespace(keys=torch.ones((1,1,n,1)),values=torch.ones((1,1,n,1))*2) for _ in range(36)]
            def get_seq_length(self,_i):return self.n
        class LM(nn.Module):
            def __init__(self):super().__init__();self.layers=[None]*36
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None,
                        cache_position=None,past_key_values=None):
                return types.SimpleNamespace(past_key_values=Cache(inputs_embeds.shape[1]))
        w.model.model.language_model=LM();w.model.model.rope_deltas=torch.tensor([[0]])
        def generate_model(**kw):
            ids=kw['input_ids'];main,ds=w.model.visual(kw['pixel_values'])
            embeds=w.model.get_input_embeddings()(ids).clone();embeds[:,2:1026]=main
            mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,2:1026]=True
            n=ids.shape[1];positions=torch.arange(n).view(1,1,n).expand(3,1,n)
            w.model.model.language_model(inputs_embeds=embeds,attention_mask=kw['attention_mask'],position_ids=positions,
                visual_pos_masks=mask,deepstack_visual_embeds=ds,cache_position=torch.arange(n),past_key_values=None)
            w.model.lm_head(torch.ones((1,1,4)))
            return torch.cat((ids,torch.tensor([[99]])),dim=1)
        w.model.generate=generate_model
        def generate(wrapper,_row):
            inp=wrapper._processor_inputs([])
            wrapper.model.generate(**inp,do_sample=False,max_new_tokens=2048)
            return ''
        row={'task':'DocVQA_VAL','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m'}
        journal=[];stored=[]
        def archive(arrays):stored.append(set(arrays));return {'kind':'full_arrays','path':'x','sha256':'y'}
        rec=run_native(torch,w,row,generate,journal.append,archive)
        self.assertEqual(rec['output_ids'],[99]);self.assertTrue(rec['eos']);self.assertEqual(len(rec['cache_fingerprints'][0]),36)
        self.assertEqual(journal[0]['output_ids'],[99]);self.assertIn('processor_input_ids',stored[0])
        self.assertNotIn('_processor_inputs',w.__dict__)

    def test_launcher_locked_without_external_approval(self):
        with patch.object(native_launch,'APPROVAL_PATH',Path('/definitely-not-an-e4-native-approval.json')):
            with self.assertRaises(ValueError):native_launch.check_approval()

    def test_raw_array_verifier_rejects_changed_and_nonfinite_logits(self):
        tensors={'visual':torch.ones((4,2),dtype=torch.bfloat16),'prefill':torch.ones((1,5,2),dtype=torch.bfloat16),
                 'prefill_logits':torch.ones((1,8),dtype=torch.float32)}
        for i in range(3):
            tensors['ds'+str(i)]=torch.ones((4,2),dtype=torch.bfloat16)*i
            tensors['injected'+str(i)]=tensors['ds'+str(i)].clone()
        for name in ('input_ids','attention_mask','image_grid_thw'):tensors['processor_'+name]=torch.ones((1,3),dtype=torch.int64)
        tensors['processor_pixel_values']=torch.ones((4,2),dtype=torch.float32)
        record={'main_visual':tensor_meta(tensors['visual']),'prefill_embeds':tensor_meta(tensors['prefill']),
                'prefill_logits':tensor_meta(tensors['prefill_logits']),
                'full_deepstack':[tensor_meta(tensors['ds'+str(i)]) for i in range(3)],
                'injected_deepstack':[tensor_meta(tensors['injected'+str(i)]) for i in range(3)],
                'processor':{k:tensor_meta(tensors['processor_'+k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
        arrays={k:v.contiguous().view(torch.uint8).numpy().copy() for k,v in tensors.items()}
        verify_array(record,'stock_native',arrays)
        bad=copy.deepcopy(arrays);bad['ds0'][0,0]^=1
        with self.assertRaisesRegex(ValueError,'array does not match'):verify_array(record,'stock_native',bad)
        tensors['prefill_logits'][0,0]=float('inf');bad_record=copy.deepcopy(record);bad_record['prefill_logits']=tensor_meta(tensors['prefill_logits'])
        bad=copy.deepcopy(arrays);bad['prefill_logits']=tensors['prefill_logits'].contiguous().view(torch.uint8).numpy().copy()
        with self.assertRaisesRegex(ValueError,'nonfinite'):verify_array(bad_record,'stock_native',bad)

if __name__=='__main__':unittest.main()
