"""CPU-only independent attention, native model and cache checks.

Run: /home/dell/miniconda3/envs/qwen3vl_clean/bin/python -m unittest discover \
     -s Qwen_vl/scripts/attention_calibration -p test_calibration.py -v
"""
import math
import unittest

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, repeat_kv

from calibration import (AttentionLayout, AttentionFrame, BiasSpec, VisualMassObserver,
                         attention_calibration, build_logit_bias, combine_attention_mask)


def tiny_model():
    torch.manual_seed(2718)
    cfg = Qwen3VLTextConfig(vocab_size=40, hidden_size=64, intermediate_size=96,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, attention_dropout=0.0,
        rope_scaling={"rope_type":"default", "mrope_section":[2,3,3]})
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).eval()


def fixture():
    model = tiny_model()
    hidden = torch.randn(1, 7, 64)
    positions = torch.stack((torch.arange(7), torch.arange(7)*2+3, torch.arange(7)*3+7))
    layout = AttentionLayout(torch.tensor([0,0,1,1,1,0,0], dtype=torch.bool), positions,
                             rope_delta=20, key_weights=torch.tensor([1.,2.,4.]))
    return model, hidden, positions[:,None], layout


class MaskMathTests(unittest.TestCase):
    def test_hand_computed_cluster_multiplicity(self):
        # Equal key logits, a singleton text value 10 and a visual cluster of
        # three identical values 2: full attention=(10+3*2)/4, not (10+2)/2.
        frame = AttentionFrame(0,1,torch.tensor([1]),torch.zeros(3,1),torch.zeros(3,2),
            torch.tensor([False]),torch.tensor([False,True]),torch.tensor([1.,3.]),
            torch.float32,torch.device("cpu"))
        bias = build_logit_bias(frame, BiasSpec(log_count_scale=1))
        mask = combine_attention_mask(None,bias,frame.cache_position,2,torch.float32)
        q = torch.zeros(1,1,1,2)
        k = torch.zeros(1,1,2,2)
        v = torch.tensor([[[[10.,0.],[2.,0.]]]])
        out = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        self.assertAlmostEqual(float(out[0,0,0,0]),4.,places=6)

    def test_causal_and_padding_never_reopened(self):
        bias = torch.full((1,1,3,3),100.)
        combined = combine_attention_mask(None,bias,torch.arange(3),3,torch.float32)
        self.assertTrue(torch.isneginf(combined[0,0,0,1:]).all())
        supplied = torch.tensor([[[[True,False,False],[True,False,False],[True,False,True]]]])
        both = combine_attention_mask(supplied,bias,torch.arange(3),3,torch.float32)
        self.assertTrue(torch.isneginf(both[~supplied]).all())
        additive = torch.zeros(1,1,3,3).masked_fill(~supplied,torch.finfo(torch.float32).min)
        self.assertTrue(torch.equal(both,combine_attention_mask(additive,bias,torch.arange(3),3,torch.float32)))

    def test_cached_multi_query_lower_right_causality(self):
        mask=combine_attention_mask(None,torch.zeros(1,1,1,5),torch.tensor([3,4]),5,torch.float32)
        self.assertEqual(mask.shape,(1,1,2,5))
        self.assertTrue(torch.isneginf(mask[0,0,0,4]))
        self.assertTrue((mask[0,0,0,:4]==0).all())
        self.assertTrue((mask[0,0,1]==0).all())

    def test_gqa_with_head_specific_visual_bias(self):
        q=torch.zeros(1,4,1,2); k=torch.zeros(1,2,2,2)
        v=torch.tensor([[[[0.,0.],[4.,0.]],[[0.,0.],[8.,0.]]]])
        frame=AttentionFrame(1,4,torch.tensor([1]),torch.zeros(3,1),torch.zeros(3,2),
            torch.tensor([False]),torch.tensor([False,True]),torch.ones(2),torch.float32,q.device)
        spec=BiasSpec(layer_head_bias={1:[0.,math.log(3),0.,-math.log(3)]})
        mask=combine_attention_mask(None,build_logit_bias(frame,spec),frame.cache_position,2,q.dtype)
        got=torch.nn.functional.scaled_dot_product_attention(q,repeat_kv(k,2),repeat_kv(v,2),attn_mask=mask)
        torch.testing.assert_close(got[0,:,0,0],torch.tensor([2.,3.,4.,2.]))

    def test_pair_callback_native_coordinates_and_text_scope(self):
        _,_,_,layout=fixture()
        vis,w,pos=layout.arrays(8,"cpu")
        cp=torch.tensor([7])
        frame=AttentionFrame(0,4,cp,pos[:,cp],pos,vis[cp],vis,w,torch.float32,torch.device("cpu"))
        self.assertTrue(torch.equal(frame.query_positions_3d,torch.full((3,1),27)))
        spec=BiasSpec(visual_bias=2.,pair_bias=lambda f: -(f.query_positions_3d[1,:,None]-f.key_positions_3d[1,None,:]).abs().float()/10)
        got=build_logit_bias(frame,spec)
        expected=2*vis.float()-(27-pos[1]).abs()/10
        torch.testing.assert_close(got[0,0,0],expected)
        bad=BiasSpec(pair_bias=torch.zeros(2,4,1,8))
        with self.assertRaises(ValueError): build_logit_bias(frame,bad)

    def test_reject_invalid_weights(self):
        _,_,_,layout=fixture()
        with self.assertRaises(ValueError):
            AttentionLayout(layout.visual_mask,layout.positions_3d,0,torch.tensor([1.,0.,2.]))
        full=torch.ones(7); full[0]=2
        with self.assertRaises(ValueError): AttentionLayout(layout.visual_mask,layout.positions_3d,0,full)


class NativeModelTests(unittest.TestCase):
    @torch.no_grad()
    def run_prefill_decode(self,model,hidden,positions,layout,spec=None,observer=None):
        cache=DynamicCache(config=model.config)
        # Actual DeepStack adds, one per first three text layers.
        ds=[torch.randn(3,64,generator=torch.Generator().manual_seed(100+i)) for i in range(3)]
        with attention_calibration(model,layout,spec,observer):
            out=model(inputs_embeds=hidden,attention_mask=torch.ones(1,7,dtype=torch.long),
                position_ids=positions,past_key_values=cache,use_cache=True,
                visual_pos_masks=layout.visual_mask[None],deepstack_visual_embeds=ds)
            first=out.last_hidden_state.clone()
            nxt=model(input_ids=torch.tensor([[3]]),attention_mask=torch.ones(1,8,dtype=torch.long),
                position_ids=torch.full((3,1,1),27),past_key_values=cache,
                cache_position=torch.tensor([7]),use_cache=True)
        return first,nxt.last_hidden_state.clone(),cache

    def test_identity_and_read_only_observer_bit_exact_with_ds_decode(self):
        model,h,pos,layout=fixture()
        ref=self.run_prefill_decode(model,h,pos,layout)
        observer=VisualMassObserver()
        got=self.run_prefill_decode(model,h,pos,layout,BiasSpec(),observer)
        self.assertTrue(torch.equal(ref[0],got[0]))
        self.assertTrue(torch.equal(ref[1],got[1]))
        self.assertEqual(len(observer.records),6)
        for a,b in zip(ref[2].layers,got[2].layers):
            self.assertTrue(torch.equal(a.keys,b.keys))
            self.assertTrue(torch.equal(a.values,b.values))
        for r in observer.records:
            self.assertTrue(torch.equal(r["original_visual_mass"],r["corrected_visual_mass"]))
            self.assertTrue(((r["corrected_visual_mass"]>=0)&(r["corrected_visual_mass"]<=1)).all())

    @torch.no_grad()
    def test_corrected_native_attention_matches_independent_eager_and_cache(self):
        model,h,pos,layout=fixture(); attn=model.layers[0].self_attn
        normalized=model.layers[0].input_layernorm(h)
        cos,sin=model.rotary_emb(h,pos)
        cache=DynamicCache(config=model.config)
        spec=BiasSpec(log_count_scale=1.,visual_bias=.3,layer_head_bias={0:[.1,.2,.3,.4]})
        observer=VisualMassObserver(layers=[0])
        capture={}
        # Hook observes actual native projected Q/K/V and output, independently
        # of calibration's observer and bias construction.
        for name in ("q_proj","k_proj","v_proj"):
            capture[name]=[]
        handles=[getattr(attn,name).register_forward_hook(lambda m,a,o,n=name:capture[n].append(o.detach().clone())) for name in capture]
        try:
            with attention_calibration(model,layout,spec,observer):
                out,_=attn(normalized,position_embeddings=(cos,sin),attention_mask=None,
                    past_key_values=cache,cache_position=torch.arange(7))
        finally:
            for handle in handles: handle.remove()
        # Last projection invocation belongs to actual attention (observer
        # projections happened first). Independent rotary formula below.
        q=attn.q_norm(capture["q_proj"][-1].view(1,7,4,16)).transpose(1,2)
        k=attn.k_norm(capture["k_proj"][-1].view(1,7,2,16)).transpose(1,2)
        v=capture["v_proj"][-1].view(1,7,2,16).transpose(1,2)
        rotate=lambda x:torch.cat((-x[...,8:],x[...,:8]),dim=-1)
        q=q*cos[:,None]+rotate(q)*sin[:,None]
        k=k*cos[:,None]+rotate(k)*sin[:,None]
        torch.testing.assert_close(cache.layers[0].keys,k)
        logits=(q@repeat_kv(k,2).transpose(-2,-1))*attn.scaling
        expected_bias=layout.key_weights.log()[None,:]+(torch.tensor([.1,.2,.3,.4])[:,None]+.3)*layout.visual_mask[None,:]
        logits=logits+expected_bias[None,:,None,:]
        logits=logits.masked_fill(torch.ones(7,7,dtype=torch.bool).triu(1)[None,None],float("-inf"))
        probability=logits.softmax(-1)
        expected=attn.o_proj((probability@repeat_kv(v,2)).transpose(1,2).reshape(1,7,64))
        torch.testing.assert_close(out,expected,atol=2e-7,rtol=2e-5)
        torch.testing.assert_close(observer.records[0]["corrected_visual_mass"],probability[0,:,-1,layout.visual_mask].sum(-1),atol=2e-7,rtol=2e-5)

    def test_decode_bias_and_mask_control_are_finite(self):
        model,h,pos,layout=fixture()
        ref=self.run_prefill_decode(model,h,pos,layout)
        controlled=self.run_prefill_decode(model,h,pos,layout,BiasSpec(force_mask=True))
        torch.testing.assert_close(controlled[0],ref[0],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(controlled[1],ref[1],atol=1e-6,rtol=1e-5)
        changed=self.run_prefill_decode(model,h,pos,layout,BiasSpec(visual_bias=2.))
        self.assertTrue(torch.isfinite(changed[0]).all() and torch.isfinite(changed[1]).all())
        self.assertFalse(torch.equal(changed[1],ref[1]))
        self.assertEqual(changed[2].get_seq_length(),8)

    def test_hook_removed_after_failure(self):
        model,h,pos,layout=fixture()
        before=[len(layer.self_attn._forward_pre_hooks) for layer in model.layers]
        with self.assertRaises(RuntimeError):
            with attention_calibration(model,layout,BiasSpec(visual_bias=1)):
                raise RuntimeError("intentional")
        self.assertEqual(before,[len(layer.self_attn._forward_pre_hooks) for layer in model.layers])
        self.assertFalse(hasattr(model,"_attention_calibration_active"))

    def test_zero_callback_preserves_native_outputs(self):
        model,h,pos,layout=fixture()
        ref=self.run_prefill_decode(model,h,pos,layout)
        got=self.run_prefill_decode(model,h,pos,layout,BiasSpec(pair_bias=lambda f:torch.zeros(f.query_length,f.key_length)))
        self.assertTrue(torch.equal(ref[0],got[0]) and torch.equal(ref[1],got[1]))

    @torch.no_grad()
    def test_prefill_only_decode_is_exact_zero_with_mask_control(self):
        model,h,pos,layout=fixture()
        calls=[]
        def pair(frame):
            calls.append(frame.is_prefill)
            return torch.full((frame.query_length,frame.key_length),.1)
        spec=BiasSpec(log_count_scale=1,visual_bias=1,pair_bias=pair,prefill_only=True,force_mask=True)
        observer=VisualMassObserver()
        actual=self.run_prefill_decode(model,h,pos,layout,spec,observer)
        self.assertEqual(calls,[True]*3)
        for row in observer.records:
            if not row["prefill"]:
                self.assertTrue(torch.equal(row["original_visual_mass"],row["corrected_visual_mass"]))
        # Independent hybrid: same corrected prefill, ordinary zero-mask decode.
        cache=DynamicCache(config=model.config)
        ds=[torch.randn(3,64,generator=torch.Generator().manual_seed(100+i)) for i in range(3)]
        with attention_calibration(model,layout,spec):
            prefix=model(inputs_embeds=h,attention_mask=torch.ones(1,7,dtype=torch.long),
                position_ids=pos,past_key_values=cache,use_cache=True,
                visual_pos_masks=layout.visual_mask[None],deepstack_visual_embeds=ds)
        with attention_calibration(model,layout,BiasSpec(force_mask=True)):
            decode=model(input_ids=torch.tensor([[3]]),attention_mask=torch.ones(1,8,dtype=torch.long),
                position_ids=torch.full((3,1,1),27),past_key_values=cache,
                cache_position=torch.tensor([7]),use_cache=True)
        self.assertTrue(torch.equal(actual[0],prefix.last_hidden_state))
        self.assertTrue(torch.equal(actual[1],decode.last_hidden_state))


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
