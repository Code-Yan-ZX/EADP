"""Meaningful tiny-native CPU gradient/position/causality checks; no GPU used."""
import types
import unittest
from pathlib import Path
import sys

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig,Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel,Qwen3VLForConditionalGeneration

from oracle import (answer_token_ids,build_gold_batch,forward_gold,frozen_decoder,
                    score_full_streams,aggregate_scores,rank_keep,teacher_native_agreement,generate_selection)


class Tokenizer:
    eos_token_id=31
    def encode(self,value,add_special_tokens=False):
        assert not add_special_tokens
        return [int(x) for x in value.split()] if value else []
    def decode(self,values,**kwargs):return " ".join(map(str,values))


def fixture():
    torch.manual_seed(1729)
    config=Qwen3VLTextConfig(vocab_size=32,hidden_size=64,intermediate_size=96,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,
        head_dim=16,attention_dropout=0.,
        rope_scaling={"rope_type":"default","mrope_section":[2,3,3]})
    config._attn_implementation="sdpa"
    text=Qwen3VLTextModel(config)
    model=torch.nn.Module();model.text=text;model.lm_head=torch.nn.Linear(64,32,bias=False)
    model.eval()
    positions=torch.stack((torch.tensor([0,1,2,2,2,2,2,2,2,2,10,11]),
        torch.tensor([0,1,2,2,3,3,4,4,5,5,10,11]),
        torch.tensor([0,1,2,3,2,3,2,3,2,3,10,11])))[:,None]
    eng=types.SimpleNamespace(model=model,text=text,tok=Tokenizer(),
        inner=types.SimpleNamespace(get_input_embeddings=lambda:text.embed_tokens),
        positions_full=lambda prep:(positions,torch.tensor([[0]])))
    prep=dict(ids=torch.arange(12)[None],n_vis=8,img_start=2,img_end=10)
    V=torch.randn(8,64)*.1
    DS=[torch.randn(8,64)*.1 for _ in range(3)]
    return eng,prep,V,DS


def native_fixture():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from model.native_qwen3 import NativeEngine
    torch.manual_seed(3141)
    text=dict(vocab_size=40,hidden_size=64,intermediate_size=96,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,attention_dropout=0.,
        eos_token_id=0,rope_scaling={"rope_type":"default","mrope_section":[2,3,3]})
    vision=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
        spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,
        deepstack_visual_indexes=[0,1,2],num_position_embeddings=16)
    config=Qwen3VLConfig(text_config=text,vision_config=vision,image_token_id=31,
        video_token_id=32,vision_start_token_id=30,vision_end_token_id=29)
    config._attn_implementation="sdpa"
    model=Qwen3VLForConditionalGeneration(config).eval()
    eng=NativeEngine(types.SimpleNamespace(model=model,processor=types.SimpleNamespace(tokenizer=Tokenizer())))
    prep=dict(ids=torch.tensor([[5,30]+[31]*8+[29,7]]),n_vis=8,img_start=2,img_end=10,
        gthw=torch.tensor([[1,4,8]]))
    return eng,prep,torch.randn(8,64)*.1,[torch.randn(8,64)*.1 for _ in range(3)]


class OracleCPU(unittest.TestCase):
    def test_complete_gold_and_eos_are_explicit(self):
        tokenizer=Tokenizer()
        self.assertEqual(answer_token_ids(tokenizer,"4 7 9").tolist(),[4,7,9])
        self.assertEqual(answer_token_ids(tokenizer,"4 7 9",append_eos=True).tolist(),[4,7,9,31])
        with self.assertRaises(ValueError):answer_token_ids(tokenizer,"")
        with self.assertRaises(ValueError):answer_token_ids(tokenizer,"4 7 9",max_answer_tokens=2)

    def test_prediction_positions_and_original_native_coordinates(self):
        eng,prep,V,DS=fixture()
        batch=build_gold_batch(eng,prep,V,DS,torch.tensor([4,7,9]),torch.tensor([0,3,5,7]))
        self.assertEqual(batch.prompt_length,8)
        self.assertEqual(batch.prediction_positions.tolist(),[7,8,9])
        self.assertEqual(batch.inputs["inputs_embeds"].shape[1],10)
        self.assertEqual(batch.inputs["position_ids"][:,:,-2:].tolist(),[[[12,13]]]*3)
        self.assertEqual([x.shape[0] for x in batch.inputs["deepstack_visual_embeds"]],[4]*3)
        torch.testing.assert_close(batch.inputs["inputs_embeds"][0,batch.visual_slots],V[[0,3,5,7]])
        self.assertFalse(batch.inputs["use_cache"])

    def test_first_answer_logit_matches_prompt_and_future_is_invisible(self):
        eng,prep,V,DS=fixture()
        with frozen_decoder(eng),torch.no_grad():
            a=build_gold_batch(eng,prep,V,DS,torch.tensor([4,7,9]))
            b=build_gold_batch(eng,prep,V,DS,torch.tensor([4,8,6]))
            single=build_gold_batch(eng,prep,V,DS,torch.tensor([4]))
            la,_=forward_gold(eng,a);lb,_=forward_gold(eng,b);ls,_=forward_gold(eng,single)
        torch.testing.assert_close(la[0],ls[0],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(la[:2],lb[:2],atol=1e-7,rtol=1e-6)
        self.assertFalse(torch.equal(la[2],lb[2]))

    def test_all_four_gradients_and_multitoken_loss(self):
        eng,prep,V,DS=fixture()
        before=[p.requires_grad for p in eng.model.parameters()]
        with torch.no_grad():
            result=score_full_streams(eng,prep,V,DS,"4 7 9",K=4,expected_visual_tokens=8)
        self.assertEqual(result["target_token_count"],3)
        self.assertEqual(result["gold_token_ids"].tolist(),[4,7,9])
        self.assertTrue((result["stream_statistics"]["gradient_l2"]>0).all())
        self.assertTrue((result["stream_statistics"]["magnitude_mass"]>0).all())
        self.assertEqual(result["magnitude_per_stream"].shape,(4,8))
        self.assertAlmostEqual(result["mean_answer_nll"],float(result["per_target_nll"].mean()),places=6)
        torch.testing.assert_close(result["scores"]["magnitude_joint"],result["magnitude_per_stream"].sum(0))
        torch.testing.assert_close(result["scores"]["signed_joint"],result["signed_per_stream"].sum(0))
        self.assertTrue(all(p.grad is None for p in eng.model.parameters()))
        self.assertEqual(before,[p.requires_grad for p in eng.model.parameters()])

    def test_shared_gate_derivative_matches_finite_difference(self):
        eng,prep,V,DS=fixture()
        targets=torch.tensor([4,7,9]);token=3;epsilon=.005
        with frozen_decoder(eng),torch.enable_grad():
            gates=torch.ones(8,requires_grad=True)
            batch=build_gold_batch(eng,prep,V*gates[:,None],[d*gates[:,None] for d in DS],targets)
            _,loss=forward_gold(eng,batch)
            derivative=torch.autograd.grad(loss,gates)[0]
        result=score_full_streams(eng,prep,V,DS,"4 7 9",K=4,expected_visual_tokens=8)
        torch.testing.assert_close(result["scores"]["signed_joint"],-derivative,atol=2e-7,rtol=1e-5)
        losses=[]
        with frozen_decoder(eng),torch.no_grad():
            for direction in (1.,-1.):
                scale=torch.ones(8);scale[token]+=direction*epsilon
                _,value=forward_gold(eng,build_gold_batch(eng,prep,V*scale[:,None],[d*scale[:,None] for d in DS],targets))
                losses.append(float(value))
        numerical=(losses[0]-losses[1])/(2*epsilon)
        self.assertAlmostEqual(float(derivative[token]),numerical,delta=1e-4)

    def test_per_token_norm_artifacts_preserve_stream_and_spatial_order(self):
        eng,prep,V,DS=fixture()
        result=score_full_streams(eng,prep,V,DS,"4 7 9",K=4,expected_visual_tokens=8)
        stats=result["stream_statistics"]
        expected=torch.stack([torch.linalg.vector_norm(value.float(),dim=-1) for value in (V,*DS)])
        torch.testing.assert_close(stats["activation_l2_per_token"],expected)
        for kind in ("activation","gradient"):
            values=stats[kind+"_l2_per_token"]
            self.assertEqual(values.shape,(4,8))
            self.assertEqual(values.device.type,"cpu")
            self.assertFalse(values.requires_grad)
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue((values>=0).all())
            torch.testing.assert_close(torch.linalg.vector_norm(values,dim=-1),stats[kind+"_l2"],atol=1e-6,rtol=1e-5)
            distributions=stats[kind+"_l2_distributions"]
            self.assertEqual(list(distributions),stats["stream_names"])
            for index,name in enumerate(stats["stream_names"]):
                dist=distributions[name]
                self.assertTrue(dist["finite"])
                self.assertEqual(dist["negative_count"],0)
                self.assertEqual(len(dist["quantiles"]),len(dist["quantile_levels"]))
                self.assertAlmostEqual(float(dist["quantiles"][0]),float(values[index].min()))
                self.assertAlmostEqual(float(dist["quantiles"][-1]),float(values[index].max()))

    def test_signed_cross_stream_cancellation_is_not_magnitude(self):
        magnitude=torch.tensor([[4.,3.,2.],[4.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        signed=torch.tensor([[4.,3.,-2.],[-4.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        scores,keep=aggregate_scores(magnitude,signed,1)
        self.assertEqual(keep["magnitude_joint"].tolist(),[0])
        self.assertEqual(keep["signed_joint"].tolist(),[1])
        self.assertEqual(scores["signed_joint"].tolist(),[0.,3.,-2.])
        self.assertEqual(rank_keep(torch.tensor([2.,2.,2.,1.]),2).tolist(),[0,1])

    def test_checkpoint_matches_eval_and_state_restores(self):
        eng,prep,V,DS=fixture()
        ordinary=score_full_streams(eng,prep,V,DS,"4 7 9",K=4,expected_visual_tokens=8)
        old_modes=[m.training for m in eng.model.modules()]
        checkpoint=score_full_streams(eng,prep,V,DS,"4 7 9",K=4,checkpointing=True,expected_visual_tokens=8)
        torch.testing.assert_close(ordinary["magnitude_per_stream"],checkpoint["magnitude_per_stream"],atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(ordinary["signed_per_stream"],checkpoint["signed_per_stream"],atol=1e-7,rtol=1e-5)
        self.assertEqual(old_modes,[m.training for m in eng.model.modules()])
        self.assertFalse(eng.text.is_gradient_checkpointing)

    def test_exception_restores_freeze_flags(self):
        eng,_,_,_=fixture()
        original=[p.requires_grad for p in eng.model.parameters()]
        with self.assertRaises(RuntimeError):
            with frozen_decoder(eng,checkpointing=True):raise RuntimeError("intentional")
        self.assertEqual(original,[p.requires_grad for p in eng.model.parameters()])
        self.assertFalse(eng.model.training)

    def test_inference_tensor_inputs_are_releafed(self):
        eng,prep,V,DS=fixture()
        with torch.inference_mode():
            V=V.clone();DS=[x.clone() for x in DS]
            result=score_full_streams(eng,prep,V,DS,"4 7",K=4,expected_visual_tokens=8)
        self.assertTrue(result["finite_gradients"])
        self.assertTrue(all(p.grad is None for p in eng.model.parameters()))

    def test_real_native_engine_forced_gold_on_off(self):
        eng,prep,V,DS=native_fixture()
        for enabled in (True,False):
            metrics=teacher_native_agreement(eng,prep,V,DS,"4 7 9",deepstack=enabled)
            self.assertLess(metrics["same_shape"]["max_abs"],1e-6)
            self.assertLess(metrics["cached"]["max_abs"],1e-6)
            self.assertLess(metrics["cached_nll_abs_difference"],1e-6)
            self.assertEqual(metrics["cache"]["actual_visual_count"],8)
            self.assertEqual(metrics["cache"]["cache_length"],metrics["cache"]["expected_cache_length"])

    def test_real_native_generation_keeps_physical_visual_rows_off(self):
        eng,prep,V,DS=native_fixture();keep=torch.tensor([0,2,4,7])
        on=generate_selection(eng,prep,V,DS,keep,True,max_tokens=4,ignore_eos=True)
        off=generate_selection(eng,prep,V,DS,keep,False,max_tokens=4,ignore_eos=True)
        self.assertEqual(on["actual_visual_count"],4)
        self.assertEqual(off["actual_visual_count"],4)
        self.assertEqual(off["native_state_n_vis"],0)
        self.assertEqual(on["positions_sha256"],off["positions_sha256"])
        self.assertEqual(on["cache_length"],off["cache_length"])
        self.assertEqual(on["decode_positions"],off["decode_positions"])


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
